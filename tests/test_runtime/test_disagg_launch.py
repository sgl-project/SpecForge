# coding=utf-8
"""Disaggregated assemble path: same features -> same tensors -> same training.

Two layers of proof that an offline run and a *disaggregated* run (producer puts
features into a shared FeatureStore on a shared mount; a separate consumer trains
from them) are aligned:

* CPU, no model: the disagg store serves byte-identical tensors to the colocated
  ``LocalFeatureStore`` file:// path, and the ref manifest round-trips losslessly
  (and carries no tensors). Since ``build_disagg_eagle3_runtime`` reuses the exact
  offline trainer assembly, identical tensors => identical training.
* GPU: ``build_disagg_eagle3_runtime`` assembles + trains end to end through FSDP.
"""

import os
import tempfile
import unittest

import torch

from specforge.runtime.contracts import assert_no_tensors
from specforge.runtime.data_plane import LocalFeatureStore, OfflineManifestReader
from specforge.runtime.data_plane.disagg_ingest import (
    ingest_offline_features,
    read_ref_manifest,
    write_ref_manifest,
)
from specforge.runtime.data_plane.disaggregated import AuthPolicy, SharedDirFeatureStore
from tests.test_runtime import _fixtures as fx

CUDA = torch.cuda.is_available()


class TestDisaggDataEquivalence(unittest.TestCase):
    """The disagg transport is value-preserving (CPU, no model/FSDP)."""

    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="disagg_")
        self.feat_dir = fx.write_offline_files(
            os.path.join(self.work, "features"), n=6, seq=16, seed=0
        )

    def test_disagg_store_serves_identical_tensors(self):
        # Colocated path: OfflineManifestReader file:// refs over LocalFeatureStore.
        local = LocalFeatureStore("local")
        offline_refs = OfflineManifestReader(self.feat_dir, run_id="off").read()

        # Disagg path: ingest the same files into a SharedDirFeatureStore.
        shared = SharedDirFeatureStore(os.path.join(self.work, "store"), store_id="st")
        disagg_refs = ingest_offline_features(shared, self.feat_dir, run_id="off")

        self.assertEqual(len(offline_refs), len(disagg_refs))
        self.assertTrue(
            all(r.feature_store_uri.startswith("disagg://") for r in disagg_refs)
        )

        for off_ref, dis_ref in zip(offline_refs, disagg_refs):
            off_t, off_h = local.get(off_ref)
            dis_t, dis_h = shared.get(dis_ref)
            self.assertEqual(set(off_t), set(dis_t))
            for k in off_t:
                self.assertTrue(
                    torch.equal(off_t[k], dis_t[k]),
                    f"tensor {k!r} differs between offline and disagg store",
                )
            local.release(off_h)
            shared.release(dis_h)

    def test_manifest_roundtrips_and_carries_no_tensors(self):
        shared = SharedDirFeatureStore(
            os.path.join(self.work, "store2"), store_id="st2"
        )
        refs = ingest_offline_features(shared, self.feat_dir, run_id="run7")
        manifest = os.path.join(self.work, "refs.json")
        write_ref_manifest(refs, manifest)  # asserts no-tensor internally

        loaded = read_ref_manifest(manifest)
        assert_no_tensors(loaded)  # control plane carries metadata only
        self.assertEqual(len(loaded), len(refs))
        for a, b in zip(refs, loaded):
            self.assertEqual(a.sample_id, b.sample_id)
            self.assertEqual(a.feature_store_uri, b.feature_store_uri)
            self.assertEqual(a.feature_keys, b.feature_keys)
            self.assertEqual(set(a.feature_specs), set(b.feature_specs))
            for k in a.feature_specs:
                self.assertEqual(a.feature_specs[k].shape, b.feature_specs[k].shape)
                self.assertEqual(a.feature_specs[k].dtype, b.feature_specs[k].dtype)
            self.assertEqual(a.strategy, b.strategy)
            self.assertEqual(
                a.metadata.get("target_repr"), b.metadata.get("target_repr")
            )

        # a consumer can fetch the reconstructed refs from the shared store
        t, h = shared.get(loaded[0])
        self.assertIn("hidden_state", t)
        shared.release(h)

    def test_retain_on_release_keeps_features_for_reiteration(self):
        # Offline training re-iterates the ref set across epochs, so the disagg
        # store must NOT consume-once-free on release (mirrors LocalFeatureStore's
        # file:// no-op release). Without retain, epoch 2 get() would KeyError.
        store = SharedDirFeatureStore(
            os.path.join(self.work, "store_ro"), store_id="ro", retain_on_release=True
        )
        refs = ingest_offline_features(store, self.feat_dir, run_id="ro")
        for _epoch in range(3):
            for r in refs:
                t, h = store.get(r)
                self.assertIn("hidden_state", t)
                store.release(h)  # retained: file kept for the next epoch
        self.assertEqual(store.health()["resident_samples"], len(refs))

    def test_default_release_is_consume_once(self):
        # Online default (retain_on_release=False) still frees on last release.
        store = SharedDirFeatureStore(
            os.path.join(self.work, "store_co"), store_id="co"
        )
        refs = ingest_offline_features(store, self.feat_dir, run_id="co")
        _t, h = store.get(refs[0])
        store.release(h)
        with self.assertRaises(KeyError):
            store.get(refs[0])

    def test_auth_required_consumer_must_present_token(self):
        root = os.path.join(self.work, "store3")
        producer = SharedDirFeatureStore(
            root, store_id="auth", auth=AuthPolicy("s3cret"), credential="s3cret"
        )
        ingest_offline_features(producer, self.feat_dir, run_id="a")
        # B9: a consumer attaching with the wrong/no credential is rejected.
        with self.assertRaises(PermissionError):
            SharedDirFeatureStore(root, store_id="auth", auth=AuthPolicy("s3cret"))


@unittest.skipUnless(CUDA, "disagg launcher FSDP path requires CUDA")
class TestDisaggLaunchFSDP(unittest.TestCase):
    def test_build_disagg_runtime_trains_through_fsdp(self):
        torch.manual_seed(0)
        fx.build_single_rank_distributed(port="29577")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.launch import build_disagg_eagle3_runtime

        TTT, ACC, MAX_OPT_STEPS, N = 3, 2, 2, 8
        work = tempfile.mkdtemp(prefix="disagg_fsdp_")
        feat_dir = fx.write_offline_files(os.path.join(work, "features"), n=N)
        eagle3_model, target_head = fx.build_eagle3(work, ttt=TTT)

        # producer: ingest the files into a shared-dir store (simulates the other pool)
        store = SharedDirFeatureStore(
            os.path.join(work, "shared_store"), store_id="e2e"
        )
        refs = ingest_offline_features(store, feat_dir, run_id="e2e", max_len=512)
        manifest = os.path.join(work, "refs.json")
        write_ref_manifest(refs, manifest)

        # consumer: read the manifest + train from the shared store
        consumer_refs = read_ref_manifest(manifest)

        def optimizer_factory(draft_module):
            return BF16Optimizer(
                draft_module,
                lr=1e-3,
                max_grad_norm=0.5,
                warmup_ratio=0.0,
                total_steps=10,
            )

        trainer, loader = build_disagg_eagle3_runtime(
            feature_store=store,
            refs=consumer_refs,
            eagle3_model=eagle3_model,
            target_head=target_head,
            optimizer_factory=optimizer_factory,
            run_id="e2e",
            output_dir=os.path.join(work, "out"),
            max_len=512,
            batch_size=1,
            accumulation_steps=ACC,
            num_epochs=3,
            max_steps=MAX_OPT_STEPS,
        )

        module = trainer.core.strategy.trainable_module()
        self.assertIsInstance(
            module, FSDP, "strategy must hold the FSDP-wrapped module"
        )

        step = trainer.fit(loader)
        self.assertEqual(step, MAX_OPT_STEPS)
        self.assertEqual(trainer.micro_step, ACC * MAX_OPT_STEPS)

        ckpt = trainer.save_checkpoint(trainer.global_step)
        self.assertTrue(ckpt.checkpoint_uri.startswith("file://"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
