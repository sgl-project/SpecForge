# coding=utf-8
"""FeatureDataLoader: queue + store -> TrainBatch, with injected transform/collate (CPU)."""

import os
import tempfile
import unittest
from dataclasses import replace

import torch

from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue


def _offline_eagle3_process_data(raw):
    """Mirror of OfflineEagle3Dataset.process_data (the aux<->target swap)."""
    max_len = 2048
    hidden_state = raw["aux_hidden_state"].squeeze(0)[:max_len][None, :]
    target = raw["hidden_state"].squeeze(0)[:max_len][None, :]
    input_ids = raw["input_ids"][:max_len][None, :]
    loss_mask = raw["loss_mask"][:max_len][None, :].clone()
    loss_mask[0, -1] = 0
    return {
        "attention_mask": torch.ones_like(loss_mask, dtype=torch.long),
        "loss_mask": loss_mask,
        "target": target,
        "hidden_state": hidden_state,
        "input_ids": input_ids,
    }


def _simple_collate(features):
    keys = features[0].keys()
    return {k: torch.cat([f[k] for f in features], dim=0) for k in keys}


class _CountingStore(LocalFeatureStore):
    """LocalFeatureStore that counts get() calls (seek must not materialize)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gets = 0

    def get(self, sample_ref, *, device="cpu", names=None):
        self.gets += 1
        return super().get(sample_ref, device=device, names=names)


class TestFeatureDataLoader(unittest.TestCase):
    def _write_offline_files(self, d, n=4, seq=8, h=4, aux=12):
        for i in range(n):
            torch.save(
                {
                    "input_ids": torch.arange(seq) + i,
                    "loss_mask": torch.ones(seq, dtype=torch.long),
                    "hidden_state": torch.randn(1, seq, h),
                    "aux_hidden_state": torch.randn(1, seq, aux),
                },
                os.path.join(d, f"{i:03d}.ckpt"),
            )

    def test_offline_loader_emits_trainbatch(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=4)
            # data-plane unit test: drive the loader from the queue primitive
            # directly (no control plane needed here).
            q = SampleRefQueue()
            q.put(OfflineManifestReader(d, run_id="run").read())
            store = LocalFeatureStore("st")
            loader = FeatureDataLoader(
                store,
                q,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=_offline_eagle3_process_data,
            )
            batches = list(loader)
            self.assertEqual(len(batches), 2)  # 4 samples / batch 2
            b = batches[0]
            self.assertEqual(len(b.sample_ids), 2)
            self.assertEqual(b.tensors["input_ids"].shape, (2, 8))
            self.assertEqual(b.tensors["target"].shape, (2, 8, 4))
            self.assertEqual(b.tensors["hidden_state"].shape, (2, 8, 12))
            # aux<->target swap preserved
            self.assertEqual(b.metadata["target_repr"], "hidden_state")
            # all refs acked
            self.assertEqual(q.in_flight(), 0)
            self.assertEqual(q.depth(), 0)

    def test_drop_last(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=3)
            refs = OfflineManifestReader(d, run_id="run").read()
            store = LocalFeatureStore("st")
            loader = FeatureDataLoader(
                store,
                refs=refs,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=_offline_eagle3_process_data,
                drop_last=True,
            )
            batches = list(loader)
            self.assertEqual(len(batches), 1)  # 3 samples, drop the trailing 1

    def test_queue_partial_batch_fails_terminally(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=3)
            q = SampleRefQueue()
            q.put(OfflineManifestReader(d, run_id="run").read())
            loader = FeatureDataLoader(
                LocalFeatureStore("st"),
                q,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=_offline_eagle3_process_data,
                drop_last=True,
            )
            with self.assertRaisesRegex(RuntimeError, "incomplete batch"):
                list(loader)
            self.assertEqual(q.depth(), 0)
            self.assertEqual(q.in_flight(), 0)

    def test_mixed_target_repr_fails_and_releases_refs(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=2)
            refs = OfflineManifestReader(d, run_id="run").read()
            refs[1] = replace(
                refs[1],
                metadata={**refs[1].metadata, "target_repr": "logits"},
            )
            q = SampleRefQueue()
            q.put(refs)
            loader = FeatureDataLoader(
                LocalFeatureStore("st"),
                q,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=_offline_eagle3_process_data,
            )
            with self.assertRaises(ValueError):
                list(loader)
            self.assertEqual(q.in_flight(), 0)
            self.assertEqual(q.depth(), 0)

    def test_refs_mode_is_reiterable_across_epochs(self):
        # offline: a fixed ref set must re-iterate every epoch (no epoch-drain).
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=4)
            refs = OfflineManifestReader(d, run_id="run").read()
            loader = FeatureDataLoader(
                LocalFeatureStore("st"),
                refs=refs,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=_offline_eagle3_process_data,
            )
            epoch1 = [b.sample_ids for b in loader]
            epoch2 = [b.sample_ids for b in loader]
            self.assertEqual(len(epoch1), 2)
            self.assertEqual(epoch1, epoch2)  # same fixed set each epoch

    def test_seek_repositions_next_pass_only(self):
        # resume: seek(k) skips the first k batches of the NEXT pass without
        # materializing them, then later epochs iterate in full again.
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=6)
            refs = OfflineManifestReader(d, run_id="run").read()
            loader = FeatureDataLoader(
                LocalFeatureStore("st"),
                refs=refs,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=_offline_eagle3_process_data,
            )
            full = [b.sample_ids for b in loader]
            self.assertEqual(len(full), 3)
            loader.seek(2)
            self.assertEqual([b.sample_ids for b in loader], full[2:])
            # one-shot: consumed by that pass, the next epoch is full again
            self.assertEqual([b.sample_ids for b in loader], full)
            # a queue stream has no position to restore
            q = SampleRefQueue()
            qloader = FeatureDataLoader(
                LocalFeatureStore("st2"), q, collate_fn=_simple_collate
            )
            with self.assertRaises(ValueError):
                qloader.seek(1)

    def test_seek_past_end_raises(self):
        # a resume position beyond the dataset must fail loudly, not yield a
        # silent empty epoch; skip == available (fully consumed epoch) is fine.
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=6)
            refs = OfflineManifestReader(d, run_id="run").read()
            loader = FeatureDataLoader(
                LocalFeatureStore("st"),
                refs=refs,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=_offline_eagle3_process_data,
            )
            with self.assertRaisesRegex(ValueError, "skips past the end of the data"):
                loader.seek(4)  # 6 refs / batch 2 = 3 batches
            loader.seek(3)  # boundary: allowed, next pass yields nothing
            self.assertEqual(list(loader), [])
            self.assertEqual(len(list(loader)), 3)  # one-shot, epoch after is full

    def test_seek_bound_counts_partial_batch_without_drop_last(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=5)
            refs = OfflineManifestReader(d, run_id="run").read()
            loader = FeatureDataLoader(
                LocalFeatureStore("st"),
                refs=refs,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=_offline_eagle3_process_data,
                drop_last=False,
            )
            loader.seek(3)  # ceil(5/2) = 3: the trailing partial batch counts
            with self.assertRaisesRegex(ValueError, "skips past the end of the data"):
                loader.seek(4)

    def test_seek_does_not_materialize_skipped_refs(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_offline_files(d, n=6)
            refs = OfflineManifestReader(d, run_id="run").read()
            store = _CountingStore("st")
            transform_calls = []

            def counting_transform(raw):
                transform_calls.append(1)
                return _offline_eagle3_process_data(raw)

            loader = FeatureDataLoader(
                store,
                refs=refs,
                batch_size=2,
                collate_fn=_simple_collate,
                per_sample_transform=counting_transform,
            )
            loader.seek(2)
            batches = list(loader)
        self.assertEqual(len(batches), 1)
        # only the yielded batch's 2 refs are fetched/transformed
        self.assertEqual(store.gets, 2)
        self.assertEqual(len(transform_calls), 2)

    def test_requires_exactly_one_source(self):
        store = LocalFeatureStore("st")
        with self.assertRaises(ValueError):
            FeatureDataLoader(store)  # neither queue nor refs
        with self.assertRaises(ValueError):
            FeatureDataLoader(store, SampleRefQueue(), refs=[])  # both


if __name__ == "__main__":
    unittest.main(verbosity=2)
