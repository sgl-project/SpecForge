# coding=utf-8
"""DFlash offline capability through the canonical strategy-neutral trainer."""

import os
import tempfile
import unittest

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry

CUDA = torch.cuda.is_available()
ALGORITHM = builtin_algorithm_registry().resolve("dflash")


@unittest.skipUnless(CUDA, "DFlash offline launcher requires CUDA")
class TestDFlashOfflineLaunch(unittest.TestCase):
    def test_dflash_trains_from_precomputed_features(self):
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29567")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge.launch import build_offline_runtime
        from specforge.optimizer import BF16Optimizer

        hidden, sequence_length = 64, 32
        workdir = tempfile.mkdtemp(prefix="dflash_offline_")
        feature_dir = fx.write_offline_files_dflash(
            os.path.join(workdir, "features"),
            n=4,
            seq=sequence_length,
            hidden=hidden,
        )
        model, width, _target_dir, _layers = fx.build_dflash(
            workdir,
            hidden=hidden,
            block_size=4,
            num_anchors=8,
            attention_backend="sdpa",
        )
        self.assertEqual(width, hidden)

        def optimizer_factory(module):
            return BF16Optimizer(
                module,
                lr=1e-3,
                max_grad_norm=0.5,
                warmup_ratio=0.0,
                total_steps=2,
            )

        trainer = build_offline_runtime(
            algorithm=ALGORITHM,
            hidden_states_path=feature_dir,
            draft_model=model,
            target_head=None,
            optimizer_factory=optimizer_factory,
            run_id="dflash-offline",
            output_dir=os.path.join(workdir, "out"),
            max_len=sequence_length,
            batch_size=1,
            num_epochs=1,
            max_steps=2,
        )

        module = trainer.core.strategy.trainable_module()
        self.assertIsInstance(module, FSDP)
        self.assertEqual(trainer.fit(), 2)
        self.assertTrue(all(torch.isfinite(p).all() for p in module.parameters()))

    def test_enabled_length_bucketing_trains_variable_length_legacy_dflash_features(
        self,
    ):
        """Exercise the canonical cache/sampler/loader path with plain DFlash.

        The shared fixture is DFlash, not DFlash2. DFlash2 numerical parity is
        covered separately by the production-model length-aware benchmark.
        """
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29593")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge.launch import build_offline_runtime
        from specforge.optimizer import BF16Optimizer
        from specforge.training.checkpoint import STATE_FILE

        with tempfile.TemporaryDirectory(prefix="dflash_length_bucket_") as workdir:
            model, captured_width, _target_dir, _layers = fx.build_dflash(
                workdir,
                hidden=64,
                block_size=4,
                num_anchors=4,
                attention_backend="sdpa",
            )
            feature_dir = os.path.join(workdir, "features")
            for index, length in enumerate((12, 60, 20, 44)):
                fx.write_offline_files_dflash(
                    os.path.join(feature_dir, f"sample-{index}"),
                    n=1,
                    seq=length,
                    hidden=captured_width,
                    seed=index,
                )
            output_dir = os.path.join(workdir, "out")
            trainer = build_offline_runtime(
                algorithm=ALGORITHM,
                hidden_states_path=feature_dir,
                draft_model=model,
                target_head=None,
                optimizer_factory=lambda module: BF16Optimizer(
                    module,
                    lr=1e-3,
                    max_grad_norm=0.5,
                    warmup_ratio=0.0,
                    total_steps=2,
                ),
                run_id="dflash-length-bucket",
                output_dir=output_dir,
                max_len=64,
                batch_size=2,
                length_bucket_size=2,
                num_epochs=1,
                max_steps=2,
                seed=7,
            )
            planned = trainer._loader._refs
            self.assertEqual(
                {
                    tuple(sorted(ref.num_tokens for ref in planned[start : start + 2]))
                    for start in (0, 2)
                },
                {(12, 20), (44, 60)},
            )
            self.assertTrue(
                os.path.isfile(
                    os.path.join(
                        output_dir, "length-cache", "offline-token-lengths.json"
                    )
                )
            )
            module = trainer.core.strategy.trainable_module()
            self.assertIsInstance(module, FSDP)
            self.assertEqual(trainer.fit(), 2)
            self.assertTrue(all(torch.isfinite(p).all() for p in module.parameters()))
            state = torch.load(
                os.path.join(output_dir, "dflash-length-bucket-latest", STATE_FILE),
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(state["global_step"], 2)
            self.assertEqual(state["offline_sampler_version"], 2)
            self.assertEqual(state["length_bucket_size"], 2)
            self.assertEqual(state["length_bucket_max_len"], 64)
            self.assertEqual(len(state["length_bucket_ref_fingerprint"]), 64)


if __name__ == "__main__":
    unittest.main(verbosity=2)
