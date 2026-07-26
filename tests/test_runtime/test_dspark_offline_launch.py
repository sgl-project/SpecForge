# coding=utf-8
"""DSpark offline capability through the canonical trainer runtime."""

import os
import tempfile
import unittest

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry

CUDA = torch.cuda.is_available()
NGPU = torch.cuda.device_count() if CUDA else 0
ALGORITHM = builtin_algorithm_registry().resolve("dspark")


@unittest.skipUnless(CUDA, "DSpark offline launcher requires CUDA")
class TestDSparkOfflineLaunch(unittest.TestCase):
    def test_dspark_trains_from_precomputed_target_features(self):
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29580")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge.launch import build_offline_runtime
        from specforge.optimizer import BF16Optimizer
        from specforge.training.strategies.base import DSparkTrainStrategy

        hidden, sequence_length = 64, 12
        workdir = tempfile.mkdtemp(prefix="dspark_offline_")
        model, captured_width = fx.build_dspark(
            workdir,
            hidden=hidden,
            block_size=4,
            num_anchors=2,
            attention_backend="sdpa",
        )
        feature_dir = fx.write_offline_files_dspark(
            os.path.join(workdir, "features"),
            n=4,
            seq=sequence_length,
            captured_width=captured_width,
            target_hidden=hidden,
        )

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
            run_id="dspark-offline",
            output_dir=os.path.join(workdir, "out"),
            max_len=sequence_length,
            batch_size=1,
            num_epochs=1,
            max_steps=2,
            total_steps=2,
        )

        strategy = trainer.core.strategy
        self.assertIsInstance(strategy, DSparkTrainStrategy)
        module = strategy.trainable_module()
        self.assertIsInstance(module, FSDP)
        self.assertEqual(trainer.fit(), 2)
        self.assertTrue(all(torch.isfinite(p).all() for p in module.parameters()))


def _usp_fsdp_worker(
    rank,
    world_size,
    port,
    workdir,
    feature_dir,
    sequence_length,
    block_size,
    num_anchors,
    num_attention_heads,
    num_key_value_heads,
):
    from tests.test_runtime import _fixtures as fx

    fx.init_rank_distributed(
        rank,
        world_size,
        tp_size=1,
        sp_ulysses_size=world_size,
        sp_ring_size=1,
        port=str(port),
    )
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge.launch import build_offline_runtime
        from specforge.optimizer import BF16Optimizer

        os.environ["FSDP_SHARDING"] = "FULL_SHARD"
        rank_dir = os.path.join(workdir, f"rank-{rank}")
        os.makedirs(rank_dir)
        model, _ = fx.build_dspark(
            rank_dir,
            hidden=64,
            block_size=block_size,
            num_anchors=num_anchors,
            attention_backend="usp",
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=max(512, sequence_length + block_size),
        )
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
                total_steps=1,
            ),
            run_id="dspark-usp-fsdp",
            output_dir=os.path.join(workdir, "out"),
            max_len=sequence_length,
            batch_size=1,
            num_epochs=1,
            max_steps=1,
            total_steps=1,
            sp_ulysses_size=world_size,
            sp_ring_size=1,
            use_usp_preprocess=True,
            dataloader_num_workers=0,
        )
        module = trainer.core.strategy.trainable_module()
        if not isinstance(module, FSDP):
            raise AssertionError(type(module))
        if trainer.backend.parallel_config.sp_size != world_size:
            raise AssertionError(trainer.backend.parallel_config.sp_size)
        if trainer.fit() != 1:
            raise AssertionError("DSpark USP FSDP did not finish one optimizer step")
        if not all(
            torch.isfinite(parameter).all() for parameter in module.parameters()
        ):
            raise AssertionError("DSpark USP FSDP produced non-finite parameters")
    finally:
        from specforge.distributed import destroy_distributed

        destroy_distributed()


@unittest.skipUnless(CUDA and NGPU >= 2, "DSpark USP FSDP requires two CUDA devices")
class TestDSparkUSPFSDPLaunch(unittest.TestCase):
    def test_two_rank_full_shard_trains_from_sequence_sharded_features(self):
        import torch.multiprocessing as mp

        from tests.test_runtime import _fixtures as fx
        from tests.utils import get_available_port

        with tempfile.TemporaryDirectory(prefix="dspark-usp-fsdp-") as workdir:
            feature_dir = fx.write_offline_files_dspark(
                os.path.join(workdir, "features"),
                n=2,
                seq=13,
                captured_width=64,
                target_hidden=64,
            )
            mp.spawn(
                _usp_fsdp_worker,
                nprocs=2,
                args=(
                    2,
                    get_available_port(),
                    workdir,
                    feature_dir,
                    13,
                    4,
                    2,
                    4,
                    2,
                ),
                join=True,
            )


@unittest.skipUnless(
    os.environ.get("SPECFORGE_RUN_LONG_CONTEXT") == "1" and CUDA and NGPU >= 8,
    "set SPECFORGE_RUN_LONG_CONTEXT=1 on an eight-GPU host",
)
class TestDSparkUSP120KSmoke(unittest.TestCase):
    def test_120k_sequence_completes_one_sp8_full_shard_step(self):
        import torch.multiprocessing as mp

        from tests.test_runtime import _fixtures as fx
        from tests.utils import get_available_port

        sequence_length = 120_000
        with tempfile.TemporaryDirectory(prefix="dspark-usp-120k-") as workdir:
            feature_dir = fx.write_offline_files_dspark(
                os.path.join(workdir, "features"),
                n=1,
                seq=sequence_length,
                captured_width=64,
                target_hidden=64,
            )
            mp.spawn(
                _usp_fsdp_worker,
                nprocs=8,
                args=(
                    8,
                    get_available_port(),
                    workdir,
                    feature_dir,
                    sequence_length,
                    7,
                    512,
                    16,
                    8,
                ),
                join=True,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
