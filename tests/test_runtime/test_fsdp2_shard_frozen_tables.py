"""``training.shard_frozen_tables``: frozen target tables in the FSDP2 root group."""

import json
import os
import tempfile
import unittest

import torch
from torch.distributed.tensor import DTensor

from specforge.training.backend import (
    BackendOptions,
    ParallelConfig,
    create_training_backend,
)


class TestShardFrozenSelection(unittest.TestCase):
    def test_fsdp1_rejects_option(self):
        from tests.test_runtime.test_fsdp2_backend import TinyComposite

        pc = ParallelConfig(world_size=1)
        backend = create_training_backend(
            "fsdp", pc, options=BackendOptions(shard_frozen_tables=True)
        )
        with self.assertRaisesRegex(ValueError, "shard_frozen_tables"):
            backend.prepare_model(TinyComposite(), optimizer_target=None)

    def test_config_requires_fsdp2(self):
        from specforge.config.schema import TrainingConfig

        with self.assertRaisesRegex(ValueError, "shard_frozen_tables"):
            TrainingConfig(shard_frozen_tables=True)
        self.assertTrue(
            TrainingConfig(backend="fsdp2", shard_frozen_tables=True).shard_frozen_tables
        )


def _worker(rank, world_size, port, results_dir):
    from tests.test_runtime import _fixtures as fx
    from tests.test_runtime.test_fsdp2_backend import TinyComposite, _optimizer

    fx.init_rank_distributed(rank, world_size, port=str(port))
    torch.manual_seed(0)
    replicated = TinyComposite().cuda()
    sharded = TinyComposite().cuda()
    sharded.load_state_dict(replicated.state_dict())
    x = torch.randn(4, 7, device="cuda")
    out = {}
    for name, model, opts in (
        ("replicated", replicated, BackendOptions()),
        ("sharded", sharded, BackendOptions(shard_frozen_tables=True)),
    ):
        # fp32 test models: match the backend's compute dtype like PR 915's gates.
        pc = ParallelConfig.from_distributed(param_dtype=torch.float32)
        backend = create_training_backend(
            "fsdp2", pc, optimizer_factory=_optimizer, options=opts
        )
        wrapped = backend.prepare_model(model, optimizer_target=model.draft_model)
        table_is_dtensor = isinstance(wrapped.lm_head.weight, DTensor)
        losses = []
        for _ in range(3):
            loss = wrapped(x).float().pow(2).mean()
            backend.backward(loss, is_boundary=True)
            backend.step()
            losses.append(float(loss.detach()))
        full = backend.state_dict()["model"]
        out[name] = {
            "losses": losses,
            "table_is_dtensor": table_is_dtensor,
            "has_lm_head_key": ("lm_head.weight" in full) if full else None,
        }
    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as f:
        json.dump(out, f)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestShardFrozenDistributed(unittest.TestCase):
    def test_sharded_tables_match_replicated(self):
        import torch.multiprocessing as mp

        results_dir = tempfile.mkdtemp(prefix="shard_frozen_")
        mp.spawn(_worker, args=(2, 29619, results_dir), nprocs=2, join=True)
        for rank in range(2):
            with open(os.path.join(results_dir, f"rank{rank}.json")) as f:
                out = json.load(f)
            self.assertFalse(out["replicated"]["table_is_dtensor"])
            self.assertTrue(out["sharded"]["table_is_dtensor"])
            for a, b in zip(out["replicated"]["losses"], out["sharded"]["losses"]):
                self.assertAlmostEqual(a, b, places=5)
            if rank == 0:
                self.assertTrue(out["sharded"]["has_lm_head_key"])


if __name__ == "__main__":
    unittest.main()
