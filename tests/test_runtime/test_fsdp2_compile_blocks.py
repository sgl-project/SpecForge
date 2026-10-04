"""``training.compile_blocks``: in-place block compilation ahead of FSDP2 sharding."""

import json
import os
import tempfile
import unittest

import torch
import torch.nn as nn

from specforge.training.backend import (
    BackendOptions,
    ParallelConfig,
    create_training_backend,
)


class TestCompileBlocksSelection(unittest.TestCase):
    def test_fsdp1_rejects_compile_blocks(self):
        from tests.test_runtime.test_fsdp2_backend import TinyComposite

        pc = ParallelConfig(world_size=1)
        backend = create_training_backend(
            "fsdp", pc, options=BackendOptions(compile_blocks=True)
        )
        with self.assertRaisesRegex(ValueError, "compile_blocks"):
            backend.prepare_model(TinyComposite(), optimizer_target=None)

    def test_config_requires_fsdp2(self):
        from specforge.config.schema import TrainingConfig

        with self.assertRaisesRegex(ValueError, "compile_blocks"):
            TrainingConfig(compile_blocks=True)
        cfg = TrainingConfig(backend="fsdp2", compile_blocks=True)
        self.assertTrue(cfg.compile_blocks)

    def test_block_targets_fall_back_to_midlayer(self):
        from specforge.training.backend import DistributedTrainingBackend

        class Draft(nn.Module):
            def __init__(self):
                super().__init__()
                self.midlayer = nn.Linear(3, 3)

        class Composite(nn.Module):
            def __init__(self):
                super().__init__()
                self.draft_model = Draft()

        model = Composite()
        targets = DistributedTrainingBackend._block_targets(
            model, set(), model.draft_model
        )
        self.assertEqual(targets, [model.draft_model.midlayer])


def _worker(rank, world_size, port, results_dir):
    from tests.test_runtime import _fixtures as fx
    from tests.test_runtime.test_fsdp2_backend import TinyComposite, _optimizer

    fx.init_rank_distributed(rank, world_size, port=str(port))
    torch.manual_seed(0)
    reference = TinyComposite().cuda()
    compiled = TinyComposite().cuda()
    compiled.load_state_dict(reference.state_dict())
    x = torch.randn(4, 7, device="cuda")

    losses = {}
    for name, model, opts in (
        ("eager", reference, BackendOptions()),
        ("compiled", compiled, BackendOptions(compile_blocks=True)),
    ):
        # fp32 test models: match the backend's compute dtype like PR 915's gates.
        pc = ParallelConfig.from_distributed(param_dtype=torch.float32)
        backend = create_training_backend(
            "fsdp2", pc, optimizer_factory=_optimizer, options=opts
        )
        wrapped = backend.prepare_model(model, optimizer_target=model.draft_model)
        blocks = [m for m in wrapped.modules() if type(m).__name__.endswith("TinyBlock")]
        compiled_flags = [m._compiled_call_impl is not None for m in blocks]
        step_losses = []
        for _ in range(3):
            loss = wrapped(x).float().pow(2).mean()
            backend.backward(loss, is_boundary=True)
            backend.step()
            step_losses.append(float(loss.detach()))
        losses[name] = {"losses": step_losses, "compiled": compiled_flags}
    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as f:
        json.dump(losses, f)


@unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
class TestCompileBlocksDistributed(unittest.TestCase):
    def test_compiled_blocks_match_eager(self):
        import torch.multiprocessing as mp

        results_dir = tempfile.mkdtemp(prefix="compile_blocks_")
        mp.spawn(_worker, args=(2, 29613, results_dir), nprocs=2, join=True)
        for rank in range(2):
            with open(os.path.join(results_dir, f"rank{rank}.json")) as f:
                out = json.load(f)
            self.assertTrue(all(out["compiled"]["compiled"]))
            self.assertFalse(any(out["eager"]["compiled"]))
            for a, b in zip(out["eager"]["losses"], out["compiled"]["losses"]):
                self.assertAlmostEqual(a, b, places=4)


if __name__ == "__main__":
    unittest.main()
