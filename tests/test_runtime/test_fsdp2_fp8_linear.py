"""``training.fp8_linear``: torchao Float8Linear blocks under the FSDP2 backend."""

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

try:
    import torchao  # noqa: F401

    HAS_TORCHAO = True
except ImportError:  # pragma: no cover
    HAS_TORCHAO = False


class WideBlock(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.up = nn.Linear(dim, 4 * dim, bias=False)
        self.down = nn.Linear(4 * dim, dim, bias=False)
        self.odd = nn.Linear(dim, 24)  # not a multiple of 16: must stay BF16

    def forward(self, x):
        return x + self.down(torch.nn.functional.silu(self.up(x))) + self.odd(x).sum(-1, keepdim=True) * 0


class WideDraft(nn.Module):
    _no_split_modules = ["WideBlock"]

    def __init__(self, dim=64):
        super().__init__()
        self.layers = nn.Sequential(WideBlock(dim), WideBlock(dim))
        self.head = nn.Linear(dim, 32, bias=False)

    def forward(self, x):
        return self.head(self.layers(x))


class WideComposite(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.draft_model = WideDraft(dim)
        self.lm_head = nn.Linear(dim, 32, bias=False).requires_grad_(False)

    def forward(self, x):
        return self.draft_model(x)


class TestFp8Selection(unittest.TestCase):
    def test_fsdp1_rejects_fp8(self):
        pc = ParallelConfig(world_size=1)
        backend = create_training_backend("fsdp", pc, options=BackendOptions(fp8_linear=True))
        with self.assertRaisesRegex(ValueError, "fp8_linear"):
            backend.prepare_model(WideComposite(), optimizer_target=None)

    def test_config_requires_fsdp2(self):
        from specforge.config.schema import TrainingConfig

        with self.assertRaisesRegex(ValueError, "fp8_linear"):
            TrainingConfig(fp8_linear=True)
        self.assertTrue(TrainingConfig(backend="fsdp2", fp8_linear=True).fp8_linear)

    def test_filter_skips_non_multiple_of_16(self):
        from specforge.training.fsdp2 import _float8_linear_filter

        self.assertTrue(_float8_linear_filter(nn.Linear(64, 256, bias=False), "up"))
        self.assertFalse(_float8_linear_filter(nn.Linear(64, 24), "odd"))
        frozen = nn.Linear(64, 256, bias=False).requires_grad_(False)
        self.assertFalse(_float8_linear_filter(frozen, "lm_head"))


def _worker(rank, world_size, port, results_dir):
    from tests.test_runtime import _fixtures as fx
    from tests.test_runtime.test_fsdp2_backend import _optimizer

    fx.init_rank_distributed(rank, world_size, port=str(port))
    torch.manual_seed(0)
    model = WideComposite().cuda().to(torch.bfloat16)
    # float8 GEMMs need every dimension (tokens included, for grad_weight) % 16 == 0
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    pc = ParallelConfig.from_distributed()
    backend = create_training_backend(
        "fsdp2", pc, optimizer_factory=_optimizer, options=BackendOptions(fp8_linear=True)
    )
    wrapped = backend.prepare_model(model, optimizer_target=model.draft_model)
    names = {type(m).__name__ for m in wrapped.modules()}
    before = [p.detach().clone() for p in wrapped.draft_model.parameters()]
    losses = []
    for _ in range(3):
        loss = wrapped(x).float().pow(2).mean()
        backend.backward(loss, is_boundary=True)
        backend.step()
        losses.append(float(loss.detach()))
    changed = any(
        not torch.equal(a.to_local() if hasattr(a, "to_local") else a,
                        b.to_local() if hasattr(b, "to_local") else b)
        for a, b in zip(before, wrapped.draft_model.parameters())
    )
    out = {
        "float8_modules": backend.fp8_linear_modules,
        "has_float8_linear": "Float8Linear" in names,
        "losses": losses,
        "finite": all(torch.isfinite(torch.tensor(losses)).tolist()),
        "changed": changed,
    }
    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as f:
        json.dump(out, f)


def _fp8_capable():
    if not (HAS_TORCHAO and torch.cuda.device_count() >= 2):
        return False
    return torch.cuda.get_device_capability(0) >= (8, 9)


@unittest.skipUnless(_fp8_capable(), "requires torchao and two sm_89+ CUDA devices")
class TestFp8Distributed(unittest.TestCase):
    def test_fp8_blocks_train(self):
        import torch.multiprocessing as mp

        results_dir = tempfile.mkdtemp(prefix="fp8_linear_")
        mp.spawn(_worker, args=(2, 29617, results_dir), nprocs=2, join=True)
        for rank in range(2):
            with open(os.path.join(results_dir, f"rank{rank}.json")) as f:
                out = json.load(f)
            self.assertEqual(out["float8_modules"], 4)  # up/down in two blocks
            self.assertTrue(out["has_float8_linear"])
            self.assertTrue(out["finite"])
            self.assertTrue(out["changed"])
            self.assertLess(out["losses"][-1], out["losses"][0])


if __name__ == "__main__":
    unittest.main()
