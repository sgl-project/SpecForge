import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from specforge.optimizer import BF16Optimizer


def _make_optimizer(seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Linear(8, 8, bias=False)
    return model, BF16Optimizer(model, lr=1e-3, max_grad_norm=0.5)


class TestClipGradNormSingleProcess(unittest.TestCase):
    def test_matches_torch_reference(self):
        """Without distributed, clipping must match torch.nn.utils.clip_grad_norm_."""
        model, opt = _make_optimizer()
        grad = torch.randn(8, 8)

        reference = [p.detach().clone().requires_grad_(True) for p in model.parameters()]
        for rp in reference:
            rp.grad = grad.clone()
        ref_norm = torch.nn.utils.clip_grad_norm_(reference, opt.max_grad_norm)

        for mp_ in opt.fp32_params:
            mp_.grad = grad.clone()
        norm = opt._clip_grad_norm()

        torch.testing.assert_close(norm, ref_norm)
        for mp_, rp in zip(opt.fp32_params, reference):
            torch.testing.assert_close(mp_.grad, rp.grad)

    def test_step_returns_norm(self):
        model, opt = _make_optimizer()
        for p in model.parameters():
            p.grad = torch.full_like(p, 0.1)
        norm = opt.step()
        expected = torch.full((8, 8), 0.1).norm()
        torch.testing.assert_close(norm, expected)


def _dist_worker(rank, world_size, init_file, results):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        _, opt = _make_optimizer()
        # Simulate FSDP sharding: each rank holds a disjoint gradient shard
        # with a different magnitude, so local norms differ across ranks.
        grad_value = 1.0 if rank == 0 else 2.0
        for mp_ in opt.fp32_params:
            mp_.grad = torch.full_like(mp_, grad_value)
        norm = opt._clip_grad_norm()
        results[rank] = (norm.item(), opt.fp32_params[0].grad.flatten()[0].item())
    finally:
        dist.destroy_process_group()


class TestClipGradNormDistributed(unittest.TestCase):
    def test_global_norm_across_ranks(self):
        """Every rank must clip by the same GLOBAL norm, not its local norm."""
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "init")
            manager = mp.Manager()
            results = manager.dict()
            mp.spawn(
                _dist_worker,
                args=(world_size, init_file, results),
                nprocs=world_size,
                join=True,
            )

        # 64 elements of 1.0 on rank 0 + 64 elements of 2.0 on rank 1
        global_norm = (64 * 1.0**2 + 64 * 2.0**2) ** 0.5
        clip_coef = 0.5 / (global_norm + 1e-6)
        for rank, grad_value in ((0, 1.0), (1, 2.0)):
            norm, clipped_first = results[rank]
            self.assertAlmostEqual(norm, global_norm, places=4)
            self.assertAlmostEqual(clipped_first, grad_value * clip_coef, places=6)


if __name__ == "__main__":
    unittest.main()
