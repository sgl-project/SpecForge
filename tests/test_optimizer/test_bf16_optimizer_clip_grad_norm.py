import copy
import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from specforge.optimizer import BF16Optimizer
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig


def _make_optimizer(seed=0, **kwargs):
    torch.manual_seed(seed)
    model = torch.nn.Linear(8, 8, bias=False)
    return model, BF16Optimizer(model, lr=1e-3, max_grad_norm=0.5, **kwargs)


_ADAMW = torch.optim.AdamW


def _unfused_adamw(params, *, fused=None, **kwargs):
    del fused
    return _ADAMW(params, **kwargs)


class _LegacyBF16Optimizer(BF16Optimizer):
    """The pre-foreach step: unfused AdamW, per-parameter kernels, host syncs."""

    def __init__(self, *args, **kwargs):
        with mock.patch("torch.optim.AdamW", _unfused_adamw):
            super().__init__(*args, **kwargs)

    def step(self):
        grads = [p.grad.detach() for p in self.model_params if p.grad is not None]
        norm_sq = torch.stack([grad.float().square().sum() for grad in grads]).sum()
        grad_norm, clip_coefficient = self._reduce_grad_norm(norm_sq)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError("non-finite global grad norm")
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                master_grad = p.grad.detach().to(device=mp.device, dtype=torch.float32)
                master_grad.mul_(clip_coefficient)
                mp.grad = master_grad
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        with torch.no_grad():
            for p, mp in zip(self.model_params, self.fp32_params):
                p.data.copy_(mp.data.to(device=p.device, dtype=p.dtype))
                p.grad = None
        return grad_norm


def _run_legacy_and_current(device, *, max_grad_norm, steps=3):
    """Drive both implementations with identical bf16 params and gradients."""
    torch.manual_seed(11)
    base = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 8)
    ).to(device, torch.bfloat16)
    legacy_model, current_model = copy.deepcopy(base), copy.deepcopy(base)
    options = dict(
        lr=1e-2,
        weight_decay=0.01,
        max_grad_norm=max_grad_norm,
        total_steps=10,
        warmup_ratio=0.0,
    )
    legacy = _LegacyBF16Optimizer(legacy_model, **options)
    current = BF16Optimizer(current_model, **options)
    generator = torch.Generator(device=device).manual_seed(5)
    norms = []
    for _ in range(steps):
        for legacy_param, current_param in zip(
            legacy_model.parameters(), current_model.parameters()
        ):
            grad = torch.randn(
                legacy_param.shape, device=device, generator=generator
            ).mul_(3)
            legacy_param.grad = grad.to(torch.bfloat16)
            current_param.grad = grad.to(torch.bfloat16)
        norms.append((legacy.step(), current.step()))
    return legacy, current, norms


class TestClipGradNormSingleProcess(unittest.TestCase):
    def test_matches_torch_reference(self):
        model, optimizer = _make_optimizer()
        grad = torch.randn(8, 8)

        reference = [
            param.detach().clone().requires_grad_(True) for param in model.parameters()
        ]
        for param in reference:
            param.grad = grad.clone()
        expected_norm = torch.nn.utils.clip_grad_norm_(
            reference, optimizer.max_grad_norm
        )

        for master in optimizer.fp32_params:
            master.grad = grad.clone()
        actual_norm = optimizer._clip_grad_norm()

        torch.testing.assert_close(actual_norm, expected_norm)
        for master, expected in zip(optimizer.fp32_params, reference):
            torch.testing.assert_close(master.grad, expected.grad)

    def test_step_returns_pre_clip_norm(self):
        model, optimizer = _make_optimizer()
        for param in model.parameters():
            param.grad = torch.full_like(param, 0.1)

        norm = optimizer.step()

        expected = torch.full((8, 8), 0.1).norm()
        torch.testing.assert_close(norm, expected)

    def test_non_finite_norm_fails_before_optimizer_or_scheduler_advance(self):
        model, optimizer = _make_optimizer()
        model_before = model.weight.detach().clone()
        scheduler_epoch_before = optimizer.scheduler.last_epoch
        for param in model.parameters():
            param.grad = torch.full_like(param, float("inf"))

        with self.assertRaisesRegex(FloatingPointError, "non-finite global grad norm"):
            optimizer.step()

        torch.testing.assert_close(model.weight, model_before)
        self.assertEqual(optimizer.scheduler.last_epoch, scheduler_epoch_before)
        self.assertFalse(optimizer.optimizer.state)
        self.assertTrue(all(param.grad is None for param in optimizer.model_params))
        self.assertTrue(all(param.grad is None for param in optimizer.fp32_params))

    def test_matches_legacy_step_within_fp32_rounding_with_clipping(self):
        legacy, current, norms = _run_legacy_and_current("cpu", max_grad_norm=0.5)

        for legacy_norm, current_norm in norms:
            self.assertGreater(float(legacy_norm), 0.5)  # clipping is active
            torch.testing.assert_close(current_norm, legacy_norm, rtol=1e-6, atol=0)
        for legacy_master, current_master in zip(
            legacy.fp32_params, current.fp32_params
        ):
            torch.testing.assert_close(
                current_master, legacy_master, rtol=1e-5, atol=1e-7
            )

    def test_matches_legacy_step_bitwise_without_clipping(self):
        legacy, current, _norms = _run_legacy_and_current("cpu", max_grad_norm=1e9)

        for legacy_master, current_master in zip(
            legacy.fp32_params, current.fp32_params
        ):
            torch.testing.assert_close(current_master, legacy_master, rtol=0, atol=0)
        for legacy_param, current_param in zip(
            legacy.model_params, current.model_params
        ):
            torch.testing.assert_close(current_param, legacy_param, rtol=0, atol=0)

    def test_valid_loss_denominator_does_not_change_the_update(self):
        plain_model, plain = _make_optimizer(seed=4)
        checked_model, checked = _make_optimizer(seed=4)
        grad = torch.linspace(-2.0, 3.0, 64).reshape(8, 8)
        plain_model.weight.grad = grad.clone()
        checked_model.weight.grad = grad.clone()

        plain_norm = plain.step()
        checked_norm = checked.step(loss_denominator=torch.tensor(12.0))

        torch.testing.assert_close(checked_norm, plain_norm, rtol=0, atol=0)
        torch.testing.assert_close(
            checked_model.weight, plain_model.weight, rtol=0, atol=0
        )

    def test_invalid_loss_denominator_fails_before_any_state_changes(self):
        for denominator in (0.0, -2.0, float("nan"), float("inf")):
            with self.subTest(denominator=denominator):
                model, optimizer = _make_optimizer()
                model_before = model.weight.detach().clone()
                scheduler_epoch_before = optimizer.scheduler.last_epoch
                model.weight.grad = torch.full_like(model.weight, 0.1)

                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    optimizer.step(loss_denominator=torch.tensor(denominator))

                torch.testing.assert_close(model.weight, model_before)
                self.assertEqual(optimizer.scheduler.last_epoch, scheduler_epoch_before)
                self.assertFalse(optimizer.optimizer.state)
                self.assertIsNone(model.weight.grad)
                self.assertTrue(all(mp.grad is None for mp in optimizer.fp32_params))

    def test_loaded_fused_flag_follows_this_runs_master_device(self):
        model, source = _make_optimizer(seed=2)
        model.weight.grad = torch.full_like(model.weight, 0.05)
        source.step()
        checkpoint = copy.deepcopy(source.state_dict())
        # A checkpoint written by a fused CUDA run records fused=True and
        # device step counters; CPU masters must keep the unfused kernel.
        for group in checkpoint["optimizer_state_dict"]["param_groups"]:
            group["fused"] = True

        _target_model, target = _make_optimizer(seed=9)
        with mock.patch("specforge.optimizer.print_on_rank0"):
            target.load_state_dict(checkpoint)

        self.assertTrue(
            all(group["fused"] is None for group in target.optimizer.param_groups)
        )
        for state in target.optimizer.state.values():
            self.assertEqual(state["step"].device.type, "cpu")

    def test_backend_configures_sharded_and_replicated_optimizers(self):
        class RecordingOptimizer:
            def configure_grad_norm_reduction(self, **kwargs):
                self.config = kwargs

        process_group = object()
        backend = FSDPTrainingBackend(
            ParallelConfig(
                sharding_strategy="SHARD_GRAD_OP",
                fsdp_process_group=process_group,
            )
        )
        backend._wrapped = True
        sharded = RecordingOptimizer()
        backend.set_optimizer(sharded)
        self.assertIs(sharded.config["process_group"], process_group)
        self.assertTrue(sharded.config["enabled"])

        backend._wrapped = False
        replicated = RecordingOptimizer()
        backend.set_optimizer(replicated)
        self.assertFalse(replicated.config["enabled"])

    def test_cpu_offload_matches_resident_optimizer_update(self):
        resident_model, resident = _make_optimizer(seed=7, offload_master=False)
        offload_model, offload = _make_optimizer(seed=7, offload_master=True)
        grad = torch.linspace(-0.2, 0.3, 64).reshape(8, 8)
        resident_model.weight.grad = grad.clone()
        offload_model.weight.grad = grad.clone()

        resident_norm = resident.step()
        offload_norm = offload.step()

        torch.testing.assert_close(offload_norm, resident_norm)
        torch.testing.assert_close(offload_model.weight, resident_model.weight)
        self.assertTrue(
            all(param.device.type == "cpu" for param in offload.fp32_params)
        )
        self.assertTrue(
            all(
                tensor.device.type == "cpu"
                for state in offload.optimizer.state.values()
                for tensor in state.values()
                if isinstance(tensor, torch.Tensor)
            )
        )

    def test_resume_allows_cpu_offload_mode_change(self):
        model, resident = _make_optimizer(seed=3, offload_master=False)
        for param in model.parameters():
            param.grad = torch.full_like(param, 0.05)
        resident.step()
        checkpoint = resident.state_dict()

        # Toggling CPU offload on resume is a pure placement change and is
        # allowed: masters and Adam moments land on the current master device.
        _offload_model, offload = _make_optimizer(seed=99, offload_master=True)
        # load_state_dict logs via rank0, which needs a process group.
        created_pg = False
        if dist.is_available() and not dist.is_initialized():
            store = dist.FileStore(
                os.path.join(tempfile.mkdtemp(prefix="opt_pg_"), "store"), 1
            )
            dist.init_process_group("gloo", store=store, rank=0, world_size=1)
            created_pg = True
        try:
            offload.load_state_dict(checkpoint)
        finally:
            if created_pg:
                dist.destroy_process_group()

        for restored, saved in zip(offload.fp32_params, checkpoint["fp32_params"]):
            self.assertEqual(restored.device.type, "cpu")
            torch.testing.assert_close(restored.detach(), saved)
        self.assertTrue(
            all(
                tensor.device.type == "cpu"
                for state in offload.optimizer.state.values()
                for tensor in state.values()
                if isinstance(tensor, torch.Tensor)
            )
        )

    @unittest.skipUnless(
        torch.cuda.is_available(), "requires CUDA to exercise offload transfers"
    )
    def test_cpu_offload_gpu_model_matches_resident_update(self):
        torch.manual_seed(7)
        resident_model = torch.nn.Linear(8, 8, bias=False).cuda()
        torch.manual_seed(7)
        offload_model = torch.nn.Linear(8, 8, bias=False).cuda()
        resident = BF16Optimizer(
            resident_model, lr=1e-3, max_grad_norm=0.5, offload_master=False
        )
        offload = BF16Optimizer(
            offload_model, lr=1e-3, max_grad_norm=0.5, offload_master=True
        )
        grad = torch.linspace(-0.2, 0.3, 64).reshape(8, 8).cuda()
        resident_model.weight.grad = grad.clone()
        offload_model.weight.grad = grad.clone()

        resident_norm = resident.step()
        offload_norm = offload.step()

        # Norm is reduced on the model device (CUDA) in both cases.
        self.assertEqual(resident_norm.device.type, "cuda")
        self.assertEqual(offload_norm.device.type, "cuda")
        torch.testing.assert_close(offload_norm, resident_norm)

        # The trainable draft stays on the accelerator and is updated in place.
        self.assertEqual(offload_model.weight.device.type, "cuda")
        torch.testing.assert_close(
            offload_model.weight, resident_model.weight, atol=1e-5, rtol=1e-4
        )

        # Resident masters/Adam state live on CUDA; offloaded ones live on CPU.
        self.assertTrue(
            all(param.device.type == "cuda" for param in resident.fp32_params)
        )
        self.assertTrue(
            all(param.device.type == "cpu" for param in offload.fp32_params)
        )
        self.assertTrue(
            all(
                tensor.device.type == "cpu"
                for state in offload.optimizer.state.values()
                for tensor in state.values()
                if isinstance(tensor, torch.Tensor)
            )
        )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestBF16OptimizerCuda(unittest.TestCase):
    def test_fused_step_matches_legacy_within_fp32_rounding(self):
        legacy, current, norms = _run_legacy_and_current("cuda", max_grad_norm=0.5)

        self.assertTrue(all(group["fused"] for group in current.optimizer.param_groups))
        for legacy_norm, current_norm in norms:
            self.assertGreater(float(legacy_norm), 0.5)  # clipping is active
            torch.testing.assert_close(current_norm, legacy_norm, rtol=1e-6, atol=0)
        for legacy_master, current_master in zip(
            legacy.fp32_params, current.fp32_params
        ):
            torch.testing.assert_close(
                current_master, legacy_master, rtol=1e-5, atol=1e-6
            )

    def test_grad_norm_groups_mixed_dtype_gradients(self):
        from specforge.optimizer import _sum_of_squares

        torch.manual_seed(2)
        grads = [
            torch.randn(n, device="cuda").mul_(3).to(dtype)
            for n, dtype in (
                (257, torch.bfloat16),
                (64, torch.float32),
                (0, torch.bfloat16),
                (1031, torch.float32),
                (5, torch.bfloat16),
            )
        ]
        expected = torch.stack([g.float().square().sum() for g in grads]).sum()

        torch.testing.assert_close(_sum_of_squares(grads), expected, rtol=1e-6, atol=0)

    def test_unfused_checkpoint_resumes_on_the_fused_kernel(self):
        torch.manual_seed(1)
        model = torch.nn.Linear(8, 8, bias=False).cuda()
        with mock.patch("torch.optim.AdamW", _unfused_adamw):
            legacy = BF16Optimizer(model, lr=1e-3, max_grad_norm=0.5)
        model.weight.grad = torch.full_like(model.weight, 0.05)
        legacy.step()
        checkpoint = legacy.state_dict()
        self.assertEqual(
            next(iter(legacy.optimizer.state.values()))["step"].device.type, "cpu"
        )

        resumed_model = torch.nn.Linear(8, 8, bias=False).cuda()
        resumed = BF16Optimizer(resumed_model, lr=1e-3, max_grad_norm=0.5)
        with mock.patch("specforge.optimizer.print_on_rank0"):
            resumed.load_state_dict(checkpoint)

        self.assertTrue(all(group["fused"] for group in resumed.optimizer.param_groups))
        for state in resumed.optimizer.state.values():
            self.assertTrue(state["step"].is_cuda)
            self.assertEqual(state["step"].dtype, torch.float32)
        resumed_model.weight.grad = torch.full_like(resumed_model.weight, 0.05)
        resumed.step()
        self.assertEqual(
            float(next(iter(resumed.optimizer.state.values()))["step"]), 2.0
        )

    def test_backend_gradient_scaling_matches_per_tensor_mul_bitwise(self):
        torch.manual_seed(3)
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128), torch.nn.Linear(128, 16, bias=False)
        ).cuda()
        model[0].to(torch.bfloat16)
        backend = FSDPTrainingBackend(ParallelConfig())
        backend.prepare_model(model, wrap=False)
        for param in model.parameters():
            param.grad = torch.randn_like(param).mul_(7)
        expected = [param.grad.clone() for param in model.parameters()]
        factor = torch.tensor(8.0, device="cuda") / torch.tensor(
            1234.567, device="cuda"
        )
        for grad in expected:
            grad.mul_(factor)

        backend.scale_gradients(factor)

        for param, grad in zip(model.parameters(), expected):
            torch.testing.assert_close(param.grad, grad, rtol=0, atol=0)


def _distributed_worker(rank, world_size, init_file, results):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        _, optimizer = _make_optimizer()
        grad_value = 1.0 if rank == 0 else 2.0
        for master in optimizer.fp32_params:
            master.grad = torch.full_like(master, grad_value)

        norm = optimizer._clip_grad_norm()
        results[rank] = (
            norm.item(),
            optimizer.fp32_params[0].grad.flatten()[0].item(),
        )

        optimizer.configure_grad_norm_reduction(enabled=False)
        for master in optimizer.fp32_params:
            master.grad = torch.ones_like(master)
        replicated_norm = optimizer._clip_grad_norm()
        results[f"replicated-{rank}"] = (
            replicated_norm.item(),
            optimizer.fp32_params[0].grad.flatten()[0].item(),
        )
    finally:
        dist.destroy_process_group()


def _distributed_step_worker(rank, world_size, init_file, results):
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model, optimizer = _make_optimizer()
        grad_value = 1.0 if rank == 0 else 2.0
        for param in model.parameters():
            param.grad = torch.full_like(param, grad_value)
        # Drive the production path: step() reduces the norm on the model device.
        norm = optimizer.step()
        results[rank] = norm.item()
    finally:
        dist.destroy_process_group()


class TestClipGradNormDistributed(unittest.TestCase):
    def test_step_reduces_norm_across_ranks(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "init")
            manager = mp.Manager()
            results = manager.dict()
            mp.spawn(
                _distributed_step_worker,
                args=(world_size, init_file, results),
                nprocs=world_size,
                join=True,
            )

        global_norm = (64 * 1.0**2 + 64 * 2.0**2) ** 0.5
        for rank in range(world_size):
            self.assertAlmostEqual(results[rank], global_norm, places=4)

    def test_disjoint_shards_use_same_global_clip_coefficient(self):
        world_size = 2
        with tempfile.TemporaryDirectory() as tmpdir:
            init_file = os.path.join(tmpdir, "init")
            manager = mp.Manager()
            results = manager.dict()
            mp.spawn(
                _distributed_worker,
                args=(world_size, init_file, results),
                nprocs=world_size,
                join=True,
            )

        global_norm = (64 * 1.0**2 + 64 * 2.0**2) ** 0.5
        clip_coef = 0.5 / (global_norm + 1e-6)
        for rank, grad_value in ((0, 1.0), (1, 2.0)):
            norm, clipped_first = results[rank]
            self.assertAlmostEqual(norm, global_norm, places=4)
            self.assertAlmostEqual(clipped_first, grad_value * clip_coef, places=6)

            replicated_norm, replicated_first = results[f"replicated-{rank}"]
            self.assertAlmostEqual(replicated_norm, 8.0, places=6)
            self.assertAlmostEqual(replicated_first, 0.5 / (8.0 + 1e-6), places=6)


if __name__ == "__main__":
    unittest.main()
