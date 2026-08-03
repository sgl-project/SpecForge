import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from specforge.optimizer import BF16Optimizer
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig


def _make_optimizer(seed=0, **kwargs):
    torch.manual_seed(seed)
    model = torch.nn.Linear(8, 8, bias=False)
    return model, BF16Optimizer(model, lr=1e-3, max_grad_norm=0.5, **kwargs)


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

    def test_backend_captures_optimizer_metadata_before_factory_call(self):
        events = []

        class RecordingOptimizer:
            def configure_grad_norm_reduction(self, **_kwargs):
                pass

        class RecordingFactory:
            def capture_parameter_metadata(self, target):
                events.append(("capture", target.weight.shape))

            def __call__(self, target):
                events.append(("create", target.weight.shape))
                return RecordingOptimizer()

        model = torch.nn.Linear(8, 8, bias=False)
        backend = FSDPTrainingBackend(
            ParallelConfig(sharding_strategy="NO_SHARD"),
            optimizer_factory=RecordingFactory(),
        )

        backend.prepare_model(model, wrap=False, optimizer_target=model)

        self.assertEqual(
            events,
            [("capture", torch.Size([8, 8])), ("create", torch.Size([8, 8]))],
        )

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
