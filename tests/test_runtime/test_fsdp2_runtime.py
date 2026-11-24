"""The FSDP2 runtime: fp32 masters, bf16 compute, fp32 gradient accumulation."""

import copy
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from specforge.optimizer import FP32Optimizer
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.backends.fsdp2 import FSDP2TrainingBackend, _store_in_fp32
from specforge.training.contracts import PrecisionSpec
from specforge.training.program import program_spec
from specforge.training.runner import DataParallelRunner
from tests.test_algorithms.test_objective_contracts import build_runtime
from tests.test_algorithms.test_objectives import training_batch
from tests.test_runtime._fixtures import train_window

_GLOO = dist.is_available() and dist.is_gloo_available()
_FSDP2_BF16 = PrecisionSpec.resolve(runtime="fsdp2", model_dtype=torch.bfloat16)


def _optimizer_factory(module):
    return FP32Optimizer(
        module, lr=1e-3, max_grad_norm=1.0, total_steps=10, warmup_ratio=0.0
    )


def _program(dtype=torch.bfloat16):
    _, program = build_runtime("DFlashDraftModel", dtype=dtype)
    return program


def _batches(program, count, seed):
    torch.manual_seed(seed)
    config = program.draft_model.config.draft
    out = []
    for index in range(count):
        tensors = training_batch(config)
        out.append(TrainBatch([f"{seed}-{index}"], "dflash", tensors))
    return out


def _runner(program, sharding):
    backend = FSDP2TrainingBackend(
        ParallelConfig(sharding_strategy=sharding, fsdp_process_group=dist.group.WORLD),
        optimizer_factory=_optimizer_factory,
        precision=_FSDP2_BF16,
    )
    spec = program_spec(program)
    wrapped = backend.prepare_model(
        program,
        optimizer_target=program.draft_model,
        frozen=program.frozen,
        blocks=spec.partition.blocks,
    )
    return DataParallelRunner(wrapped, backend, spec), backend


def _worker(rank, world, store_path, out_dir, sharding):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world
    )
    try:
        torch.manual_seed(0)
        program = _program()
        bound = {
            name: program.draft_model.get_submodule(name).weight.dtype
            for name in program.draft_model.config.draft.target_bindings
        }
        runner, backend = _runner(program, sharding)
        report = {"bound_dtypes": bound}
        report["trainable_dtypes"] = sorted(
            {str(p.dtype) for p in program.draft_model.parameters() if p.requires_grad}
        )
        report["bound_after"] = {
            name: program.draft_model.get_submodule(name).weight.dtype for name in bound
        }
        report["bound_is_plain"] = all(
            type(program.draft_model.get_submodule(name).weight) is nn.Parameter
            for name in bound
        )
        metrics = []
        for window in range(2):
            batches = _batches(program, 2, seed=100 * window + rank)
            metrics.append(train_window(runner, batches))
        report["losses"] = [m.loss for m in metrics]
        report["grad_norms"] = [float(m.grad_norm) for m in metrics]
        state = backend.state_dict()
        # What the generator draws next: a resumed run must draw the same.
        report["rng_draw"] = torch.rand(4)
        if rank == 0:
            report["model_dtypes"] = sorted(
                {
                    str(value.dtype)
                    for name, value in state["model"].items()
                    if name.startswith("draft_model.") and value.is_floating_point()
                }
            )
        torch.save(state, os.path.join(out_dir, f"state{rank}.pt"))
        # One more window, kept to compare with a resumed run.
        batches = _batches(program, 2, seed=300 + rank)
        report["next_loss"] = train_window(runner, batches).loss
        report["next_weights"] = {
            name: value.full_tensor().detach().clone()
            for name, value in program.draft_model.named_parameters()
            if value.requires_grad
        }
        torch.save(report, os.path.join(out_dir, f"report{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _resume_worker(rank, world, store_path, out_dir, sharding):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world
    )
    try:
        torch.manual_seed(0)
        program = _program()
        saved = torch.load(os.path.join(out_dir, f"state{rank}.pt"), weights_only=False)
        shared = torch.load(os.path.join(out_dir, "state0.pt"), weights_only=False)
        # Draft weights load before the wrap, as the trainer does; the masters
        # come back with this rank's optimizer state.
        program.load_state_dict(shared["model"], strict=False)
        runner, backend = _runner(program, sharding)
        backend.load_state_dict({"optimizer": saved["optimizer"], "rng": saved["rng"]})
        report = {"rng_draw": torch.rand(4)}
        batches = _batches(program, 2, seed=300 + rank)
        report["next_loss"] = train_window(runner, batches).loss
        report["next_weights"] = {
            name: value.full_tensor().detach().clone()
            for name, value in program.draft_model.named_parameters()
            if value.requires_grad
        }
        torch.save(report, os.path.join(out_dir, f"resumed{rank}.pt"))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(_GLOO, "needs gloo")
class FSDP2RuntimeTest(unittest.TestCase):
    def _run(self, worker, directory, sharding, world=2):
        store = os.path.join(directory, f"store-{worker.__name__}")
        mp.spawn(worker, args=(world, store, directory, sharding), nprocs=world)

    def test_two_ranks_train_resume_and_write_the_draft_in_its_dtype(self):
        for sharding in ("SHARD_GRAD_OP", "NO_SHARD"):
            with self.subTest(sharding=sharding), tempfile.TemporaryDirectory() as d:
                self._run(_worker, d, sharding)
                report = torch.load(os.path.join(d, "report0.pt"), weights_only=False)
                # The trainable draft is fp32; bound target tables stay as built,
                # replicated, outside FSDP.
                self.assertEqual(report["trainable_dtypes"], ["torch.float32"])
                self.assertEqual(report["bound_after"], report["bound_dtypes"])
                self.assertTrue(report["bound_is_plain"])
                # The checkpoint's draft weights keep the draft's dtype.
                self.assertEqual(report["model_dtypes"], ["torch.bfloat16"])
                self.assertTrue(all(torch.isfinite(torch.tensor(report["losses"]))))
                # A resumed run continues bit for bit.
                self._run(_resume_worker, d, sharding)
                for rank in range(2):
                    kept = torch.load(
                        os.path.join(d, f"report{rank}.pt"), weights_only=False
                    )
                    resumed = torch.load(
                        os.path.join(d, f"resumed{rank}.pt"), weights_only=False
                    )
                    self.assertTrue(torch.equal(resumed["rng_draw"], kept["rng_draw"]))
                    self.assertEqual(resumed["next_loss"], kept["next_loss"])
                    for name, value in kept["next_weights"].items():
                        self.assertTrue(
                            torch.equal(resumed["next_weights"][name], value), name
                        )


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 5, bias=False)  # five rows: uneven on two ranks
        self.table = nn.Embedding(3, 4)  # frozen, inside a sharded block
        self.table.weight.requires_grad_(False)

    def forward(self, x):
        return self.proj(x + self.table.weight.sum(0))


class _Uneven(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _Block()
        self.head = nn.Linear(5, 3, bias=False)

    def forward(self, x):
        return self.head(self.block(x))


def _toy_backend(sharding, group=None):
    return FSDP2TrainingBackend(
        ParallelConfig(
            sharding_strategy=sharding, fsdp_process_group=group or dist.group.WORLD
        ),
        optimizer_factory=_optimizer_factory,
        precision=_FSDP2_BF16,
    )


def _wrapped_toy(sharding, weights=None):
    program = _Uneven().to(torch.bfloat16)
    if weights is not None:
        program.load_state_dict(weights)
    backend = _toy_backend(sharding)
    backend.prepare_model(program, frozen=(program.block.table,), blocks=("_Block",))
    return program, backend


def _toy_step(program, backend, x):
    backend.backward(program(x).float().pow(2).mean(), is_boundary=True)
    backend.optimizer.step()


def _full(program):
    return {
        name: value.full_tensor().detach().clone()
        for name, value in program.named_parameters()
        if value.requires_grad
    }


def _toy_worker(rank, world, store_path, out_dir, sharding):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world
    )
    try:
        # Each rank builds its own initialization: NO_SHARD must replace it with
        # rank 0's, as DDP does.
        torch.manual_seed(rank)
        program, backend = _wrapped_toy(sharding)
        report = {
            "table_is_plain": type(program.block.table.weight) is nn.Parameter,
            "replicated": backend.optimizer_state_is_replicated,
            "local": {
                name: value.to_local().detach().clone()
                for name, value in program.named_parameters()
                if value.requires_grad
            },
        }
        inputs = torch.randn(2, 2, 4, generator=torch.Generator().manual_seed(7))
        _toy_step(program, backend, inputs[0].to(torch.bfloat16))
        weights = {
            name: value.to(torch.bfloat16) for name, value in _full(program).items()
        }
        weights["block.table.weight"] = program.block.table.weight.detach().clone()
        # Copied as a checkpoint copies it: the live state keeps training.
        optimizer = copy.deepcopy(backend.optimizer.state_dict())
        _toy_step(program, backend, inputs[1].to(torch.bfloat16))
        report["continued"] = _full(program)
        # A new program from the checkpoint's weights and this rank's optimizer
        # state continues exactly, uneven shards included.
        again, resumed = _wrapped_toy(sharding, weights)
        resumed.optimizer.load_state_dict(optimizer)
        _toy_step(again, resumed, inputs[1].to(torch.bfloat16))
        report["resumed"] = _full(again)
        if sharding == "NO_SHARD":
            # Replicating over part of the job is refused.
            part = dist.new_group([0])
            if rank == 0:
                try:
                    _toy_backend(sharding, part)._mesh(torch.device("cpu"))
                except ValueError as error:
                    report["part_refused"] = str(error)
        torch.save(report, os.path.join(out_dir, f"toy{rank}.pt"))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(_GLOO, "needs gloo")
class FSDP2LayoutTest(unittest.TestCase):
    def test_uneven_shards_resume_frozen_blocks_and_aligned_replicas(self):
        for sharding in ("SHARD_GRAD_OP", "NO_SHARD"):
            with self.subTest(sharding=sharding), tempfile.TemporaryDirectory() as d:
                store = os.path.join(d, "store")
                mp.spawn(_toy_worker, args=(2, store, d, sharding), nprocs=2)
                reports = [
                    torch.load(os.path.join(d, f"toy{rank}.pt"), weights_only=False)
                    for rank in range(2)
                ]
                for report in reports:
                    self.assertTrue(report["table_is_plain"])
                    self.assertEqual(report["replicated"], sharding == "NO_SHARD")
                    for name, value in report["continued"].items():
                        self.assertTrue(torch.equal(report["resumed"][name], value), name)
                if sharding == "NO_SHARD":
                    for name, value in reports[0]["local"].items():
                        self.assertTrue(torch.equal(reports[1]["local"][name], value))
                    self.assertIn("every rank of the job", reports[0]["part_refused"])


class _Aliased(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
        self.b = self.a
        self.register_buffer("codes", torch.ones(2, dtype=torch.bfloat16))
        self.register_buffer(
            "cache", torch.ones(2, dtype=torch.bfloat16), persistent=False
        )
        self.frozen = nn.Linear(2, 2).to(torch.bfloat16).requires_grad_(False)


class StoreInFP32Test(unittest.TestCase):
    def test_aliases_stay_one_tensor_and_buffers_are_written_as_built(self):
        module = _Aliased()
        module.frozen.register_buffer("scale", torch.ones(2, dtype=torch.bfloat16))
        written = _store_in_fp32(
            module, frozen=(module.frozen,), artifact_dtype=torch.float16
        )
        self.assertIs(module.a, module.b)
        self.assertEqual(module.a.dtype, torch.float32)
        self.assertEqual(module.codes.dtype, torch.float32)
        self.assertEqual(module.cache.dtype, torch.float32)
        self.assertEqual(module.frozen.weight.dtype, torch.bfloat16)
        self.assertEqual(module.frozen.scale.dtype, torch.bfloat16)
        # Parameters leave in the artifact dtype, buffers as they were built.
        self.assertEqual(
            written,
            {"a": torch.float16, "b": torch.float16, "codes": torch.bfloat16},
        )


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(64, 64, bias=False)

    def forward(self, x):
        return self.layer(x)


@unittest.skipUnless(_GLOO, "needs gloo")
class AccumulationPrecisionTest(unittest.TestCase):
    """Micro-batch gradients accumulate in fp32, not in the compute dtype."""

    def test_window_gradient_is_the_fp32_sum_of_the_bf16_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            dist.init_process_group(
                "gloo",
                init_method=f"file://{os.path.join(directory, 'store')}",
                rank=0,
                world_size=1,
            )
            try:
                torch.manual_seed(0)
                reference = _Toy().to(torch.bfloat16)
                program = _Toy().to(torch.bfloat16)
                program.load_state_dict(reference.state_dict())
                backend = FSDP2TrainingBackend(
                    ParallelConfig(
                        sharding_strategy="SHARD_GRAD_OP",
                        fsdp_process_group=dist.group.WORLD,
                    ),
                    precision=_FSDP2_BF16,
                )
                backend.prepare_model(program)
                micro_batches = [
                    torch.randn(8, 64, dtype=torch.bfloat16) * (1 + 3 * index)
                    for index in range(32)
                ]
                exact = torch.zeros(64, 64, dtype=torch.float64)
                in_bf16 = torch.zeros(64, 64, dtype=torch.bfloat16)
                for index, x in enumerate(micro_batches):
                    # The reference computes each gradient in bf16, as the
                    # runtime does, then sums exactly and in bf16.
                    reference.zero_grad()
                    reference(x).float().pow(2).mean().backward()
                    exact += reference.layer.weight.grad.double()
                    in_bf16 += reference.layer.weight.grad
                    backend.backward(
                        program(x).float().pow(2).mean(),
                        is_boundary=index == len(micro_batches) - 1,
                    )
                grad = program.layer.weight.grad.full_tensor()
                self.assertEqual(grad.dtype, torch.float32)
                fp32_error = (grad.double() - exact).abs().max() / exact.abs().max()
                bf16_error = (in_bf16.double() - exact).abs().max() / exact.abs().max()
                self.assertLess(float(fp32_error), 1e-6)
                self.assertGreater(float(bf16_error), 100 * float(fp32_error))
            finally:
                dist.destroy_process_group()


class FSDP2PrecisionTest(unittest.TestCase):
    def test_refuses_a_precision_it_does_not_hold(self):
        reference = PrecisionSpec.resolve(runtime="fsdp", model_dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "fp32 parameters as the masters"):
            FSDP2TrainingBackend(ParallelConfig(), precision=reference)


class RuntimeStateTest(unittest.TestCase):
    def test_fsdp2_loads_no_model_state(self):
        backend = FSDP2TrainingBackend(ParallelConfig(), precision=_FSDP2_BF16)
        with self.assertRaisesRegex(ValueError, "before the wrap"):
            backend.load_state_dict({"model": {}})

    def test_each_runtime_refuses_the_other_s_optimizer_state(self):
        with self.assertRaisesRegex(ValueError, "not written by the fsdp2 runtime"):
            FP32Optimizer(nn.Linear(2, 2), lr=1e-3).load_state_dict(
                {"lr_scheduler_type": "cosine"}
            )
        reference = FSDPTrainingBackend(
            ParallelConfig(),
            precision=PrecisionSpec.resolve(runtime="fsdp", model_dtype=torch.bfloat16),
        )
        with self.assertRaisesRegex(ValueError, "resume it with training.runtime=fsdp2"):
            reference.load_state_dict({"optimizer": {FP32Optimizer.FORMAT_KEY: True}})


class FP32OptimizerTest(unittest.TestCase):
    def test_refuses_low_precision_parameters(self):
        with self.assertRaisesRegex(ValueError, "fp32 parameters"):
            FP32Optimizer(nn.Linear(2, 2).to(torch.bfloat16), lr=1e-3)

    def test_step_clips_checks_and_clears(self):
        module = nn.Linear(4, 4)
        optimizer = FP32Optimizer(module, lr=1e-2, max_grad_norm=0.1, total_steps=5)
        module(torch.randn(3, 4)).pow(2).sum().backward()
        norm = torch.cat([p.grad.flatten() for p in module.parameters()]).norm()
        returned = optimizer.step()
        self.assertAlmostEqual(float(returned), float(norm), places=5)
        self.assertTrue(all(p.grad is None for p in module.parameters()))
        module(torch.randn(3, 4)).pow(2).sum().backward()
        with self.assertRaisesRegex(ValueError, "denominator"):
            optimizer.step(loss_denominator=torch.tensor(0.0))
        self.assertTrue(all(p.grad is None for p in module.parameters()))


if __name__ == "__main__":
    unittest.main()
