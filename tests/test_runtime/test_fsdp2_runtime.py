"""The FSDP2 runtime: fp32 masters, bf16 compute, fp32 gradient accumulation,
direct registry-built optimizers/schedulers, and conformance checks."""

from __future__ import annotations

import copy
import os
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.tensor import DTensor

from specforge.config import Config
from specforge.optimizer import (
    _OPTIMIZERS,
    FSDP2_OPTIMIZER_FORMAT_KEY,
    BF16Optimizer,
    local_tensor,
    register_optimizer,
)
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.backends.fsdp2 import FSDP2TrainingBackend, _store_in_fp32
from specforge.training.contracts import (
    OptimizerSpec,
    PrecisionSpec,
    SchedulerSpec,
    StepContext,
)
from specforge.training.frame import run_window
from specforge.training.program import program_spec
from specforge.training.runner import DataParallelRunner
from tests.test_algorithms.test_objective_contracts import build_runtime
from tests.test_algorithms.test_objectives import training_batch
from tests.test_runtime._fixtures import train_window
from tests.test_runtime.conformance import (
    KitProgram,
    ProtocolChecks,
    batch,
    reference_runner,
)

_GLOO = dist.is_available() and dist.is_gloo_available()
_FSDP2_BF16 = PrecisionSpec.resolve(runtime="fsdp2", model_dtype=torch.bfloat16)
_FSDP2_FP32 = PrecisionSpec.resolve(runtime="fsdp2", model_dtype=torch.float32)


def _opt_spec(**overrides):
    defaults = dict(kind="adamw", lr=1e-3, weight_decay=0.0)
    defaults.update(overrides)
    return OptimizerSpec(**defaults)


def _sched_spec(**overrides):
    defaults = dict(kind="cosine", total_steps=10, warmup_ratio=0.0)
    defaults.update(overrides)
    return SchedulerSpec(**defaults)


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


def _runner(program, sharding, *, opt_spec=None, sched_spec=None, max_grad_norm=1.0):
    backend = FSDP2TrainingBackend(
        ParallelConfig(sharding_strategy=sharding, fsdp_process_group=dist.group.WORLD),
        optimizer_spec=opt_spec or _opt_spec(),
        scheduler_spec=sched_spec or _sched_spec(),
        max_grad_norm=max_grad_norm,
        precision=_FSDP2_BF16,
    )
    spec = program_spec(program)
    wrapped = backend.prepare_model(
        program,
        optimizer_target=program.draft_model,
        frozen_modules=program.frozen_modules,
        blocks=spec.partition.blocks,
    )
    return DataParallelRunner(wrapped, backend, spec), backend


def _worker(rank, world, store_path, out_dir, sharding):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world
    )
    backend = None
    try:
        torch.manual_seed(0)
        program = _program()
        bound = {
            name: program.draft_model.get_submodule(name).weight.dtype
            for name in program.draft_model.config.draft.target_bindings
        }
        runner, backend = _runner(program, sharding)
        report = {"bound_dtypes": bound}
        report["optimizer_type"] = type(backend.optimizer).__name__
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
        if backend is not None:
            backend.close()
        dist.destroy_process_group()


def _resume_worker(rank, world, store_path, out_dir, sharding):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world
    )
    backend = None
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
        if backend is not None:
            backend.close()
        dist.destroy_process_group()


@unittest.skipUnless(_GLOO, "needs gloo")
class FSDP2RuntimeTest(unittest.TestCase):
    def _run(self, worker, directory, sharding, world=2):
        store = os.path.join(directory, f"store-{worker.__name__}")
        mp.spawn(worker, args=(world, store, directory, sharding), nprocs=world)

    def test_two_ranks_train_resume_and_write_the_draft_in_its_dtype(self):
        for sharding in ("SHARD_GRAD_OP", "FULL_SHARD", "NO_SHARD"):
            with self.subTest(sharding=sharding), tempfile.TemporaryDirectory() as d:
                self._run(_worker, d, sharding)
                report = torch.load(os.path.join(d, "report0.pt"), weights_only=False)
                # Drives PyTorch AdamW directly — no separate FP32Optimizer wrapper.
                self.assertEqual(report["optimizer_type"], "AdamW")
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


def _toy_backend(sharding, group=None, *, opt_spec=None, sched_spec=None):
    return FSDP2TrainingBackend(
        ParallelConfig(
            sharding_strategy=sharding, fsdp_process_group=group or dist.group.WORLD
        ),
        optimizer_spec=opt_spec or _opt_spec(),
        scheduler_spec=sched_spec or _sched_spec(),
        max_grad_norm=1.0,
        precision=_FSDP2_BF16,
    )


def _wrapped_toy(sharding, weights=None, *, opt_spec=None, sched_spec=None):
    program = _Uneven().to(torch.bfloat16)
    if weights is not None:
        program.load_state_dict(weights)
    backend = _toy_backend(sharding, opt_spec=opt_spec, sched_spec=sched_spec)
    backend.prepare_model(
        program, frozen_modules=(program.block.table,), blocks=("_Block",)
    )
    return program, backend


def _toy_step(program, backend, x):
    backend.backward(program(x).float().pow(2).mean(), is_boundary=True)
    backend.step()


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
    backend = None
    resumed = None
    try:
        # Each rank builds its own initialization: NO_SHARD must replace it with
        # rank 0's, as DDP does.
        torch.manual_seed(rank)
        program, backend = _wrapped_toy(sharding)
        report = {
            "table_is_plain": type(program.block.table.weight) is nn.Parameter,
            "replicated": backend.optimizer_state_is_replicated,
            "owned_groups_before_close": len(backend._owned_groups),
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
        saved = copy.deepcopy(backend.state_dict())
        _toy_step(program, backend, inputs[1].to(torch.bfloat16))
        report["continued"] = _full(program)
        # A new program from the checkpoint's weights and this rank's optimizer
        # state continues exactly, uneven shards included.
        again, resumed = _wrapped_toy(sharding, weights)
        resumed.load_state_dict({"optimizer": saved["optimizer"], "rng": saved["rng"]})
        _toy_step(again, resumed, inputs[1].to(torch.bfloat16))
        report["resumed"] = _full(again)
        # close() destroys NO_SHARD mesh subgroups idempotently without harming WORLD.
        backend.close()
        backend.close()
        report["owned_groups_after_close"] = len(backend._owned_groups)
        report["world_still_alive"] = dist.is_initialized()
        if sharding == "NO_SHARD":
            # Replicating over part of the job is refused.
            part = dist.new_group([0])
            if rank == 0:
                try:
                    _toy_backend(sharding, part)._mesh(torch.device("cpu"))
                except ValueError as error:
                    report["part_refused"] = str(error)
            dist.destroy_process_group(part)
        torch.save(report, os.path.join(out_dir, f"toy{rank}.pt"))
    finally:
        if resumed is not None:
            resumed.close()
        if backend is not None:
            backend.close()
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
                    self.assertEqual(report["owned_groups_after_close"], 0)
                    self.assertTrue(report["world_still_alive"])
                    for name, value in report["continued"].items():
                        self.assertTrue(
                            torch.equal(report["resumed"][name], value), name
                        )
                if sharding == "NO_SHARD":
                    self.assertGreater(reports[0]["owned_groups_before_close"], 0)
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
            module, frozen_modules=(module.frozen,), artifact_dtype=torch.float16
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


class FSDP2PrecisionAndConfigTest(unittest.TestCase):
    def test_refuses_a_precision_it_does_not_hold(self):
        reference = PrecisionSpec.resolve(runtime="fsdp", model_dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "fp32 parameters as the masters"):
            FSDP2TrainingBackend(ParallelConfig(), precision=reference)
        with self.assertRaisesRegex(ValueError, "fp32 floating buffers"):
            FSDP2TrainingBackend(
                ParallelConfig(), precision=replace(_FSDP2_BF16, buffer_dtype=None)
            )

    def test_refuses_wrap_false_and_loading_model_state(self):
        backend = FSDP2TrainingBackend(ParallelConfig(), precision=_FSDP2_BF16)
        with self.assertRaisesRegex(ValueError, "requires wrap=True"):
            backend.prepare_model(nn.Linear(2, 2), wrap=False)
        with self.assertRaisesRegex(ValueError, "before the wrap"):
            backend.load_state_dict({"model": {}})

    def test_refuses_bf16_optimizer_and_cross_runtime_optimizer_states(self):
        backend = FSDP2TrainingBackend(
            ParallelConfig(),
            optimizer_spec=_opt_spec(),
            scheduler_spec=_sched_spec(),
            max_grad_norm=1.0,
            precision=_FSDP2_BF16,
        )
        with self.assertRaisesRegex(ValueError, "master_dtype=None"):
            backend.set_optimizer(
                BF16Optimizer(nn.Linear(2, 2), lr=1e-3, total_steps=2)
            )
        backend.set_optimizer(torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=1e-3))
        with self.assertRaisesRegex(ValueError, "not written by the fsdp2 runtime"):
            backend._load_optimizer_state_dict({"lr_scheduler_type": "cosine"})

        from specforge.training.contracts import RankFailure

        reference = FSDPTrainingBackend(
            ParallelConfig(),
            precision=PrecisionSpec.resolve(runtime="fsdp", model_dtype=torch.bfloat16),
        )
        with self.assertRaisesRegex(
            RankFailure, "resume it with training.runtime=fsdp2"
        ) as raised:
            reference.load_state_dict({"optimizer": {FSDP2_OPTIMIZER_FORMAT_KEY: True}})
        self.assertIsInstance(raised.exception.__cause__, ValueError)

    def test_config_refuses_fsdp2_with_cpu_offload_or_usp(self):
        base = {
            "model": {"target_model_path": "target"},
            "data": {"hidden_states_path": "features"},
            "draft": "specforge/config/draft/presets/qwen3-8b-dflash.yaml",
        }
        with self.assertRaisesRegex(ValueError, "optimizer_cpu_offload"):
            Config.model_validate(
                {
                    **base,
                    "training": {"runtime": "fsdp2", "optimizer_cpu_offload": True},
                }
            )
        with self.assertRaisesRegex(ValueError, "sequence parallelism"):
            Config.model_validate(
                {
                    **base,
                    "training": {
                        "runtime": "fsdp2",
                        "attention_backend": "usp",
                        "sp_ulysses_size": 2,
                    },
                }
            )

    def test_config_and_backend_gradient_dtype_opt_in(self):
        from specforge.training.assembly import _training_runtime

        base = {
            "model": {"target_model_path": "target", "torch_dtype": "bfloat16"},
            "data": {"hidden_states_path": "features"},
            "draft": "specforge/config/draft/presets/qwen3-8b-dflash.yaml",
        }
        cfg = Config.model_validate(
            {**base, "training": {"runtime": "fsdp2", "gradient_dtype": "bfloat16"}}
        )
        runtime = _training_runtime(cfg)
        self.assertEqual(runtime.precision.gradient_dtype, torch.bfloat16)
        self.assertEqual(runtime.precision.parameter_dtype, torch.float32)

        with self.assertRaisesRegex(ValueError, "training.gradient_dtype"):
            Config.model_validate(
                {**base, "training": {"runtime": "fsdp2", "gradient_dtype": "float16"}}
            )
        with self.assertRaisesRegex(ValueError, "model.torch_dtype"):
            Config.model_validate(
                {**base, "training": {"runtime": "fsdp", "gradient_dtype": "float32"}}
            )
        with self.assertRaisesRegex(ValueError, "fp32 or compute-dtype gradients"):
            FSDP2TrainingBackend(
                ParallelConfig(),
                precision=replace(_FSDP2_BF16, gradient_dtype=torch.float16),
            )

    def test_backward_reshard_strategy_and_finally_restore(self):
        for strategy, expected_reshard in (
            ("FULL_SHARD", True),
            ("SHARD_GRAD_OP", False),
            ("NO_SHARD", False),
        ):
            with self.subTest(sharding_strategy=strategy):
                backend = FSDP2TrainingBackend(
                    ParallelConfig(sharding_strategy=strategy),
                    precision=_FSDP2_BF16,
                )
                calls = []

                class _FakeModule:
                    def set_requires_gradient_sync(self, flag: bool) -> None:
                        calls.append(("sync", flag))

                    def set_reshard_after_backward(self, flag: bool) -> None:
                        calls.append(("reshard", flag))

                class _FailingLoss:
                    def backward(self) -> None:
                        calls.append(("backward", None))
                        raise RuntimeError("boom")

                backend.module = _FakeModule()
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    backend.backward(_FailingLoss(), is_boundary=False)
                self.assertEqual(
                    calls,
                    [
                        ("sync", False),
                        ("reshard", expected_reshard),
                        ("backward", None),
                        ("sync", True),
                        ("reshard", True),
                    ],
                )


class _MixedShapeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 6, bias=True)
        self.norm = nn.LayerNorm(6)

    def forward(self, x):
        return self.norm(self.proj(x))


class _MatrixAwareOptimizer(torch.optim.Optimizer):
    """Muon-style probe optimizer verifying 2-D DTensor geometry is preserved."""

    def __init__(self, param_groups, lr: float):
        super().__init__(param_groups, {"lr": lr})
        self.observed_shapes = []

    @torch.no_grad()
    def step(self, closure=None):
        del closure
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.observed_shapes.append(
                    (isinstance(p, DTensor), tuple(p.shape), p.ndim)
                )
                local_tensor(p).add_(local_tensor(p.grad), alpha=-lr)
        return None


def _registry_worker(rank, world, store_path, out_dir):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world
    )
    backend = None
    backend_b = None
    muon_backend = None
    try:
        # 1. WSD scheduler + no_decay_1d param groups + bit-exact resume on FSDP2.
        torch.manual_seed(11)
        model = _MixedShapeModel().to(torch.bfloat16)
        opt_spec = OptimizerSpec(
            kind="adamw",
            lr=1e-2,
            weight_decay=0.05,
            param_group_policy="no_decay_1d",
        )
        sched_spec = SchedulerSpec(
            kind="wsd",
            total_steps=6,
            warmup_ratio=0.0,
            min_lr_ratio=0.1,
            extra={"decay_ratio": 0.5, "decay_style": "cosine"},
        )
        backend = FSDP2TrainingBackend(
            ParallelConfig(
                sharding_strategy="SHARD_GRAD_OP", fsdp_process_group=dist.group.WORLD
            ),
            optimizer_spec=opt_spec,
            scheduler_spec=sched_spec,
            max_grad_norm=0.5,
            precision=_FSDP2_BF16,
        )
        backend.prepare_model(model)
        x = torch.randn(2, 4, generator=torch.Generator().manual_seed(3 + rank)).to(
            torch.bfloat16
        )
        _toy_step(model, backend, x)
        saved = copy.deepcopy(backend.state_dict())
        lrs_a = [backend.learning_rate()]
        for _ in range(3):
            _toy_step(model, backend, x)
            lrs_a.append(backend.learning_rate())

        torch.manual_seed(11)
        model_b = _MixedShapeModel().to(torch.bfloat16)
        backend_b = FSDP2TrainingBackend(
            ParallelConfig(
                sharding_strategy="SHARD_GRAD_OP", fsdp_process_group=dist.group.WORLD
            ),
            optimizer_spec=opt_spec,
            scheduler_spec=sched_spec,
            max_grad_norm=0.5,
            precision=_FSDP2_BF16,
        )
        backend_b.prepare_model(model_b)
        backend_b.load_state_dict(
            {"optimizer": saved["optimizer"], "rng": saved["rng"]}
        )
        lrs_b = [backend_b.learning_rate()]
        for _ in range(3):
            _toy_step(model_b, backend_b, x)
            lrs_b.append(backend_b.learning_rate())

        # 2. Custom Muon-style registered optimizer receives preserved 2-D DTensor shapes.
        with mock.patch.dict(_OPTIMIZERS, dict(_OPTIMIZERS), clear=True):
            register_optimizer(
                "muon_probe",
                lambda groups, spec: _MatrixAwareOptimizer(groups, lr=spec.lr),
            )
            torch.manual_seed(11)
            muon_model = _MixedShapeModel().to(torch.bfloat16)
            muon_backend = FSDP2TrainingBackend(
                ParallelConfig(
                    sharding_strategy="SHARD_GRAD_OP",
                    fsdp_process_group=dist.group.WORLD,
                ),
                optimizer_spec=OptimizerSpec(kind="muon_probe", lr=1e-2),
                scheduler_spec=SchedulerSpec(
                    kind="linear", total_steps=4, warmup_ratio=0.0
                ),
                max_grad_norm=1.0,
                precision=_FSDP2_BF16,
            )
            muon_backend.prepare_model(muon_model)
            _toy_step(muon_model, muon_backend, x)
            observed_shapes = list(muon_backend.optimizer.observed_shapes)

        torch.save(
            {
                "group_wds": [
                    g["weight_decay"] for g in backend.optimizer.param_groups
                ],
                "lrs_a": lrs_a,
                "lrs_b": lrs_b,
                "weights_match": all(
                    torch.equal(pa.full_tensor(), pb.full_tensor())
                    for pa, pb in zip(model.parameters(), model_b.parameters())
                ),
                "observed_shapes": observed_shapes,
            },
            os.path.join(out_dir, f"reg{rank}.pt"),
        )
    finally:
        for b in (muon_backend, backend_b, backend):
            if b is not None:
                b.close()
        dist.destroy_process_group()


@unittest.skipUnless(_GLOO, "needs gloo")
class FSDP2RegistryIntegrationTest(unittest.TestCase):
    def test_wsd_no_decay_and_custom_matrix_optimizer_on_sharded_fsdp2(self):
        with tempfile.TemporaryDirectory() as d:
            store = os.path.join(d, "store")
            mp.spawn(_registry_worker, args=(2, store, d), nprocs=2)
            for rank in range(2):
                report = torch.load(
                    os.path.join(d, f"reg{rank}.pt"), weights_only=False
                )
                self.assertEqual(report["group_wds"], [0.05, 0.0])
                self.assertEqual(report["lrs_a"], report["lrs_b"])
                self.assertLess(report["lrs_a"][-1], report["lrs_a"][0])
                self.assertTrue(report["weights_match"])
                # proj.weight arrives as a 2-D DTensor with full global shape (6, 4).
                self.assertIn((True, (6, 4), 2), report["observed_shapes"])


@unittest.skipUnless(_GLOO, "needs gloo")
class FSDP2ProtocolTest(ProtocolChecks, unittest.TestCase):
    """Conformance kit ProtocolChecks over DataParallelRunner + FSDP2TrainingBackend."""

    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        dist.init_process_group(
            "gloo",
            init_method=f"file://{os.path.join(cls._dir.name, 'store')}",
            rank=0,
            world_size=1,
        )

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        cls._dir.cleanup()

    @staticmethod
    def make_runner(program, spec):
        backend = FSDP2TrainingBackend(
            ParallelConfig(
                sharding_strategy="SHARD_GRAD_OP",
                fsdp_process_group=dist.group.WORLD,
            ),
            optimizer_spec=OptimizerSpec(kind="adamw", lr=0.05),
            scheduler_spec=SchedulerSpec(
                kind="cosine", total_steps=4, warmup_ratio=0.5
            ),
            max_grad_norm=1.0,
            precision=_FSDP2_FP32,
        )
        wrapped = backend.prepare_model(
            program,
            frozen_modules=program.frozen_modules,
            blocks=spec.partition.blocks,
        )
        return DataParallelRunner(wrapped, backend, spec)


@unittest.skipUnless(_GLOO, "needs gloo")
class FSDP2EquivalentUpdateTest(unittest.TestCase):
    """Equivalent first-update parity between FSDP2 and the reference runtime."""

    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory()
        dist.init_process_group(
            "gloo",
            init_method=f"file://{os.path.join(cls._dir.name, 'store')}",
            rank=0,
            world_size=1,
        )

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()
        cls._dir.cleanup()

    def test_window_and_microbatch_first_update_match_reference(self):
        for reduction in ("window", "microbatch"):
            for dtype, atol in ((torch.float32, 1e-6), (torch.bfloat16, 2e-2)):
                with self.subTest(reduction=reduction, dtype=dtype):
                    torch.manual_seed(17)
                    ref_prog = KitProgram(reduction=reduction, dtype=dtype)
                    fsdp2_prog = copy.deepcopy(ref_prog)
                    spec = ref_prog.spec()

                    ref_runner = reference_runner(ref_prog, spec)
                    fsdp2_backend = FSDP2TrainingBackend(
                        ParallelConfig(
                            sharding_strategy="SHARD_GRAD_OP",
                            fsdp_process_group=dist.group.WORLD,
                        ),
                        optimizer_spec=OptimizerSpec(kind="adamw", lr=0.05),
                        scheduler_spec=SchedulerSpec(
                            kind="cosine", total_steps=4, warmup_ratio=0.5
                        ),
                        max_grad_norm=1.0,
                        precision=PrecisionSpec.resolve(
                            runtime="fsdp2", model_dtype=dtype
                        ),
                    )
                    wrapped = fsdp2_backend.prepare_model(
                        fsdp2_prog,
                        frozen_modules=fsdp2_prog.frozen_modules,
                        blocks=spec.partition.blocks,
                    )
                    fsdp2_runner = DataParallelRunner(wrapped, fsdp2_backend, spec)

                    ctx = StepContext(
                        global_step=0, total_steps=4, accumulation_steps=2
                    )
                    batches = [batch(0), batch(1)]
                    torch.manual_seed(99)
                    ref_metrics = run_window(ref_runner, batches, ctx)
                    torch.manual_seed(99)
                    fsdp2_metrics = run_window(fsdp2_runner, batches, ctx)

                    self.assertAlmostEqual(
                        fsdp2_metrics.loss, ref_metrics.loss, delta=atol
                    )
                    self.assertAlmostEqual(
                        fsdp2_metrics.grad_norm, ref_metrics.grad_norm, delta=atol
                    )
                    self.assertAlmostEqual(
                        fsdp2_metrics.learning_rate,
                        ref_metrics.learning_rate,
                        places=12,
                    )
                    for (name, ref_p), (_, f2_p) in zip(
                        ref_prog.named_parameters(),
                        fsdp2_prog.named_parameters(),
                    ):
                        diff = (
                            (
                                f2_p.detach().full_tensor().float()
                                - ref_p.detach().float()
                            )
                            .abs()
                            .max()
                        )
                        self.assertLessEqual(float(diff), atol, name)


class _InnerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.proj(x)


class _OuterBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = _InnerBlock()
        self.out = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.out(self.inner(x))


class _NestedProgram(nn.Module):
    def __init__(self):
        super().__init__()
        self.root_proj = nn.Linear(4, 4, bias=False)
        self.outer = _OuterBlock()

    def forward(self, x):
        return self.outer(self.root_proj(x))


class _ConditionalBranchModel(nn.Module):
    """Parameter ``branch_b`` receives no grad on micro-batch 0 and one on 1."""

    def __init__(self):
        super().__init__()
        self.branch_a = nn.Linear(4, 4, bias=False)
        self.branch_b = nn.Linear(4, 4, bias=False)

    def forward(self, x, *, use_b: bool):
        out = self.branch_a(x)
        if use_b:
            out = out + self.branch_b(x)
        return out


def _nested_and_bf16_grad_worker(rank, world, store_path, out_dir):
    from torch.distributed.fsdp import fully_shard

    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world
    )
    try:
        # 1. Nested _no_split_modules under FULL_SHARD: bottom-up wrap succeeds,
        # blocks reshard after forward, root keeps reshard_after_forward=False,
        # and A=2 accumulation runs cleanly.
        torch.manual_seed(0)
        nested = _NestedProgram().to(torch.bfloat16)
        backend = FSDP2TrainingBackend(
            ParallelConfig(
                sharding_strategy="FULL_SHARD",
                fsdp_process_group=dist.group.WORLD,
            ),
            optimizer_spec=_opt_spec(lr=1e-2),
            scheduler_spec=_sched_spec(),
            max_grad_norm=1.0,
            precision=_FSDP2_BF16,
        )
        backend.prepare_model(
            nested,
            blocks=("_OuterBlock", "_InnerBlock"),
        )
        x0 = torch.randn(2, 4, dtype=torch.bfloat16)
        x1 = torch.randn(2, 4, dtype=torch.bfloat16)
        backend.backward(nested(x0).float().pow(2).mean(), is_boundary=False)
        backend.backward(nested(x1).float().pow(2).mean(), is_boundary=True)
        norm_full = backend.step()

        root_state = fully_shard.state(nested)
        outer_state = fully_shard.state(nested.outer)
        inner_state = fully_shard.state(nested.outer.inner)
        root_reshards = any(
            g.post_forward_mesh_info is not None for g in root_state._fsdp_param_groups
        )
        outer_reshards = all(
            g.post_forward_mesh_info is not None for g in outer_state._fsdp_param_groups
        )
        inner_reshards = all(
            g.post_forward_mesh_info is not None for g in inner_state._fsdp_param_groups
        )

        # 2. gradient_dtype=bf16 opt-in avoids torch 2.14 uniform-dtype assert
        # when a parameter receives grad only on the boundary micro-batch, and
        # still delivers fp32 grads to the fp32 DTensor masters.
        torch.manual_seed(1)
        cond = _ConditionalBranchModel().to(torch.bfloat16)
        bf16_grad_precision = PrecisionSpec.resolve(
            runtime="fsdp2",
            model_dtype=torch.bfloat16,
            gradient_dtype=torch.bfloat16,
        )
        cond_backend = FSDP2TrainingBackend(
            ParallelConfig(
                sharding_strategy="FULL_SHARD",
                fsdp_process_group=dist.group.WORLD,
            ),
            optimizer_spec=_opt_spec(lr=1e-2),
            scheduler_spec=_sched_spec(),
            max_grad_norm=1.0,
            precision=bf16_grad_precision,
        )
        cond_backend.prepare_model(cond)
        cond_backend.backward(
            cond(x0, use_b=False).float().pow(2).mean(), is_boundary=False
        )
        cond_backend.backward(
            cond(x1, use_b=True).float().pow(2).mean(), is_boundary=True
        )
        grad_dtypes = sorted(
            {str(p.grad.dtype) for p in cond.parameters() if p.grad is not None}
        )
        norm_bf16 = cond_backend.step()

        torch.save(
            {
                "norm_full_finite": bool(torch.isfinite(norm_full)),
                "root_reshards": root_reshards,
                "outer_reshards": outer_reshards,
                "inner_reshards": inner_reshards,
                "grad_dtypes": grad_dtypes,
                "norm_bf16_finite": bool(torch.isfinite(norm_bf16)),
            },
            os.path.join(out_dir, f"nested_{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(_GLOO, "needs gloo")
class FSDP2NestedBlocksAndBF16GradTest(unittest.TestCase):
    def test_nested_blocks_full_shard_and_bf16_gradient_dtype_opt_in(self):
        with tempfile.TemporaryDirectory() as d:
            store = os.path.join(d, "store")
            mp.spawn(_nested_and_bf16_grad_worker, args=(2, store, d), nprocs=2)
            for rank in range(2):
                report = torch.load(
                    os.path.join(d, f"nested_{rank}.pt"), weights_only=False
                )
                self.assertTrue(report["norm_full_finite"])
                self.assertFalse(report["root_reshards"])
                self.assertTrue(report["outer_reshards"])
                self.assertTrue(report["inner_reshards"])
                self.assertEqual(report["grad_dtypes"], ["torch.float32"])
                self.assertTrue(report["norm_bf16_finite"])

    def test_unknown_blocks_fails_before_fp32_mutation_or_mesh_creation(self):
        model = nn.Linear(4, 4, bias=False).to(torch.bfloat16)
        backend = FSDP2TrainingBackend(
            ParallelConfig(sharding_strategy="NO_SHARD"),
            precision=_FSDP2_BF16,
        )
        with self.assertRaisesRegex(ValueError, "name no module class"):
            backend.prepare_model(model, blocks=("DoesNotExist",))
        self.assertEqual(model.weight.dtype, torch.bfloat16)
        self.assertEqual(backend._owned_groups, ())


if __name__ == "__main__":
    unittest.main()
