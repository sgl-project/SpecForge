"""Tests for shared optimizer/scheduler registries, runtime-owned updates, and grad_reduction."""

from __future__ import annotations

import copy
import math
import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from specforge.algorithms.objectives.base import ObjectiveResult
from specforge.lr_scheduler import (
    _LR_SCHEDULERS,
    FunctionalLRScheduler,
    build_lr_scheduler,
    register_lr_curve,
    registered_lr_schedulers,
)
from specforge.optimizer import (
    _OPTIMIZERS,
    _PARAM_GROUP_POLICIES,
    build_optimizer,
    build_param_groups,
    register_optimizer,
    register_param_group_policy,
    registered_optimizers,
    registered_param_group_policies,
)
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.contracts import (
    OptimizerSpec,
    PrecisionSpec,
    ProgramSpec,
    SchedulerSpec,
    StepContext,
)
from specforge.training.frame import run_window
from specforge.training.runner import DataParallelRunner


class _SmallModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 3, bias=True)
        self.norm = nn.LayerNorm(3)
        self.frozen = nn.Parameter(torch.ones(2), requires_grad=False)


class OptimizerRegistryTest(unittest.TestCase):
    def test_builtin_adamw_and_param_group_policies(self):
        self.assertIn("adamw", registered_optimizers())
        self.assertIn("default", registered_param_group_policies())
        self.assertIn("no_decay_1d", registered_param_group_policies())

        model = _SmallModel()
        default_groups = build_param_groups(
            model.named_parameters(), OptimizerSpec(weight_decay=0.1)
        )
        self.assertEqual(len(default_groups), 1)
        self.assertEqual(len(default_groups[0]["params"]), 4)  # excludes frozen

        no_decay_groups = build_param_groups(
            model.named_parameters(),
            OptimizerSpec(weight_decay=0.1, param_group_policy="no_decay_1d"),
        )
        self.assertEqual(len(no_decay_groups), 2)
        self.assertEqual(no_decay_groups[0]["weight_decay"], 0.1)
        self.assertEqual(no_decay_groups[0]["params"], [model.proj.weight])
        self.assertEqual(no_decay_groups[1]["weight_decay"], 0.0)
        self.assertEqual(
            no_decay_groups[1]["params"],
            [model.proj.bias, model.norm.weight, model.norm.bias],
        )

        opt = build_optimizer(
            model,
            OptimizerSpec(
                kind="adamw",
                lr=2e-3,
                weight_decay=0.1,
                param_group_policy="no_decay_1d",
            ),
        )
        self.assertIsInstance(opt, torch.optim.AdamW)
        self.assertEqual([g["weight_decay"] for g in opt.param_groups], [0.1, 0.0])

    def test_custom_optimizer_and_policy_registration(self):
        seen_names = []

        def custom_policy(named_params, spec, *, param_ndims):
            seen_names.extend(name for name, _ in named_params)
            return [{"params": [p for _, p in named_params], "lr": spec.lr}]

        def custom_builder(param_groups, spec):
            return torch.optim.SGD(
                param_groups, lr=spec.lr, momentum=spec.extra.get("momentum", 0.0)
            )

        with (
            mock.patch.dict(_OPTIMIZERS, dict(_OPTIMIZERS), clear=True),
            mock.patch.dict(
                _PARAM_GROUP_POLICIES, dict(_PARAM_GROUP_POLICIES), clear=True
            ),
        ):
            register_param_group_policy("test_policy", custom_policy)
            register_optimizer("test_sgd", custom_builder)
            model = _SmallModel()
            opt = build_optimizer(
                model,
                OptimizerSpec(
                    kind="test_sgd",
                    lr=0.05,
                    param_group_policy="test_policy",
                    extra={"momentum": 0.9},
                ),
            )
            self.assertIsInstance(opt, torch.optim.SGD)
            self.assertEqual(opt.param_groups[0]["momentum"], 0.9)
            self.assertEqual(
                seen_names,
                ["proj.weight", "proj.bias", "norm.weight", "norm.bias"],
            )

    def test_unknown_optimizer_and_policy_are_refused(self):
        model = _SmallModel()
        with self.assertRaisesRegex(ValueError, "unsupported optimizer"):
            build_optimizer(model, OptimizerSpec(kind="no_such_opt"))
        with self.assertRaisesRegex(ValueError, "unsupported param_group_policy"):
            build_optimizer(model, OptimizerSpec(param_group_policy="no_such_policy"))
        with self.assertRaisesRegex(ValueError, "extra options"):
            build_optimizer(model, OptimizerSpec(kind="adamw", extra={"bogus": 1}))


class SchedulerRegistryTest(unittest.TestCase):
    def _opt(self, lr: float = 1e-3):
        p = nn.Parameter(torch.ones(2))
        return torch.optim.SGD([p], lr=lr)

    def test_linear_and_cosine_min_lr_ratio(self):
        self.assertTrue(
            {"cosine", "constant", "linear", "wsd"}.issubset(
                set(registered_lr_schedulers())
            )
        )
        opt = self._opt(lr=1.0)
        sched = build_lr_scheduler(
            opt,
            SchedulerSpec(
                kind="linear", total_steps=5, warmup_ratio=0.2, min_lr_ratio=0.2
            ),
        )
        self.assertIsInstance(sched, FunctionalLRScheduler)
        # warmup=1: step 0 -> 1.0, then linear decay from 1.0 to 0.2 over 4 steps.
        lrs = [opt.param_groups[0]["lr"]]
        for _ in range(5):
            opt.step()
            sched.step()
            lrs.append(opt.param_groups[0]["lr"])
        expected = [1.0, 1.0, 0.8, 0.6, 0.4, 0.2]
        for got, want in zip(lrs, expected):
            self.assertAlmostEqual(got, want, places=6)

        opt_cos = self._opt(lr=1.0)
        sched_cos = build_lr_scheduler(
            opt_cos,
            SchedulerSpec(
                kind="cosine", total_steps=4, warmup_ratio=0.0, min_lr_ratio=0.1
            ),
        )
        for _ in range(4):
            opt_cos.step()
            sched_cos.step()
        self.assertAlmostEqual(opt_cos.param_groups[0]["lr"], 0.1, places=6)

    def test_wsd_linear_and_cosine_decay_and_resume(self):
        for style in ("linear", "cosine"):
            with self.subTest(decay_style=style):
                spec = SchedulerSpec(
                    kind="wsd",
                    total_steps=10,
                    warmup_ratio=0.2,  # 2 steps: 0.5, 1.0
                    min_lr_ratio=0.1,
                    extra={"decay_ratio": 0.4, "decay_style": style},  # decay 4 steps
                )
                opt_full = self._opt(lr=1.0)
                sched_full = build_lr_scheduler(opt_full, spec)
                full_lrs = [opt_full.param_groups[0]["lr"]]
                for step in range(10):
                    opt_full.step()
                    sched_full.step()
                    full_lrs.append(opt_full.param_groups[0]["lr"])
                    if step == 4:
                        saved = copy.deepcopy(sched_full.state_dict())

                # Step 0 (warmup): 0.5; step 1 (warmup): 1.0; steps 2..5 (stable): 1.0;
                # step 6..9 (decay): monotonically down to 0.1 at step 10.
                self.assertAlmostEqual(full_lrs[0], 0.5, places=6)
                self.assertAlmostEqual(full_lrs[1], 1.0, places=6)
                self.assertAlmostEqual(full_lrs[5], 1.0, places=6)
                self.assertLess(full_lrs[8], full_lrs[6])
                self.assertAlmostEqual(full_lrs[10], 0.1, places=6)

                # Mid-run resume reproduces the remaining curve bit for bit.
                opt_res = self._opt(lr=1.0)
                sched_res = build_lr_scheduler(opt_res, spec)
                sched_res.load_state_dict(saved)
                resumed_lrs = [opt_res.param_groups[0]["lr"]]
                for _ in range(5):
                    opt_res.step()
                    sched_res.step()
                    resumed_lrs.append(opt_res.param_groups[0]["lr"])
                self.assertEqual(resumed_lrs, full_lrs[5:])

    def test_custom_lr_curve_registration(self):
        with mock.patch.dict(_LR_SCHEDULERS, dict(_LR_SCHEDULERS), clear=True):
            register_lr_curve(
                "sqrt_decay", lambda step, _spec: 1.0 / math.sqrt(step + 1)
            )
            opt = self._opt(lr=1.0)
            sched = build_lr_scheduler(
                opt, SchedulerSpec(kind="sqrt_decay", total_steps=4, warmup_ratio=0.0)
            )
            self.assertAlmostEqual(opt.param_groups[0]["lr"], 1.0)
            opt.step()
            sched.step()
            self.assertAlmostEqual(
                opt.param_groups[0]["lr"], 1.0 / math.sqrt(2.0), places=6
            )


class RuntimeOwnedStepTest(unittest.TestCase):
    def _build_direct_backend(self, *, policy: str = "no_decay_1d"):
        torch.manual_seed(7)
        model = _SmallModel()
        opt_spec = OptimizerSpec(
            kind="adamw", lr=1e-2, weight_decay=0.05, param_group_policy=policy
        )
        sched_spec = SchedulerSpec(
            kind="wsd",
            total_steps=6,
            warmup_ratio=0.0,
            min_lr_ratio=0.1,
            extra={"decay_ratio": 0.5},
        )
        backend = FSDPTrainingBackend(
            ParallelConfig(),
            scheduler_spec=sched_spec,
            max_grad_norm=0.5,
        )
        backend.prepare_model(model, wrap=False)
        opt = build_optimizer(model, opt_spec)
        sched = build_lr_scheduler(opt, sched_spec)
        backend.set_optimizer(opt, scheduler=sched)
        return model, backend

    def test_clips_steps_clears_advances_schedule_and_round_trips_state(self):
        model, backend = self._build_direct_backend()
        self.assertTrue(backend.checks_loss_denominator)
        initial_lr = backend.learning_rate()
        self.assertAlmostEqual(initial_lr, 1e-2)

        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, 2.0)
        grad_norm = backend.step(loss_denominator=torch.tensor(4.0))
        self.assertGreater(float(grad_norm), 0.5)
        for p in model.parameters():
            if p.requires_grad:
                self.assertIsNone(p.grad)

        # Resume round-trip with multi-group no_decay_1d policy reproduces step 2.
        saved = backend.state_dict()
        model_b, backend_b = self._build_direct_backend()
        backend_b.load_state_dict(saved)
        for _ in range(3):
            for p_a, p_b in zip(model.parameters(), model_b.parameters()):
                if p_a.requires_grad:
                    grad = torch.full_like(p_a, 0.3)
                    p_a.grad = grad.clone()
                    p_b.grad = grad.clone()
            norm_a = backend.step(loss_denominator=torch.tensor(2.0))
            norm_b = backend_b.step(loss_denominator=torch.tensor(2.0))
            self.assertTrue(torch.equal(norm_a, norm_b))
            self.assertEqual(backend.learning_rate(), backend_b.learning_rate())
        for p_a, p_b in zip(model.parameters(), model_b.parameters()):
            self.assertTrue(torch.equal(p_a, p_b))

    def test_refuses_non_finite_norm_or_bad_denominator_before_state_mutation(self):
        model, backend = self._build_direct_backend()
        before_weights = [p.detach().clone() for p in model.parameters()]
        before_lr = backend.learning_rate()

        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.ones_like(p)
        with self.assertRaisesRegex(ValueError, "global loss denominator"):
            backend.step(loss_denominator=torch.tensor(0.0))
        for p, prev in zip(model.parameters(), before_weights):
            self.assertTrue(torch.equal(p, prev))
            self.assertIsNone(p.grad)
        self.assertEqual(backend.learning_rate(), before_lr)

        for p in model.parameters():
            if p.requires_grad:
                p.grad = torch.full_like(p, float("nan"))
        with self.assertRaises(FloatingPointError):
            backend.step(loss_denominator=torch.tensor(1.0))
        for p, prev in zip(model.parameters(), before_weights):
            self.assertTrue(torch.equal(p, prev))
            self.assertIsNone(p.grad)
        self.assertEqual(backend.learning_rate(), before_lr)


class _DummyProgram(nn.Module):
    def __init__(self, *, microbatch: bool = False) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        self.frozen_modules = ()
        self._microbatch = microbatch

    def host_inputs(self, tensors, *, sample_ids, epoch):
        del tensors, sample_ids, epoch
        return {}

    def forward(self, tensors, *, step, total_steps, collect_detailed_metrics, **kw):
        del step, total_steps, collect_detailed_metrics, kw
        num = self.linear(tensors["x"]).sum()
        den = torch.tensor(2.0)
        if self._microbatch:
            return ObjectiveResult(num, den, terms=((1.0, num, den),))
        return ObjectiveResult(num, den)


class GradReductionContractTest(unittest.TestCase):
    def _run_simulated_two_rank(self, grad_reduction: str, *, microbatch: bool):
        """Simulate a 2-rank reduction: ``mean`` averages grads, ``sum`` sums them."""
        torch.manual_seed(0)
        program = _DummyProgram(microbatch=microbatch)
        backend = FSDPTrainingBackend(
            ParallelConfig(world_size=2),
            optimizer_factory=lambda m: torch.optim.SGD(m.parameters(), lr=0.1),
        )
        backend.grad_reduction = grad_reduction
        backend.prepare_model(program, wrap=False)

        orig_backward = backend.backward

        def simulated_backward(loss, *, is_boundary=True):
            # Emulate two ranks carrying identical micro-batches: sum = 2x, mean = 1x.
            orig_backward(loss, is_boundary=is_boundary)
            if is_boundary and grad_reduction == "sum":
                with torch.no_grad():
                    for p in program.parameters():
                        if p.grad is not None:
                            p.grad.mul_(2.0)

        backend.backward = simulated_backward
        spec = ProgramSpec(
            name="dummy",
            inputs=("x",),
            reduction="microbatch" if microbatch else "window",
        )
        runner = DataParallelRunner(program, backend, spec)
        scales = []
        orig_scale = backend.scale_gradients

        def record_scale(factor):
            scales.append(float(factor.item()))
            orig_scale(factor)

        backend.scale_gradients = record_scale
        batch = TrainBatch(
            sample_ids=["0"],
            strategy="dummy",
            tensors={"x": torch.tensor([[1.0, 2.0]])},
        )
        with (
            mock.patch.object(runner, "_world_size", return_value=2),
            mock.patch.object(runner, "reduce_sum", side_effect=lambda v: v * 2),
        ):
            metrics = run_window(runner, [batch], StepContext(accumulation_steps=1))
        return scales, metrics.loss, program.linear.weight.detach().clone()

    def test_sum_and_mean_reductions_match_for_window_and_microbatch(self):
        for microbatch in (False, True):
            with self.subTest(microbatch=microbatch):
                mean_scales, mean_loss, mean_w = self._run_simulated_two_rank(
                    "mean", microbatch=microbatch
                )
                sum_scales, sum_loss, sum_w = self._run_simulated_two_rank(
                    "sum", microbatch=microbatch
                )
                self.assertAlmostEqual(mean_scales[0], sum_scales[0] * 2.0, places=6)
                self.assertAlmostEqual(mean_loss, sum_loss, places=6)
                self.assertTrue(torch.allclose(mean_w, sum_w, atol=1e-7, rtol=0.0))

    def test_invalid_grad_reduction_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unsupported backend grad_reduction"):
            self._run_simulated_two_rank("bogus", microbatch=False)


class CanonicalRegistryValidationTest(unittest.TestCase):
    def test_resolve_run_rejects_unknown_optimizer_policy_or_scheduler(self):
        from specforge.config import Config
        from specforge.config.draft.codec import encode_config
        from specforge.training.canonical import resolve_run
        from tests.test_config.draft_fixtures import draft_config

        base = {
            "model": {"target_model_path": "t"},
            "draft": encode_config(draft_config()),
            "data": {"hidden_states_path": "/tmp/features"},
        }
        target_cfg = {
            "hidden_size": 32,
            "num_hidden_layers": 16,
            "vocab_size": 64,
        }
        for field, value, pattern in (
            ("optimizer", "unknown_opt", "unsupported training.optimizer"),
            (
                "param_group_policy",
                "unknown_policy",
                "unsupported training.param_group_policy",
            ),
            ("lr_scheduler", "unknown_sched", "unsupported training.lr_scheduler"),
        ):
            with self.subTest(field=field):
                cfg = Config.model_validate({**base, "training": {field: value}})
                with self.assertRaisesRegex(ValueError, pattern):
                    resolve_run(cfg, target_config=target_cfg)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4, bias=True)
        self.norm = nn.LayerNorm(4)


class _DraftWithBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.root_proj = nn.Linear(4, 4, bias=True)
        self.block = _Block()


class _RootProgram(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.draft_model = _DraftWithBlock()


def _fsdp1_no_decay_worker(rank, world, store_path, out_dir):
    from torch.distributed.fsdp import FullyShardedDataParallel

    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world
    )
    original_init = FullyShardedDataParallel.__init__

    def _cpu_fsdp_init(self, module, *args, **kwargs):
        kwargs.setdefault("device_id", torch.device("cpu"))
        original_init(self, module, *args, **kwargs)

    try:
        torch.manual_seed(0)
        program = _RootProgram().to(torch.bfloat16)
        backend = FSDPTrainingBackend(
            ParallelConfig(
                world_size=world,
                sharding_strategy="SHARD_GRAD_OP",
                fsdp_process_group=dist.group.WORLD,
            ),
            optimizer_spec=OptimizerSpec(
                kind="adamw",
                lr=1e-3,
                weight_decay=0.05,
                param_group_policy="no_decay_1d",
            ),
            scheduler_spec=SchedulerSpec(kind="constant", total_steps=4),
            max_grad_norm=1.0,
            precision=PrecisionSpec.resolve(
                runtime="fsdp",
                model_dtype=torch.bfloat16,
                sharding_strategy="SHARD_GRAD_OP",
            ),
        )
        with mock.patch.object(FullyShardedDataParallel, "__init__", _cpu_fsdp_init):
            backend.prepare_model(
                program,
                optimizer_target=program.draft_model,
                blocks=("_Block",),
            )
        opt = backend.optimizer
        by_master = {
            id(mp): name
            for (name, _), mp in zip(
                [
                    (n, p)
                    for n, p in program.draft_model.named_parameters()
                    if p.requires_grad
                ],
                opt.fp32_params,
            )
        }
        groups = [
            (g["weight_decay"], [by_master[id(p)] for p in g["params"]])
            for g in opt.optimizer.param_groups
        ]
        torch.save(
            {
                "all_shards_1d": all(p.ndim == 1 for p in opt.model_params),
                "groups": groups,
            },
            os.path.join(out_dir, f"fsdp1_wd_{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


class FSDP1NoDecay1DTest(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "needs gloo")
    def test_fsdp1_preserves_pre_sharding_ndims_for_no_decay_1d(self):
        with tempfile.TemporaryDirectory() as d:
            store = os.path.join(d, "store")
            mp.spawn(_fsdp1_no_decay_worker, args=(2, store, d), nprocs=2)
            for rank in range(2):
                report = torch.load(
                    os.path.join(d, f"fsdp1_wd_{rank}.pt"), weights_only=False
                )
                # Confirms FSDP1 genuinely flattened every shard to 1-D.
                self.assertTrue(report["all_shards_1d"])
                groups = report["groups"]
                self.assertEqual([wd for wd, _ in groups], [0.05, 0.0])
                decay_names, no_decay_names = groups[0][1], groups[1][1]
                # Both root-level and auto-wrapped block 2-D weights get decay.
                self.assertEqual(len(decay_names), 2)
                self.assertTrue(
                    any(n.endswith("root_proj.weight") for n in decay_names)
                )
                self.assertTrue(any(n.endswith("linear.weight") for n in decay_names))
                # All biases and 1-D LayerNorm params get weight_decay=0.0.
                self.assertEqual(len(no_decay_names), 4)
                self.assertTrue(
                    all(n.endswith(".bias") or ".norm." in n for n in no_decay_names)
                )


def _dtensor_and_grad_norm_worker(rank, world, store_path, out_dir):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard

    from specforge.optimizer import like_parameter, local_tensor

    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world
    )
    try:
        mesh = init_device_mesh("cpu", (world,))
        # 1. local_tensor & like_parameter on uneven Shard(0) DTensor (5 rows -> 3 / 2).
        full = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        local_shard = full[:3].clone() if rank == 0 else full[3:].clone()
        dt_param = nn.Parameter(
            DTensor.from_local(
                local_shard,
                mesh,
                [Shard(0)],
                run_check=False,
                shape=full.shape,
                stride=full.stride(),
            )
        )
        plain = torch.tensor([1.0, 2.0])
        assert local_tensor(plain) is plain
        assert torch.equal(local_tensor(dt_param), local_shard)

        # Scalar tensor passes through like_parameter unchanged.
        step_scalar = torch.tensor(3.0)
        assert like_parameter(step_scalar, dt_param) is step_scalar

        # Matching local shard wraps back with global shape/stride.
        restored = like_parameter(local_shard.clone() * 2.0, dt_param)
        assert isinstance(restored, DTensor)
        assert restored.shape == full.shape
        assert restored.stride() == full.stride()
        assert torch.equal(restored.to_local(), local_shard * 2.0)

        # Mismatched local shard shape raises ValueError.
        mismatch_raised = False
        try:
            like_parameter(torch.zeros(4, 3), dt_param)
        except ValueError as exc:
            mismatch_raised = "does not match this rank's shard" in str(exc)
        assert mismatch_raised

        # 2. _global_grad_norm with rank 0 having NO gradients (DTensor AND plain sharded):
        #    must still enter collective without deadlocking and agree on rank 1's norm.
        dt_backend = FSDPTrainingBackend(
            ParallelConfig(
                world_size=world,
                sharding_strategy="SHARD_GRAD_OP",
                fsdp_process_group=dist.group.WORLD,
            ),
            max_grad_norm=1.0,
        )
        if rank == 1:
            dt_param.grad = DTensor.from_local(
                torch.full_like(local_shard, 2.0),
                mesh,
                [Shard(0)],
                run_check=False,
                shape=full.shape,
                stride=full.stride(),
            )
        else:
            dt_param.grad = None
        dt_norm = float(dt_backend._global_grad_norm([dt_param]).item())

        plain_p = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        plain_backend = FSDPTrainingBackend(
            ParallelConfig(
                world_size=world,
                sharding_strategy="SHARD_GRAD_OP",
                fsdp_process_group=dist.group.WORLD,
            ),
            max_grad_norm=1.0,
        )
        plain_backend._wrapped = True
        plain_p.grad = torch.tensor([3.0, 4.0, 0.0, 0.0]) if rank == 1 else None
        plain_norm = float(plain_backend._global_grad_norm([plain_p]).item())

        torch.save(
            {"dt_norm": dt_norm, "plain_norm": plain_norm},
            os.path.join(out_dir, f"norm_{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


class ReviewFixesTest(unittest.TestCase):
    def test_set_optimizer_refuses_scheduler_without_max_grad_norm(self):
        model = _SmallModel()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1)
        backend = FSDPTrainingBackend(ParallelConfig())
        with self.assertRaisesRegex(ValueError, "max_grad_norm"):
            backend.set_optimizer(opt, scheduler=sched)
        self.assertIsNone(backend.optimizer)
        self.assertIsNone(backend.scheduler)

    def test_restore_optimizer_kernel_state_updates_adamw_flags_and_step(self):
        from specforge.optimizer import restore_optimizer_kernel_state

        model = _SmallModel()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        opt.step()
        # Simulate loading a checkpoint written with fused=True on a CPU run.
        for g in opt.param_groups:
            g["fused"] = True
            g["foreach"] = False
        restore_optimizer_kernel_state(opt)
        for g in opt.param_groups:
            self.assertIsNone(g["fused"])
        for st in opt.state.values():
            self.assertEqual(st["step"].device.type, "cpu")

        # Explicit fused=True sets fused=True and clears foreach.
        restore_optimizer_kernel_state(opt, fused=True)
        for g in opt.param_groups:
            self.assertTrue(g["fused"])
            self.assertIsNone(g["foreach"])

        # Non-AdamW optimizer is untouched.
        sgd = torch.optim.SGD(model.parameters(), lr=0.1)
        restore_optimizer_kernel_state(sgd, fused=True)
        self.assertIsNone(sgd.param_groups[0].get("fused"))

    def test_torch_optimizer_state_dict_include_params_and_format_key(self):
        from specforge.lr_scheduler import build_lr_scheduler
        from specforge.optimizer import build_optimizer

        def _make_backend(opt_spec, sched_spec):
            torch.manual_seed(0)
            model = _SmallModel()
            backend = FSDPTrainingBackend(
                ParallelConfig(),
                optimizer_spec=opt_spec,
                scheduler_spec=sched_spec,
                max_grad_norm=1.0,
            )
            backend.module = model
            opt = build_optimizer(model, opt_spec)
            sched = build_lr_scheduler(opt, sched_spec)
            backend.set_optimizer(opt, scheduler=sched)
            return model, backend

        opt_spec = OptimizerSpec(
            kind="adamw", lr=1e-2, param_group_policy="no_decay_1d"
        )
        sched_spec = SchedulerSpec(
            kind="wsd",
            total_steps=10,
            warmup_ratio=0.1,
            min_lr_ratio=0.1,
            extra={"decay_ratio": 0.2, "decay_style": "cosine"},
        )
        m1, b1 = _make_backend(opt_spec, sched_spec)
        for p in m1.parameters():
            p.grad = torch.full_like(p, 0.25)
        b1.step()
        saved = b1._torch_optimizer_state_dict(
            include_params=True, format_key="parameters_are_masters"
        )
        self.assertTrue(saved["parameters_are_masters"])
        m2, b2 = _make_backend(opt_spec, sched_spec)
        trainable_m1 = [p for p in m1.parameters() if p.requires_grad]
        trainable_m2 = [p for p in m2.parameters() if p.requires_grad]
        self.assertEqual(len(saved["fp32_params"]), len(trainable_m1))

        # Missing format_key is rejected.
        with self.assertRaisesRegex(ValueError, "was not written"):
            b2._load_torch_optimizer_state_dict(
                {k: v for k, v in saved.items() if k != "parameters_are_masters"},
                restore_params=True,
                format_key="parameters_are_masters",
            )

        # Corrupted fp32_params shape is rejected.
        bad_shape = {
            **saved,
            "fp32_params": [torch.zeros(1), *saved["fp32_params"][1:]],
        }
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            b2._load_torch_optimizer_state_dict(
                bad_shape, restore_params=True, format_key="parameters_are_masters"
            )

        # Valid restore recovers weights, moments, and scheduler bit-for-bit.
        with torch.no_grad():
            for p in trainable_m2:
                p.zero_()
        b2._load_torch_optimizer_state_dict(
            saved, restore_params=True, format_key="parameters_are_masters"
        )
        for p1, p2 in zip(trainable_m1, trainable_m2):
            self.assertTrue(torch.equal(p1, p2))
        self.assertEqual(b2.learning_rate(), b1.learning_rate())

    def test_resume_rejects_mismatched_optimizer_policy_and_scheduler_options(self):
        from specforge.lr_scheduler import build_lr_scheduler
        from specforge.optimizer import BF16Optimizer, build_optimizer

        # 1. Cosine min_lr_ratio=0.0 <-> 0.1 rejected in BOTH directions on BF16Optimizer
        #    and on direct runtime backend (prevents silent TwoStage <-> Functional drift).
        m_zero = _SmallModel().to(torch.bfloat16)
        opt_zero = BF16Optimizer(
            m_zero,
            optimizer_spec=OptimizerSpec(kind="adamw", lr=1e-2),
            scheduler_spec=SchedulerSpec(
                kind="cosine", total_steps=10, warmup_ratio=0.2, min_lr_ratio=0.0
            ),
            max_grad_norm=1.0,
        )
        for _ in range(4):
            for p in m_zero.parameters():
                p.grad = torch.full_like(p, 0.1)
            opt_zero.step()
        sd_zero = opt_zero.state_dict()

        m_pos = _SmallModel().to(torch.bfloat16)
        opt_pos = BF16Optimizer(
            m_pos,
            optimizer_spec=OptimizerSpec(kind="adamw", lr=1e-2),
            scheduler_spec=SchedulerSpec(
                kind="cosine", total_steps=10, warmup_ratio=0.2, min_lr_ratio=0.1
            ),
            max_grad_norm=1.0,
        )
        for _ in range(4):
            for p in m_pos.parameters():
                p.grad = torch.full_like(p, 0.1)
            opt_pos.step()
        sd_pos = opt_pos.state_dict()

        with self.assertRaisesRegex(ValueError, "lr_scheduler_min_lr_ratio"):
            opt_pos.load_state_dict(sd_zero)
        with self.assertRaisesRegex(ValueError, "lr_scheduler_min_lr_ratio"):
            opt_zero.load_state_dict(sd_pos)

        # Legacy checkpoint without the new metadata keys resumes into defaults
        # (adamw / default / cosine / min_lr_ratio=0.0), and rejects min_lr_ratio=0.1.
        legacy_sd = {
            k: v
            for k, v in sd_zero.items()
            if k
            not in (
                "optimizer_type",
                "param_group_policy",
                "optimizer_options",
                "lr_scheduler_min_lr_ratio",
                "lr_scheduler_options",
            )
        }
        with mock.patch("specforge.optimizer.print_on_rank0"):
            opt_zero.load_state_dict(legacy_sd)
        with self.assertRaisesRegex(ValueError, "lr_scheduler_min_lr_ratio"):
            opt_pos.load_state_dict(legacy_sd)

        # 2. param_group_policy mismatch rejected.
        m_nd = _SmallModel().to(torch.bfloat16)
        opt_nd = BF16Optimizer(
            m_nd,
            optimizer_spec=OptimizerSpec(
                kind="adamw", lr=1e-2, param_group_policy="no_decay_1d"
            ),
            scheduler_spec=SchedulerSpec(
                kind="cosine", total_steps=10, warmup_ratio=0.2, min_lr_ratio=0.0
            ),
            max_grad_norm=1.0,
        )
        with self.assertRaisesRegex(ValueError, "param_group_policy"):
            opt_nd.load_state_dict(sd_zero)

        # 3. WSD option mismatch rejected on direct runtime backend.
        def _direct_wsd(extra):
            m = _SmallModel()
            ospec = OptimizerSpec(kind="adamw", lr=1e-2)
            sspec = SchedulerSpec(
                kind="wsd",
                total_steps=10,
                warmup_ratio=0.1,
                min_lr_ratio=0.1,
                extra=extra,
            )
            b = FSDPTrainingBackend(
                ParallelConfig(),
                optimizer_spec=ospec,
                scheduler_spec=sspec,
                max_grad_norm=1.0,
            )
            b.module = m
            o = build_optimizer(m, ospec)
            b.set_optimizer(o, scheduler=build_lr_scheduler(o, sspec))
            return b

        b_lin = _direct_wsd({"decay_ratio": 0.2, "decay_style": "linear"})
        b_cos = _direct_wsd({"decay_ratio": 0.2, "decay_style": "cosine"})
        with self.assertRaisesRegex(ValueError, "lr_scheduler_options"):
            b_cos._load_torch_optimizer_state_dict(b_lin._torch_optimizer_state_dict())

    def test_resolve_run_rejects_invalid_optimizer_and_scheduler_option_combinations(
        self,
    ):
        from specforge.config import Config
        from specforge.config.draft.codec import encode_config
        from specforge.training.canonical import resolve_run
        from tests.test_config.draft_fixtures import draft_config

        base = {
            "model": {"target_model_path": "t"},
            "draft": encode_config(draft_config()),
            "data": {"hidden_states_path": "/tmp/features"},
        }
        target_cfg = {
            "hidden_size": 32,
            "num_hidden_layers": 16,
            "vocab_size": 64,
        }
        for overrides, pattern in (
            ({"optimizer_options": {"bogus": 1}}, "adamw optimizer accepts no extra"),
            (
                {"lr_scheduler": "constant", "min_lr_ratio": 0.1},
                "constant scheduler does not use min_lr_ratio",
            ),
            (
                {"lr_scheduler": "wsd", "lr_scheduler_options": {"decay_style": "exp"}},
                "wsd decay_style",
            ),
            (
                {"lr_scheduler": "wsd", "lr_scheduler_options": {"decay_ratio": 1.5}},
                "wsd decay_ratio",
            ),
        ):
            with self.subTest(overrides=overrides):
                cfg = Config.model_validate({**base, "training": overrides})
                with self.assertRaisesRegex(ValueError, pattern):
                    resolve_run(cfg, target_config=target_cfg)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "needs gloo")
    def test_dtensor_helpers_and_grad_norm_with_empty_rank_do_not_deadlock(self):
        with tempfile.TemporaryDirectory() as d:
            store = os.path.join(d, "store")
            mp.spawn(_dtensor_and_grad_norm_worker, args=(2, store, d), nprocs=2)
            r0 = torch.load(os.path.join(d, "norm_0.pt"), weights_only=False)
            r1 = torch.load(os.path.join(d, "norm_1.pt"), weights_only=False)
            # Rank 1 had 2*3 = 6 elements of value 2.0 -> sqrt(6 * 4) = sqrt(24).
            expected_dt = (6.0 * 4.0) ** 0.5
            self.assertAlmostEqual(r0["dt_norm"], expected_dt, places=5)
            self.assertAlmostEqual(r1["dt_norm"], expected_dt, places=5)
            # Rank 1 had [3, 4, 0, 0] -> norm = 5.0.
            self.assertAlmostEqual(r0["plain_norm"], 5.0, places=5)
            self.assertAlmostEqual(r1["plain_norm"], 5.0, places=5)


if __name__ == "__main__":
    unittest.main()
