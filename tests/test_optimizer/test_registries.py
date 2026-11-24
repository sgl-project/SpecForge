"""Tests for shared optimizer/scheduler registries, runtime-owned updates, and grad_reduction."""

from __future__ import annotations

import copy
import math
import unittest
from unittest import mock

import torch
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

        def custom_policy(named_params, spec):
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
        self.frozen = ()
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


if __name__ == "__main__":
    unittest.main()
