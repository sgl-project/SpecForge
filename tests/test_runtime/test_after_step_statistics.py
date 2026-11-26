"""Tests for the ``after_step_statistics`` contract across DDP, FSDP1, and FSDP2."""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from specforge.algorithms.objectives.base import ObjectiveModel, ObjectiveResult
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.backends.fsdp2 import FSDP2TrainingBackend
from specforge.training.checkpoint import CheckpointManager
from specforge.training.contracts import (
    OptimizerSpec,
    PrecisionSpec,
    SchedulerSpec,
    StepContext,
)
from specforge.training.frame import run_window
from specforge.training.runner import DataParallelRunner
from tests.test_algorithms.test_objective_contracts import build_runtime
from tests.test_runtime._fixtures import FakeProgram, fake_spec

_GLOO = dist.is_available() and dist.is_gloo_available()


class _MoELikeProgram(FakeProgram):
    """Minimal program mimicking MoE #1179 per-layer expert counts and bias updates."""

    def __init__(self, *, dtype=torch.float32, fail_grad=False, zero_denom=False):
        super().__init__()
        torch.manual_seed(7)
        self.proj = nn.Linear(4, 4, bias=False).to(dtype)
        # Two layers with different expert counts, like heterogeneous MoE drafts.
        self.register_buffer("expert_bias_0", torch.zeros(4, dtype=torch.float32))
        self.register_buffer("expert_bias_1", torch.zeros(6, dtype=torch.float32))
        self.fail_grad = fail_grad
        self.zero_denom = zero_denom
        self.received_pooled = []

    def spec(self):
        return fake_spec(
            "x",
            "counts_0",
            "counts_1",
            name="moe_like",
            after_step_statistics=("counts_0", "counts_1"),
        )

    def forward(
        self,
        tensors,
        *,
        step=0,
        total_steps=None,
        collect_detailed_metrics=True,
        prepass=False,
        **context,
    ):
        x = tensors["x"].to(self.proj.weight.dtype)
        y = self.proj(x)
        loss = (y * y).sum()
        if self.fail_grad:
            loss = loss * float("nan")
        denom = torch.zeros(()) if self.zero_denom else torch.ones(())
        result = ObjectiveResult(loss, denom)
        if self.training and not prepass:
            result.statistics["counts_0"] = tensors["counts_0"]
            result.statistics["counts_1"] = tensors["counts_1"]
        return result

    def apply_statistics(self, values):
        c0 = values["counts_0"]
        c1 = values["counts_1"]
        self.received_pooled.append(
            (c0.detach().cpu().clone(), c1.detach().cpu().clone())
        )
        with torch.no_grad():
            self.expert_bias_0.add_(0.1 * c0.to(self.expert_bias_0.dtype))
            self.expert_bias_1.add_(0.05 * c1.to(self.expert_bias_1.dtype))
        balance = (c0.float().mean() + c1.float().mean()) / 2.0
        return {"moe/step_balance": balance}


def _make_batch(sample_id: str, c0, c1, *, x_val: float = 1.0):
    return TrainBatch(
        [sample_id],
        "moe_like",
        {
            "x": torch.full((2, 4), x_val, dtype=torch.float32),
            "counts_0": torch.tensor(c0, dtype=torch.int64),
            "counts_1": torch.tensor(c1, dtype=torch.float32),
        },
    )


def _plain_runner(program):
    backend = FSDPTrainingBackend(
        ParallelConfig(),
        optimizer_spec=OptimizerSpec(kind="adamw", lr=1e-2),
        scheduler_spec=SchedulerSpec(kind="constant", total_steps=10, warmup_ratio=0.0),
        max_grad_norm=1.0,
    )
    backend.prepare_model(program, wrap=False)
    return DataParallelRunner(program, backend, program.spec())


def _on_host_fsdp1():
    """FSDP1 computes on the host only when told so (avoids macOS MPS probe)."""
    from torch.distributed.fsdp import FullyShardedDataParallel

    original = FullyShardedDataParallel.__init__

    def on_host(self, module, *args, **kwargs):
        kwargs.setdefault("device_id", torch.device("cpu"))
        original(self, module, *args, **kwargs)

    return mock.patch.object(FullyShardedDataParallel, "__init__", on_host)


def _distributed_after_step_worker(rank, world, store_path, out_dir, mode, report_dir):
    dist.init_process_group(
        "gloo", init_method=f"file://{store_path}", rank=rank, world_size=world
    )
    try:
        dtype = torch.bfloat16 if mode == "fsdp2" else torch.float32
        program = _MoELikeProgram(dtype=dtype)
        spec = program.spec()
        opt_spec = OptimizerSpec(kind="adamw", lr=1e-2)
        sched_spec = SchedulerSpec(kind="constant", total_steps=10, warmup_ratio=0.0)
        if mode == "ddp":
            backend = FSDPTrainingBackend(
                ParallelConfig(
                    world_size=world,
                    sharding_strategy="NO_SHARD",
                    fsdp_process_group=dist.group.WORLD,
                ),
                optimizer_spec=opt_spec,
                scheduler_spec=sched_spec,
                max_grad_norm=1.0,
                precision=PrecisionSpec.resolve(
                    runtime="fsdp",
                    model_dtype=torch.float32,
                    sharding_strategy="NO_SHARD",
                ),
            )
        elif mode == "fsdp1":
            backend = FSDPTrainingBackend(
                ParallelConfig(
                    world_size=world,
                    sharding_strategy="SHARD_GRAD_OP",
                    fsdp_process_group=dist.group.WORLD,
                ),
                optimizer_spec=opt_spec,
                scheduler_spec=sched_spec,
                max_grad_norm=1.0,
                precision=PrecisionSpec.resolve(
                    runtime="fsdp",
                    model_dtype=torch.float32,
                    sharding_strategy="SHARD_GRAD_OP",
                ),
            )
        elif mode == "fsdp2":
            backend = FSDP2TrainingBackend(
                ParallelConfig(
                    world_size=world,
                    sharding_strategy="SHARD_GRAD_OP",
                    fsdp_process_group=dist.group.WORLD,
                ),
                optimizer_spec=opt_spec,
                scheduler_spec=sched_spec,
                max_grad_norm=1.0,
                precision=PrecisionSpec.resolve(
                    runtime="fsdp2",
                    model_dtype=torch.bfloat16,
                    sharding_strategy="SHARD_GRAD_OP",
                ),
            )
        else:
            raise ValueError(mode)

        with _on_host_fsdp1() if mode == "fsdp1" else contextlib.nullcontext():
            wrapped = backend.prepare_model(
                program, frozen_modules=program.frozen_modules
            )
        runner = DataParallelRunner(wrapped, backend, spec)

        # Distinct per-rank, per-microbatch counts across two micro-batches:
        # rank 0: m0=[1,2,3,4]/[1..6], m1=[2,0,1,0]/[0.5]*6
        # rank 1: m0=[5,1,0,2]/[2..7], m1=[0,3,2,1]/[1.5]*6
        if rank == 0:
            window = [
                _make_batch("r0-0", [1, 2, 3, 4], [1, 2, 3, 4, 5, 6], x_val=1.0),
                _make_batch("r0-1", [2, 0, 1, 0], [0.5] * 6, x_val=0.5),
            ]
        else:
            window = [
                _make_batch("r1-0", [5, 1, 0, 2], [2, 3, 4, 5, 6, 7], x_val=1.5),
                _make_batch("r1-1", [0, 3, 2, 1], [1.5] * 6, x_val=0.75),
            ]

        metrics = run_window(runner, window, StepContext(accumulation_steps=2))
        materialized = metrics.materialize()

        # Snapshot checkpoint right after step: updated expert_bias must be inside.
        ckpt_state = runner.checkpoint_state()
        mgr = CheckpointManager(out_dir, f"after-step-{mode}")
        shared = (
            {
                "global_step": 1,
                "world_size": world,
                "draft_state_dict": ckpt_state.draft,
            }
            if rank == 0
            else None
        )
        mgr.save(shared, 1, rank_state=ckpt_state.rank)

        c0_seen, c1_seen = program.received_pooled[0]
        report = {
            "pooled_0": c0_seen.tolist(),
            "pooled_1": c1_seen.tolist(),
            "pooled_0_dtype": str(c0_seen.dtype),
            "pooled_1_dtype": str(c1_seen.dtype),
            "bias_0": program.expert_bias_0.detach().cpu().tolist(),
            "bias_1": program.expert_bias_1.detach().cpu().tolist(),
            "step_balance": materialized["moe/step_balance"],
            "ckpt_bias_0": (
                ckpt_state.draft["expert_bias_0"].detach().cpu().tolist()
                if rank == 0
                else None
            ),
            "ckpt_bias_1": (
                ckpt_state.draft["expert_bias_1"].detach().cpu().tolist()
                if rank == 0
                else None
            ),
        }
        with open(os.path.join(report_dir, f"rank{rank}.json"), "w") as fh:
            json.dump(report, fh)
    finally:
        dist.destroy_process_group()


class AfterStepStatisticsTest(unittest.TestCase):
    def test_accumulates_across_window_and_updates_before_checkpoint(self):
        program = _MoELikeProgram()
        runner = _plain_runner(program)
        window = [
            _make_batch("s0", [1, 2, 3, 4], [1, 0, 1, 0, 1, 0]),
            _make_batch("s1", [2, 1, 0, 1], [0, 2, 0, 2, 0, 2]),
            _make_batch("s2", [0, 3, 1, 0], [1, 1, 1, 1, 1, 1]),
        ]
        metrics = run_window(runner, window, StepContext(accumulation_steps=3))
        # Pooled sums: [3, 6, 4, 5] and [2, 3, 2, 3, 2, 3].
        self.assertEqual(len(program.received_pooled), 1)
        c0, c1 = program.received_pooled[0]
        self.assertEqual(c0.dtype, torch.int64)
        self.assertEqual(c1.dtype, torch.float32)
        self.assertEqual(c0.tolist(), [3, 6, 4, 5])
        self.assertEqual(c1.tolist(), [2.0, 3.0, 2.0, 3.0, 2.0, 3.0])

        expected_b0 = [0.3, 0.6, 0.4, 0.5]
        expected_b1 = [0.1, 0.15, 0.1, 0.15, 0.1, 0.15]
        torch.testing.assert_close(program.expert_bias_0, torch.tensor(expected_b0))
        torch.testing.assert_close(program.expert_bias_1, torch.tensor(expected_b1))

        # Checkpoint immediately after the step captures the updated bias.
        ckpt = runner.checkpoint_state()
        torch.testing.assert_close(
            ckpt.draft["expert_bias_0"], torch.tensor(expected_b0)
        )
        torch.testing.assert_close(
            ckpt.draft["expert_bias_1"], torch.tensor(expected_b1)
        )

        # Post-step metric merged into StepMetrics.
        self.assertAlmostEqual(
            metrics.materialize()["moe/step_balance"],
            (sum([3, 6, 4, 5]) / 4.0 + sum([2, 3, 2, 3, 2, 3]) / 6.0) / 2.0,
            places=5,
        )

    def test_failed_optimizer_step_does_not_apply_after_step_statistics(self):
        for failure_kind, kwargs, exc_type in (
            ("zero_denom", {"zero_denom": True}, ValueError),
            ("non_finite_grad", {"fail_grad": True}, FloatingPointError),
        ):
            with self.subTest(failure_kind=failure_kind):
                program = _MoELikeProgram(**kwargs)
                runner = _plain_runner(program)
                with self.assertRaises(exc_type):
                    run_window(
                        runner,
                        [_make_batch("s0", [4, 4, 4, 4], [2] * 6)],
                        StepContext(accumulation_steps=1),
                    )
                self.assertEqual(program.received_pooled, [])
                torch.testing.assert_close(program.expert_bias_0, torch.zeros(4))
                torch.testing.assert_close(program.expert_bias_1, torch.zeros(6))

    def test_missing_or_unexpected_after_step_statistics_fail_before_backward(self):
        class MissingStat(_MoELikeProgram):
            def forward(self, tensors, **context):
                res = super().forward(tensors, **context)
                res.statistics.pop("counts_1", None)
                return res

        program = MissingStat()
        runner = _plain_runner(program)
        with self.assertRaisesRegex(ValueError, "missing=\\['counts_1'\\]"):
            run_window(
                runner,
                [_make_batch("s0", [1, 1, 1, 1], [1] * 6)],
                StepContext(accumulation_steps=1),
            )
        self.assertIsNone(program.proj.weight.grad)

        class UndeclaredStat(FakeProgram):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 1, bias=False)

            def spec(self):
                return fake_spec("x", name="undeclared")

            def forward(self, tensors, **context):
                res = ObjectiveResult(self.linear(tensors["x"]).sum(), torch.ones(()))
                res.statistics["secret"] = torch.ones(2)
                return res

        undeclared = UndeclaredStat()
        runner2 = _plain_runner(undeclared)
        with self.assertRaisesRegex(ValueError, "unexpected=\\['secret'\\]"):
            run_window(
                runner2,
                [_make_batch("s0", [1, 1, 1, 1], [1] * 6)],
                StepContext(accumulation_steps=1),
            )
        self.assertIsNone(undeclared.linear.weight.grad)

    def test_shape_drift_and_metric_collision_fail_loudly(self):
        program = _MoELikeProgram()
        runner = _plain_runner(program)
        bad_window = [
            _make_batch("s0", [1, 2, 3, 4], [1] * 6),
            TrainBatch(
                ["s1"],
                "moe_like",
                {
                    "x": torch.ones((2, 4)),
                    "counts_0": torch.ones(3, dtype=torch.int64),
                    "counts_1": torch.ones(6, dtype=torch.float32),
                },
            ),
        ]
        with self.assertRaisesRegex(ValueError, "changed shape/dtype mid-window"):
            run_window(runner, bad_window, StepContext(accumulation_steps=2))

        class CollidingMetric(_MoELikeProgram):
            def apply_statistics(self, values):
                return {"loss": torch.tensor(1.0)}

        collider = CollidingMetric()
        runner2 = _plain_runner(collider)
        with self.assertRaisesRegex(ValueError, "collide with step metrics"):
            run_window(
                runner2,
                [_make_batch("s0", [1, 1, 1, 1], [1] * 6)],
                StepContext(accumulation_steps=1),
            )

    def test_objective_model_delegates_apply_statistics_or_refuses_unhandled(self):
        _, runtime = build_runtime("DFlashDraftModel")
        self.assertEqual(runtime.apply_statistics({}), {})
        with self.assertRaisesRegex(NotImplementedError, "without an apply_statistics"):
            runtime.apply_statistics({"moe_counts": torch.ones(4)})

        seen = []

        class ObjectiveWithHandler:
            def apply_statistics(self, model, values):
                seen.append((model, values["moe_counts"].tolist()))
                return {"moe/step_balance": torch.tensor(0.75)}

        model = ObjectiveModel(
            runtime.draft_model,
            ObjectiveWithHandler(),
            registration=runtime.registration,
            name=runtime.objective_name,
            loss_configuration=runtime.loss_configuration,
            after_step_statistics=("moe_counts",),
        )
        out = model.apply_statistics({"moe_counts": torch.tensor([2, 4, 6, 8])})
        self.assertEqual(len(seen), 1)
        self.assertIs(seen[0][0], model)
        self.assertEqual(seen[0][1], [2, 4, 6, 8])
        self.assertAlmostEqual(float(out["moe/step_balance"]), 0.75)

    @unittest.skipUnless(_GLOO, "requires torch.distributed gloo backend")
    def test_two_rank_pooling_updates_bias_and_checkpoint_across_ddp_fsdp1_fsdp2(self):
        # Global expected sums across both ranks and both micro-batches:
        # counts_0: [1,2,3,4]+[2,0,1,0]+[5,1,0,2]+[0,3,2,1] = [8, 6, 6, 7]
        # counts_1: [1,2,3,4,5,6]+[0.5]*6+[2,3,4,5,6,7]+[1.5]*6 = [5,7,9,11,13,15]
        expected_c0 = [8, 6, 6, 7]
        expected_c1 = [5.0, 7.0, 9.0, 11.0, 13.0, 15.0]
        expected_b0 = [0.1 * v for v in expected_c0]
        expected_b1 = [0.05 * v for v in expected_c1]
        expected_bal = (sum(expected_c0) / 4.0 + sum(expected_c1) / 6.0) / 2.0

        for mode in ("ddp", "fsdp1", "fsdp2"):
            with self.subTest(mode=mode):
                workdir = tempfile.mkdtemp(prefix=f"after_step_{mode}_")
                store_path = os.path.join(workdir, "gloo.store")
                out_dir = os.path.join(workdir, "checkpoints")
                report_dir = os.path.join(workdir, "reports")
                os.makedirs(out_dir, exist_ok=True)
                os.makedirs(report_dir, exist_ok=True)
                mp.spawn(
                    _distributed_after_step_worker,
                    args=(2, store_path, out_dir, mode, report_dir),
                    nprocs=2,
                    join=True,
                )
                reports = []
                for rank in range(2):
                    with open(os.path.join(report_dir, f"rank{rank}.json")) as fh:
                        reports.append(json.load(fh))

                for rank, rep in enumerate(reports):
                    self.assertEqual(rep["pooled_0"], expected_c0, f"{mode} r{rank}")
                    self.assertEqual(rep["pooled_1"], expected_c1, f"{mode} r{rank}")
                    self.assertEqual(rep["pooled_0_dtype"], "torch.int64")
                    self.assertEqual(rep["pooled_1_dtype"], "torch.float32")
                    torch.testing.assert_close(
                        torch.tensor(rep["bias_0"]), torch.tensor(expected_b0)
                    )
                    torch.testing.assert_close(
                        torch.tensor(rep["bias_1"]), torch.tensor(expected_b1)
                    )
                    self.assertAlmostEqual(rep["step_balance"], expected_bal, places=5)

                # Rank 0 checkpoint artifact holds the post-step bias.
                torch.testing.assert_close(
                    torch.tensor(reports[0]["ckpt_bias_0"]), torch.tensor(expected_b0)
                )
                torch.testing.assert_close(
                    torch.tensor(reports[0]["ckpt_bias_1"]), torch.tensor(expected_b1)
                )
                # Saved training_state.pt on disk carries the same updated bias.
                saved = CheckpointManager.read_resume_state(
                    out_dir, require_full_state=False
                )
                torch.testing.assert_close(
                    saved["draft_state_dict"]["expert_bias_0"],
                    torch.tensor(expected_b0),
                )
                torch.testing.assert_close(
                    saved["draft_state_dict"]["expert_bias_1"],
                    torch.tensor(expected_b1),
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
