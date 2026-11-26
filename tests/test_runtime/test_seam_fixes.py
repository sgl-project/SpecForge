# coding=utf-8
"""CPU tests for runtime and training seams.

Covered here: durable ack, MetadataStore, ParallelConfig handles,
checkpoint record shape, and DFlash plug-in wiring.
"""

import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from specforge.algorithms.objectives.base import ObjectiveModel, ObjectiveResult
from specforge.runtime.contracts import SampleRef, TrainBatch
from specforge.runtime.control_plane import DataFlowController, InMemoryMetadataStore
from specforge.training.backend import (
    FSDPTrainingBackend,
    ParallelConfig,
    TrainingBackend,
)
from specforge.training.controller import TrainerController
from tests.test_runtime._fixtures import build_runner, forward_program, train_window


def _ref(i):
    return SampleRef(
        sample_id=f"s{i}",
        run_id="r",
        source_task_id=None,
        feature_store_uri=f"mem://x/s{i}",
        feature_keys={},
        feature_specs={},
        strategy="eagle3",
    )


class TestDurableAckTransaction(unittest.TestCase):
    def test_ack_records_durable_marker(self):
        ctrl = DataFlowController("run")
        refs = [_ref(0), _ref(1)]
        ctrl.commit_samples("ref-distributor", refs)
        leased = ctrl.sample_queue.get(2)
        ctrl.ack_train_refs(
            "t0", [r.sample_id for r in leased], global_step=7, optimizer_durable=True
        )
        marker = ctrl.store.durable_marker()
        self.assertEqual(marker["global_step"], 7)
        self.assertTrue(marker["optimizer_durable"])
        self.assertEqual(marker["acked"], {"s0", "s1"})
        st = ctrl.status()
        self.assertEqual(st["durable_global_step"], 7)
        self.assertEqual(st["durable_acked"], 2)
        self.assertEqual(ctrl.sample_queue.depth(), 0)

    def test_ack_without_step_records_ids_only(self):
        ctrl = DataFlowController("run")
        ctrl.commit_samples("ref-distributor", [_ref(0)])
        ctrl.sample_queue.get(1)
        ctrl.ack_train_refs("t0", ["s0"])
        marker = ctrl.store.durable_marker()
        self.assertEqual(marker["acked"], {"s0"})
        self.assertIsNone(marker["global_step"])


class TestMetadataStore(unittest.TestCase):
    def test_commit_dedup(self):
        s = InMemoryMetadataStore()
        self.assertTrue(s.commit_sample(_ref(0)))
        self.assertFalse(s.commit_sample(_ref(0)))  # dup
        self.assertEqual(s.committed_count(), 1)


class TestParallelConfigHandles(unittest.TestCase):
    def test_carries_target_and_draft_parallel_state(self):
        pc = ParallelConfig(tp_size=2, sp_ulysses_size=2, sp_ring_size=2)
        self.assertIsNone(pc.fsdp_process_group)
        self.assertEqual(pc.sharding_strategy, "SHARD_GRAD_OP")
        self.assertEqual(pc.tp_size, 2)
        self.assertEqual(pc.sp_size, 4)

    def test_from_distributed_no_dist(self):
        # Keep this test independent of process groups initialized by earlier
        # GPU launcher tests in the same unittest process.
        with mock.patch(
            "specforge.training.backend.dist.is_initialized", return_value=False
        ):
            pc = ParallelConfig.from_distributed(tp_size=2, sp_ulysses_size=2)
        self.assertIsNone(pc.fsdp_process_group)
        self.assertEqual(pc.world_size, 1)
        self.assertEqual(pc.tp_size, 2)
        self.assertEqual(pc.sp_size, 2)

    def test_backend_scales_gradients_before_optimizer_step(self):
        model = nn.Linear(2, 1, bias=False)
        backend = FSDPTrainingBackend(ParallelConfig())
        backend.prepare_model(model, wrap=False)
        model.weight.grad = torch.tensor([[4.0, 8.0]])

        backend.scale_gradients(torch.tensor(0.25))

        torch.testing.assert_close(model.weight.grad, torch.tensor([[1.0, 2.0]]))


class _FakeBackend(TrainingBackend):
    process_group = None
    name = "fake"

    def __init__(self, model):
        self.model = model
        self.steps = 0

    def prepare_model(self, model):
        self.module = model
        return model

    def backward(self, loss, *, is_boundary=True):
        loss.backward()

    def step(self):
        self.steps += 1
        return torch.tensor(1.0)

    def scale_gradients(self, scale):
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(scale)

    def state_dict(self):
        return {
            "model": {"draft_model.w": self.model.w.detach().clone()},
            "optimizer": None,
            "rng": {},
        }

    def load_state_dict(self, state):
        pass


class _FakeDFlashModel(ObjectiveModel):
    """An objective model declared as DFlash, with a one-parameter draft."""

    def __init__(self):
        from specforge.algorithms.objectives.registry import TrainingRequirements

        draft = nn.Module()
        draft.register_parameter("w", nn.Parameter(torch.ones(1)))
        draft.config = SimpleNamespace(draft=None)
        draft._no_split_modules = []
        super().__init__(
            draft,
            objective=None,
            registration=SimpleNamespace(
                inputs=("hidden_states",), training=TrainingRequirements()
            ),
            name="dflash",
            loss_configuration=None,
        )

    @property
    def w(self):
        return self.draft_model.w

    def forward(self, batch, **context):
        loss = (self.w * batch["hidden_states"].float().sum()).abs()
        return ObjectiveResult(loss, loss.new_ones(()))


class TestDFlashSharesLifecycle(unittest.TestCase):
    def test_objective_model_uses_shared_trainer_lifecycle(self):
        model = _FakeDFlashModel()
        backend = _FakeBackend(model)
        runner = build_runner(model, backend)
        batch = TrainBatch(["s0"], "dflash", {"hidden_states": torch.randn(1, 4, 8)})
        metrics = train_window(runner, [batch])
        self.assertIsNotNone(metrics.grad_norm)
        self.assertEqual(backend.steps, 1)
        self.assertIsNotNone(model.w.grad)

    def test_missing_required_feature_fails(self):
        with self.assertRaises(ValueError):
            forward_program(_FakeDFlashModel(), TrainBatch(["s0"], "dflash", {}))


class TestCheckpointRecord(unittest.TestCase):
    def test_save_checkpoint_returns_plain_record(self):
        model = _FakeDFlashModel()
        backend = _FakeBackend(model)
        backend.prepare_model(model)
        runner = build_runner(model, backend)
        with tempfile.TemporaryDirectory() as d:
            ctrl = TrainerController(
                runner,
                strategy_name=runner.spec.name,
                run_id="r",
                output_dir=d,
            )
            ckpt = ctrl.save_checkpoint(10)
        # weight-version/publish/serving-gate deferred to M7: just a checkpoint record
        self.assertTrue(ckpt.checkpoint_uri.startswith("file://"))
        self.assertEqual(ckpt.global_step, 10)
        self.assertEqual(ckpt.strategy, "dflash")


if __name__ == "__main__":
    unittest.main(verbosity=2)
