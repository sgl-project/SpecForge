# coding=utf-8
"""CPU tests for runtime and training seams.

Covered here: durable ack, MetadataStore, ParallelConfig handles,
checkpoint record shape, and DFlash plug-in wiring.
"""

import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn

from specforge.runtime.contracts import SampleRef, TrainBatch
from specforge.runtime.control_plane import DataFlowController, InMemoryMetadataStore
from specforge.training.backend import ParallelConfig, TrainingBackend
from specforge.training.controller import TrainerController, TrainerCore
from specforge.training.strategies.base import DFlashTrainStrategy


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


class _FakeBackend(TrainingBackend):
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

    def state_dict(self):
        return {
            "model": {"draft_model.w": self.model.w.detach().clone()},
            "optimizer": None,
            "rng": {},
        }

    def load_state_dict(self, state):
        pass


class _FakeDFlashModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, hidden_states, loss_mask):
        # mirrors OnlineDFlashModel's (loss, accuracy, metrics) contract
        loss = (self.w * hidden_states.float().sum()).abs()
        acc = torch.tensor(0.5)
        return loss, acc, {}


class TestDFlashSharesLifecycle(unittest.TestCase):
    def test_dflash_strategy_plugs_into_trainer_core(self):
        model = _FakeDFlashModel()
        strat = DFlashTrainStrategy(model)
        backend = _FakeBackend(model)
        core = TrainerCore(strat, backend, accumulation_steps=1)
        batch = TrainBatch(
            sample_ids=["s0"],
            strategy="dflash",
            tensors={
                "input_ids": torch.zeros(1, 4, dtype=torch.long),
                "hidden_states": torch.randn(1, 4, 8),
                "loss_mask": torch.ones(1, 4, dtype=torch.long),
            },
            metadata={},
        )
        rep = core.train_step(batch)
        self.assertTrue(rep.optimizer_stepped)  # optimizer stepped via the shared core
        self.assertEqual(backend.steps, 1)
        self.assertAlmostEqual(rep.metrics["acc"], 0.5)
        out = strat.forward_loss(batch)
        self.assertAlmostEqual(float(out.metrics["accuracy"]), 0.5)

    def test_dflash_validate_batch_rejects_missing(self):
        strat = DFlashTrainStrategy(_FakeDFlashModel())
        bad = TrainBatch(
            sample_ids=["s"],
            strategy="dflash",
            tensors={"input_ids": torch.zeros(1, 4, dtype=torch.long)},
            metadata={},
        )
        with self.assertRaises(ValueError):
            strat.forward_loss(bad)


class TestCheckpointRecord(unittest.TestCase):
    def test_save_checkpoint_returns_plain_record(self):
        model = _FakeDFlashModel()
        strat = DFlashTrainStrategy(model)
        backend = _FakeBackend(model)
        backend.prepare_model(model)
        core = TrainerCore(strat, backend)
        with tempfile.TemporaryDirectory() as d:
            ctrl = TrainerController(core, run_id="r", output_dir=d)
            ckpt = ctrl.save_checkpoint(10)
        # weight-version/publish/serving-gate deferred to M7: just a checkpoint record
        self.assertTrue(ckpt.checkpoint_uri.startswith("file://"))
        self.assertEqual(ckpt.global_step, 10)
        self.assertEqual(ckpt.strategy, "dflash")


if __name__ == "__main__":
    unittest.main(verbosity=2)
