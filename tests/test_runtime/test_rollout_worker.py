# coding=utf-8
"""RolloutWorker drives PromptTask -> RefSource.produce_refs -> SampleRef commit (CPU)."""

import unittest
from dataclasses import dataclass

import torch

from specforge.inference.capture import CaptureConfig
from specforge.inference.rollout_worker import RolloutWorker
from specforge.runtime.control_plane import DataFlowController
from specforge.runtime.data_plane import LocalFeatureStore

H = 8


def _capture(layer_ids=(1, 2, 3)):
    return CaptureConfig.from_strategy(
        required_features={
            "input_ids",
            "attention_mask",
            "loss_mask",
            "hidden_state",
            "target",
        },
        aux_hidden_state_layer_ids=layer_ids,
        target_repr="logits",
        target_hidden_size=H,
        target_vocab_size=32,
    )


@dataclass(frozen=True)
class _TaskFailure:
    task_id: str
    reason: str
    retryable: bool = True


class FakeRefSource:
    """Stand-in for a RefSource: writes features into a store and returns SampleRefs."""

    def __init__(self, store, *, run_id="run", seq=4, target_dim=32, aux_layers=3):
        self.store = store
        self.run_id = run_id
        self.seq = seq
        self.target_dim = target_dim
        self.aux_layers = aux_layers
        self.task_batches = []

    def produce_refs(self, tasks, *, capture):
        self.task_batches.append([task.task_id for task in tasks])
        refs = []
        for task in tasks:
            feats = {
                "input_ids": torch.zeros(1, self.seq, dtype=torch.long),
                "attention_mask": torch.ones(1, self.seq, dtype=torch.long),
                "loss_mask": torch.ones(1, self.seq, dtype=torch.long),
                "hidden_state": torch.randn(1, self.seq, self.aux_layers * H),
                "target": torch.randn(1, self.seq, self.target_dim),
            }
            refs.append(
                self.store.put(
                    feats,
                    sample_id=f"{self.run_id}:{task.task_id}",
                    metadata={
                        "run_id": self.run_id,
                        "source_task_id": task.task_id,
                        "strategy": "eagle3",
                        "target_repr": capture.target_repr,
                    },
                )
            )
        return refs


class RaisingRefSource:
    def produce_refs(self, tasks, *, capture):
        raise RuntimeError("temporary engine failure")


class ShortRefSource:
    def produce_refs(self, tasks, *, capture):
        return []


class MixedFailureRefSource(FakeRefSource):
    def __init__(self, store, *, retryable=False):
        super().__init__(store)
        self.retryable = retryable

    def produce_refs(self, tasks, *, capture):
        refs = super().produce_refs(tasks[:-1], capture=capture)
        refs.append(
            _TaskFailure(
                task_id=tasks[-1].task_id,
                reason="capture mismatch",
                retryable=self.retryable,
            )
        )
        return refs


class BadReturnRefSource:
    def produce_refs(self, tasks, *, capture):
        return [object() for _ in tasks]


class TestRolloutWorker(unittest.TestCase):
    def test_incompatible_source_rejected_at_construction(self):
        ctrl = DataFlowController("run")
        with self.assertRaisesRegex(TypeError, "produce_refs"):
            RolloutWorker(ctrl, object(), _capture())
        self.assertEqual(ctrl.status()["rollout_workers"], 0)

    def test_run_once_commits_refs(self):
        ctrl = DataFlowController("run")
        ctrl.ingest_prompts([{"payload": {"text": "a"}}, {"payload": {"text": "b"}}])
        store = LocalFeatureStore("st")
        w = RolloutWorker(ctrl, FakeRefSource(store), _capture())
        w.start()
        refs = w.run_once(max_tasks=8)
        self.assertEqual(len(refs), 2)
        self.assertEqual(ctrl.status()["samples_committed"], 2)
        self.assertEqual(ctrl.sample_queue.depth(), 2)
        out, _ = store.get(refs[0])
        self.assertEqual(
            set(out),
            {"input_ids", "attention_mask", "loss_mask", "hidden_state", "target"},
        )
        self.assertEqual(w.health()["state"], "ready")
        self.assertEqual(w.health()["committed"], 2)

    def test_partial_task_failure_commits_good_refs_and_releases_all_leases(self):
        ctrl = DataFlowController("run")
        ctrl.ingest_prompts([{"payload": {"text": "a"}}, {"payload": {"text": "bad"}}])
        store = LocalFeatureStore("st")
        w = RolloutWorker(
            ctrl,
            MixedFailureRefSource(store, retryable=False),
            _capture(),
        )
        w.start()
        refs = w.run_once(max_tasks=8)
        self.assertEqual(len(refs), 1)
        st = ctrl.status()
        self.assertEqual(st["samples_committed"], 1)
        self.assertEqual(st["queue_depth"], 1)
        self.assertEqual(st["prompts_leased"], 0)
        self.assertEqual(st["prompts_failed"], 1)

    def test_retryable_task_failure_requeues_prompt(self):
        ctrl = DataFlowController("run")
        ids = ctrl.ingest_prompts(
            [{"payload": {"text": "a"}}, {"payload": {"text": "retry"}}]
        )
        store = LocalFeatureStore("st")
        w = RolloutWorker(
            ctrl,
            MixedFailureRefSource(store, retryable=True),
            _capture(),
        )
        w.start()
        refs = w.run_once(max_tasks=8)
        self.assertEqual(len(refs), 1)
        st = ctrl.status()
        self.assertEqual(st["prompts_leased"], 0)
        self.assertEqual(st["prompts_pending"], 1)
        self.assertEqual(ctrl.lease_prompt_tasks("w2", 1)[0].task_id, ids[1])

    def test_produce_refs_exception_requeues_prompt(self):
        ctrl = DataFlowController("run")
        ids = ctrl.ingest_prompts([{"payload": {"text": "a"}}])
        w = RolloutWorker(ctrl, RaisingRefSource(), _capture())
        w.start()
        with self.assertRaises(RuntimeError):
            w.run_once(max_tasks=8)
        st = ctrl.status()
        self.assertEqual(st["prompts_leased"], 0)
        self.assertEqual(st["prompts_pending"], 1)
        self.assertEqual(w.health()["state"], "unhealthy")
        self.assertEqual(ctrl.lease_prompt_tasks("w2", 1)[0].task_id, ids[0])

    def test_wrong_record_count_fails_prompt(self):
        ctrl = DataFlowController("run")
        ctrl.ingest_prompts([{"payload": {"text": "a"}}])
        w = RolloutWorker(ctrl, ShortRefSource(), _capture())
        w.start()
        with self.assertRaises(ValueError):
            w.run_once(max_tasks=8)
        st = ctrl.status()
        self.assertEqual(st["prompts_leased"], 0)
        self.assertEqual(st["prompts_failed"], 1)
        self.assertEqual(w.health()["state"], "unhealthy")

    def test_unknown_result_type_fails_prompt(self):
        ctrl = DataFlowController("run")
        ctrl.ingest_prompts([{"payload": {"text": "a"}}])
        w = RolloutWorker(ctrl, BadReturnRefSource(), _capture())
        w.start()
        with self.assertRaises(TypeError):
            w.run_once(max_tasks=8)
        st = ctrl.status()
        self.assertEqual(st["prompts_leased"], 0)
        self.assertEqual(st["prompts_failed"], 1)

    def test_no_tasks_returns_empty(self):
        ctrl = DataFlowController("run")
        store = LocalFeatureStore("st")
        w = RolloutWorker(ctrl, FakeRefSource(store), _capture())
        w.start()
        self.assertEqual(w.run_once(8), [])

    def test_drain_and_stop_prevent_new_leases(self):
        ctrl = DataFlowController("run")
        ctrl.ingest_prompts([{"payload": {"text": "a"}}])
        store = LocalFeatureStore("st")
        w = RolloutWorker(ctrl, FakeRefSource(store), _capture())
        w.start()
        w.drain()
        self.assertEqual(w.run_once(8), [])
        self.assertEqual(ctrl.status()["prompts_pending"], 1)
        w.stop()
        self.assertEqual(w.run_once(8), [])
        self.assertEqual(w.health()["state"], "stopped")


if __name__ == "__main__":
    unittest.main(verbosity=2)
