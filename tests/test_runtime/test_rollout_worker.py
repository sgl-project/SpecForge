# coding=utf-8
"""RolloutWorker drives PromptTask -> SampleRef commit (CPU, fake ref source)."""

import unittest

from specforge.inference.capture import CaptureConfig
from specforge.inference.rollout_worker import RolloutWorker
from specforge.runtime.contracts import FeatureSpec, SampleRef
from specforge.runtime.control_plane import DataFlowController

H = 8


def _capture():
    return CaptureConfig.from_strategy(
        required_features={"input_ids", "loss_mask", "hidden_state", "target"},
        aux_hidden_state_layer_ids=(1, 2, 3),
        target_repr="logits",
        target_hidden_size=H,
        target_vocab_size=32,
    )


def _ref(task):
    sample_id = f"run:{task.task_id}"
    spec = FeatureSpec(name="hidden_state", shape=(1, 4, 3 * H), dtype="float32")
    return SampleRef(
        sample_id=sample_id,
        run_id="run",
        source_task_id=task.task_id,
        feature_store_uri=f"mooncake://run/{sample_id}",
        feature_keys={"hidden_state": f"{sample_id}/hidden_state"},
        feature_specs={"hidden_state": spec},
        strategy="eagle3",
        num_tokens=4,
    )


class FakeSource:
    """Stand-in for a capture server: returns the ref of each task."""

    def produce_refs(self, tasks, *, capture):
        return [_ref(task) for task in tasks]


class RaisingSource:
    def produce_refs(self, tasks, *, capture):
        raise RuntimeError("temporary server failure")


class ShortSource:
    def produce_refs(self, tasks, *, capture):
        return []


class TestRolloutWorker(unittest.TestCase):
    def test_run_once_commits_refs(self):
        ctrl = DataFlowController("run")
        ctrl.ingest_prompts([{"payload": {"text": "a"}}, {"payload": {"text": "b"}}])
        w = RolloutWorker(ctrl, FakeSource(), _capture())
        w.start()
        refs = w.run_once(max_tasks=8)
        self.assertEqual(len(refs), 2)
        self.assertEqual(ctrl.status()["samples_committed"], 2)
        self.assertEqual(ctrl.sample_queue.depth(), 2)
        self.assertEqual(w.health()["state"], "ready")

    def test_source_failure_requeues_prompt(self):
        ctrl = DataFlowController("run")
        ids = ctrl.ingest_prompts([{"payload": {"text": "a"}}])
        w = RolloutWorker(ctrl, RaisingSource(), _capture())
        w.start()
        with self.assertRaises(RuntimeError):
            w.run_once(max_tasks=8)
        st = ctrl.status()
        self.assertEqual(st["prompts_leased"], 0)
        self.assertEqual(st["prompts_pending"], 1)
        self.assertEqual(ctrl.lease_prompt_tasks("w2", 1)[0].task_id, ids[0])

    def test_wrong_ref_count_fails_prompt(self):
        ctrl = DataFlowController("run")
        ctrl.ingest_prompts([{"payload": {"text": "a"}}])
        w = RolloutWorker(ctrl, ShortSource(), _capture())
        w.start()
        with self.assertRaises(ValueError):
            w.run_once(max_tasks=8)
        st = ctrl.status()
        self.assertEqual(st["prompts_leased"], 0)
        self.assertEqual(st["prompts_failed"], 1)

    def test_no_tasks_returns_empty(self):
        ctrl = DataFlowController("run")
        w = RolloutWorker(ctrl, FakeSource(), _capture())
        w.start()
        self.assertEqual(w.run_once(8), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
