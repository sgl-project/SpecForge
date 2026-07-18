# coding=utf-8
"""M1 gate: the controller carries no tensor across any API (CPU-only)."""

import unittest

import torch

from specforge.runtime.contracts import FeatureSpec, SampleRef
from specforge.runtime.control_plane.controller import DataFlowController
from specforge.runtime.control_plane.metadata_store import NoOpMetadataStore


def _ref(i: int, store_uri="mem://x") -> SampleRef:
    return SampleRef(
        sample_id=f"s{i}",
        run_id="r",
        source_task_id=None,
        feature_store_uri=f"{store_uri}/s{i}",
        feature_keys={"input_ids": f"s{i}/input_ids"},
        feature_specs={"input_ids": FeatureSpec("input_ids", (1, 8), "int64")},
        strategy="eagle3",
    )


class TestControllerCarriesNoTensor(unittest.TestCase):
    def test_controller_carries_no_tensor(self):
        """Prompt and sample APIs reject any tensor payload."""
        ctrl = DataFlowController("run1")

        # a SampleRef with a tensor smuggled into metadata must be rejected
        bad = SampleRef(
            sample_id="bad",
            run_id="r",
            source_task_id=None,
            feature_store_uri="mem://x/bad",
            feature_keys={},
            feature_specs={},
            strategy="eagle3",
            metadata={"sneaky": torch.zeros(4)},
        )
        with self.assertRaises(TypeError):
            ctrl.commit_samples("w0", [bad])

        # a prompt carrying a tensor in its payload must be rejected
        with self.assertRaises(TypeError):
            ctrl.ingest_prompts([{"payload": {"ids": torch.zeros(3)}}])

    def test_prompt_batch_validation_is_atomic(self):
        ctrl = DataFlowController("run1")

        with self.assertRaises(TypeError):
            ctrl.ingest_prompts(
                [
                    {"task_id": "good", "payload": {"ids": [1, 2, 3]}},
                    {"task_id": "bad", "payload": {"ids": torch.zeros(3)}},
                ]
            )

        status = ctrl.status()
        self.assertEqual(status["prompts"], 0)
        self.assertEqual(status["prompts_pending"], 0)

    def test_sample_batch_validation_is_atomic(self):
        ctrl = DataFlowController("run1")
        bad = SampleRef(
            sample_id="bad",
            run_id="r",
            source_task_id=None,
            feature_store_uri="mem://x/bad",
            feature_keys={},
            feature_specs={},
            strategy="eagle3",
            metadata={"sneaky": torch.zeros(4)},
        )

        with self.assertRaises(TypeError):
            ctrl.commit_samples("w0", [_ref(0), bad])

        self.assertEqual(ctrl.status()["samples_committed"], 0)
        self.assertEqual(ctrl.sample_queue.depth(), 0)

    def test_prompt_batch_preserves_each_task_id(self):
        ctrl = DataFlowController("run1")
        expected = ["task-0", "task-1"]

        task_ids = ctrl.ingest_prompts(
            [
                {"task_id": task_id, "payload": {"ids": [1, 2, 3]}}
                for task_id in expected
            ]
        )
        tasks = ctrl.lease_prompt_tasks("w0", 2)

        self.assertEqual(task_ids, expected)
        self.assertEqual([task.task_id for task in tasks], expected)

    def test_commit_samples_idempotent(self):
        ctrl = DataFlowController("run1")
        ref = _ref(0)
        self.assertEqual(ctrl.commit_samples("w0", [ref]), [ref])
        # At-least-once delivery is deduplicated on sample_id.
        self.assertEqual(
            ctrl.commit_samples("w0", [ref]),
            [],
        )
        self.assertEqual(ctrl.status()["samples_committed"], 1)
        self.assertEqual(ctrl.sample_queue.depth(), 1)

    def test_prompt_lease_and_commit_clears_lease(self):
        ctrl = DataFlowController("run1")
        ids = ctrl.ingest_prompts([{"payload": {"text": "hi"}, "max_length": 16}])
        tasks = ctrl.lease_prompt_tasks("w0", 4)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].task_id, ids[0])
        ref = _ref(0)
        ref = SampleRef(**{**ref.__dict__, "source_task_id": ids[0]})
        ctrl.commit_samples("w0", [ref])
        self.assertEqual(ctrl.status()["prompts_leased"], 0)

    def test_producer_controller_completes_prompt_without_retaining_refs(self):
        ctrl = DataFlowController(
            "run1",
            metadata_store=NoOpMetadataStore(),
            enable_sample_queue=False,
        )
        [task_id] = ctrl.ingest_prompts(
            [{"task_id": "task-0", "payload": {"text": "hi"}}]
        )
        ctrl.lease_prompt_tasks("w0", 1)
        ref = _ref(0)
        ref = SampleRef(**{**ref.__dict__, "source_task_id": task_id})
        ctrl.commit_samples("w0", [ref])

        self.assertEqual(ctrl.status()["prompts_leased"], 0)
        self.assertIsNone(ctrl.sample_queue)
        self.assertEqual(ctrl.store.committed_count(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
