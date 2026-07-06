# coding=utf-8
"""Backpressure policy + its wiring into DataFlowController (CPU, no torch)."""

import unittest

from specforge.runtime.contracts import SampleRef
from specforge.runtime.control_plane.backpressure import (
    BackpressureConfig,
    BackpressureController,
)
from specforge.runtime.control_plane.controller import DataFlowController


class _FakeCapacity:
    """Minimal CapacityReporter: a mutable health dict, ints only (no tensors)."""

    def __init__(self) -> None:
        self.resident_bytes = 0
        self.max_resident_bytes = None
        self.avg_age_s = 0.0
        self.oldest_age_s = 0.0

    def health(self):
        return {
            "resident_bytes": self.resident_bytes,
            "max_resident_bytes": self.max_resident_bytes,
            "avg_age_s": self.avg_age_s,
            "oldest_age_s": self.oldest_age_s,
        }


def _ref(sid: str) -> SampleRef:
    return SampleRef(
        sample_id=sid,
        run_id="r",
        source_task_id=None,
        feature_store_uri=f"mem://st/{sid}",
        feature_keys={"x": f"{sid}/x"},
        feature_specs={},
        strategy="eagle3",
    )


class TestBackpressurePolicy(unittest.TestCase):
    def test_config_rejects_low_above_high(self):
        with self.assertRaises(ValueError):
            BackpressureConfig(high_watermark_bytes=100, low_watermark_bytes=200)

    def test_no_config_never_pauses(self):
        bp = BackpressureController()  # empty config, no capacity
        self.assertFalse(bp.should_pause_prompts())

    def test_hysteresis_pause_and_resume(self):
        cap = _FakeCapacity()
        bp = BackpressureController(
            BackpressureConfig(high_watermark_bytes=100, low_watermark_bytes=50),
            capacity=cap,
        )
        cap.resident_bytes = 40
        self.assertFalse(bp.should_pause_prompts())  # below high
        cap.resident_bytes = 100
        self.assertTrue(bp.should_pause_prompts())  # hit high -> pause
        cap.resident_bytes = 70  # between low and high: still paused (latched)
        self.assertTrue(bp.should_pause_prompts())
        cap.resident_bytes = 50  # at/below low -> resume
        self.assertFalse(bp.should_pause_prompts())
        snap = bp.snapshot()
        self.assertEqual(snap["pause_transitions"], 1)
        self.assertEqual(snap["resume_transitions"], 1)

    def test_cap_prompt_grant(self):
        bp = BackpressureController(
            BackpressureConfig(max_inflight_prompts_per_worker=4)
        )
        self.assertEqual(bp.cap_prompt_grant(worker_inflight=1, requested=10), 3)
        self.assertEqual(bp.cap_prompt_grant(worker_inflight=4, requested=10), 0)
        self.assertEqual(bp.cap_prompt_grant(worker_inflight=5, requested=10), 0)

    def test_cap_train_lease(self):
        bp = BackpressureController(BackpressureConfig(max_train_lease=2))
        self.assertEqual(bp.cap_train_lease(10), 2)
        self.assertEqual(bp.cap_train_lease(1), 1)


class TestBackpressureInController(unittest.TestCase):
    def test_paused_rollout_gets_no_prompts_and_counts_starvation(self):
        cap = _FakeCapacity()
        bp = BackpressureController(
            BackpressureConfig(high_watermark_bytes=100, low_watermark_bytes=50),
            capacity=cap,
        )
        ctrl = DataFlowController("run", backpressure=bp)
        ctrl.ingest_prompts([{"task_id": f"t{i}", "payload": {}} for i in range(5)])
        cap.resident_bytes = 100  # over high watermark
        granted = ctrl.lease_prompt_tasks("w0", 5)
        self.assertEqual(granted, [])
        self.assertEqual(ctrl.status()["backpressure"]["rollout_starved"], 1)
        # drains below low watermark -> resumes
        cap.resident_bytes = 10
        granted = ctrl.lease_prompt_tasks("w0", 5)
        self.assertEqual(len(granted), 5)

    def test_per_worker_inflight_cap(self):
        bp = BackpressureController(
            BackpressureConfig(max_inflight_prompts_per_worker=2)
        )
        ctrl = DataFlowController("run", backpressure=bp)
        ctrl.ingest_prompts([{"task_id": f"t{i}", "payload": {}} for i in range(5)])
        first = ctrl.lease_prompt_tasks("w0", 5)
        self.assertEqual(len(first), 2)  # capped at 2 in-flight
        second = ctrl.lease_prompt_tasks("w0", 5)
        self.assertEqual(len(second), 0)  # still 2 in-flight, no headroom
        # a different worker has its own budget
        other = ctrl.lease_prompt_tasks("w1", 5)
        self.assertEqual(len(other), 2)

    def test_trainer_starvation_counted_on_empty_lease(self):
        bp = BackpressureController(BackpressureConfig(max_train_lease=4))
        ctrl = DataFlowController("run", backpressure=bp)
        out = ctrl.lease_train_refs("trainer", 8)  # queue empty
        self.assertEqual(out, [])
        self.assertEqual(ctrl.status()["backpressure"]["trainer_starved"], 1)

    def test_train_lease_capped(self):
        bp = BackpressureController(BackpressureConfig(max_train_lease=2))
        ctrl = DataFlowController("run", backpressure=bp)
        ctrl.enqueue_offline_refs([_ref(f"s{i}") for i in range(5)])
        out = ctrl.lease_train_refs("trainer", 8)
        self.assertEqual(len(out), 2)  # capped

    def test_train_backlog_signal(self):
        ctrl = DataFlowController("run")  # no backpressure needed for this signal
        ctrl.enqueue_offline_refs([_ref(f"s{i}") for i in range(3)])
        self.assertEqual(ctrl.status()["train_backlog"], 3)
        refs = ctrl.lease_train_refs("trainer", 3)
        ctrl.ack_train_refs("trainer", [r.sample_id for r in refs], global_step=1)
        self.assertEqual(ctrl.status()["train_backlog"], 0)

    def test_no_backpressure_is_unchanged_behavior(self):
        ctrl = DataFlowController("run")  # M1-M4 behavior
        ctrl.ingest_prompts([{"task_id": f"t{i}", "payload": {}} for i in range(5)])
        self.assertEqual(len(ctrl.lease_prompt_tasks("w0", 10)), 5)
        self.assertNotIn("backpressure", ctrl.status())


if __name__ == "__main__":
    unittest.main(verbosity=2)
