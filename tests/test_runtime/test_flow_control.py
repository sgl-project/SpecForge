# coding=utf-8
"""Canonical producer flow-control policy (CPU-only)."""

import unittest

from specforge.runtime.control_plane.flow_control import (
    FlowControlLimits,
    ProducerFlowControl,
)


class FlowControlTest(unittest.TestCase):
    def test_reference_and_byte_hysteresis_share_one_pause_latch(self):
        policy = ProducerFlowControl(
            FlowControlLimits(
                high_watermark_refs=10,
                low_watermark_refs=4,
                high_watermark_bytes=100,
                low_watermark_bytes=40,
            )
        )
        self.assertFalse(policy.should_pause(in_flight_refs=9, resident_bytes=99))
        self.assertTrue(policy.should_pause(in_flight_refs=10, resident_bytes=20))
        self.assertTrue(policy.should_pause(in_flight_refs=5, resident_bytes=20))
        self.assertFalse(policy.should_pause(in_flight_refs=4, resident_bytes=40))
        self.assertTrue(policy.should_pause(in_flight_refs=1, resident_bytes=100))
        self.assertTrue(policy.should_pause(in_flight_refs=1, resident_bytes=50))
        self.assertFalse(policy.should_pause(in_flight_refs=1, resident_bytes=40))

        snapshot = policy.snapshot(in_flight_refs=1, resident_bytes=40)
        self.assertEqual(snapshot["pause_transitions"], 2)
        self.assertEqual(snapshot["resume_transitions"], 2)
        self.assertGreater(snapshot["wait_checks"], 0)

    def test_prompt_lease_is_capped_per_worker(self):
        policy = ProducerFlowControl(FlowControlLimits(max_prompt_lease_per_worker=3))
        self.assertEqual(policy.prompt_lease(8), 3)
        self.assertEqual(policy.prompt_lease(2), 2)

    def test_invalid_watermarks_are_rejected(self):
        with self.assertRaises(ValueError):
            FlowControlLimits(high_watermark_refs=4, low_watermark_refs=5)
        with self.assertRaises(ValueError):
            FlowControlLimits(low_watermark_bytes=10)
        with self.assertRaises(ValueError):
            FlowControlLimits(
                high_watermark_bytes=10,
                low_watermark_bytes=11,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
