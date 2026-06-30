# coding=utf-8
"""SampleRefQueue lease / ack / fail / depth semantics (CPU-only)."""

import unittest

from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue


def _ref(i: int) -> SampleRef:
    return SampleRef(
        sample_id=f"s{i}",
        run_id="r",
        source_task_id=None,
        feature_store_uri=f"mem://x/s{i}",
        feature_keys={},
        feature_specs={},
        strategy="eagle3",
    )


class TestSampleRefQueue(unittest.TestCase):
    def test_put_get_ack_depth(self):
        q = SampleRefQueue()
        q.put([_ref(0), _ref(1), _ref(2)])
        self.assertEqual(q.depth(), 3)
        got = q.get(2)
        self.assertEqual([r.sample_id for r in got], ["s0", "s1"])
        self.assertEqual(q.depth(), 1)
        self.assertEqual(q.in_flight(), 2)
        q.ack(got)
        self.assertEqual(q.in_flight(), 0)

    def test_put_is_idempotent_on_sample_id(self):
        q = SampleRefQueue()
        q.put([_ref(0)])
        q.put([_ref(0)])  # duplicate sample_id ignored (at-least-once)
        self.assertEqual(q.depth(), 1)

    def test_fail_retryable_requeues(self):
        q = SampleRefQueue()
        q.put([_ref(0)])
        got = q.get(1)
        q.fail(got, reason="boom", retryable=True)
        self.assertEqual(q.depth(), 1)
        again = q.get(1)
        self.assertEqual(again[0].sample_id, "s0")

    def test_fail_terminal_drops(self):
        q = SampleRefQueue()
        q.put([_ref(0)])
        got = q.get(1)
        q.fail(got, reason="corrupt", retryable=False)
        self.assertEqual(q.depth(), 0)
        self.assertEqual(q.in_flight(), 0)

    def test_get_empty_returns_immediately(self):
        q = SampleRefQueue()
        self.assertEqual(q.get(4, timeout_s=0.0), [])

    def test_ack_unknown_is_noop(self):
        q = SampleRefQueue()
        q.ack([_ref(99)])  # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
