# coding=utf-8
"""Tests for StreamingRefChannel: the cross-process online ref stream."""

import os
import tempfile
import unittest

from specforge.runtime.contracts import FeatureSpec, SampleRef
from specforge.runtime.data_plane.streaming_ref_channel import (
    StreamingRefChannel,
    StreamingRefQueue,
)


def _ref(sid="s0", gen=1):
    spec = FeatureSpec(name="hidden_state", shape=(4, 8), dtype="float32")
    return SampleRef(
        sample_id=sid,
        run_id="run0",
        source_task_id=None,
        feature_store_uri=f"mooncake://run0/{sid}",
        feature_keys={"hidden_state": f"{sid}/hidden_state"},
        feature_specs={"hidden_state": spec},
        strategy="eagle3",
        num_tokens=4,
        metadata={"generation": gen},
    )


class TestStreamingRefChannel(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="refstream_")
        self.path = os.path.join(self.dir, "stream.jsonl")

    def test_publish_poll_roundtrip(self):
        w = StreamingRefChannel(self.path)
        r = StreamingRefChannel(self.path)  # separate instance = separate process
        w.publish(_ref("s0", 1))
        w.publish(_ref("s1", 2))
        got = r.poll()
        self.assertEqual([x.sample_id for x in got], ["s0", "s1"])
        self.assertEqual(got[0].feature_specs["hidden_state"].shape, (4, 8))
        self.assertEqual(got[1].metadata["generation"], 2)
        # nothing new -> empty
        self.assertEqual(r.poll(), [])

    def test_incremental_poll(self):
        w = StreamingRefChannel(self.path)
        r = StreamingRefChannel(self.path)
        w.publish(_ref("s0"))
        self.assertEqual([x.sample_id for x in r.poll()], ["s0"])
        w.publish(_ref("s1"))
        w.publish(_ref("s2"))
        self.assertEqual([x.sample_id for x in r.poll()], ["s1", "s2"])

    def test_partial_trailing_line_is_not_parsed(self):
        # a half-written record (no newline yet) must not be parsed; it is held
        # until its newline arrives.
        r = StreamingRefChannel(self.path)
        with open(self.path, "w") as f:
            f.write('{"sample_id": "s0", "incomplete')  # torn write, no newline
        self.assertEqual(r.poll(), [])  # nothing complete yet
        # the writer finishes the line with a real ref
        w = StreamingRefChannel(self.path)
        # overwrite with a clean complete line (simulating the real append path)
        with open(self.path, "w") as f:
            f.write("")
        r2 = StreamingRefChannel(self.path)
        w.publish(_ref("s0"))
        self.assertEqual([x.sample_id for x in r2.poll()], ["s0"])

    def test_max_n_limits_batch(self):
        w = StreamingRefChannel(self.path)
        r = StreamingRefChannel(self.path)
        for i in range(5):
            w.publish(_ref(f"s{i}"))
        first = r.poll(max_n=2)
        self.assertEqual(len(first), 2)
        rest = r.poll()
        self.assertEqual(len(rest), 3)

    def test_stream_drains_then_stops_on_close(self):
        w = StreamingRefChannel(self.path)
        r = StreamingRefChannel(self.path)
        for i in range(3):
            w.publish(_ref(f"s{i}"))
        w.close()
        got = list(r.stream(poll_s=0.0))  # closed -> drains 3 then terminates
        self.assertEqual([x.sample_id for x in got], ["s0", "s1", "s2"])

    def test_backpressure_in_flight_remote(self):
        producer = StreamingRefChannel(self.path)
        consumer = StreamingRefChannel(self.path)
        for i in range(4):
            producer.publish(_ref(f"s{i}"))
        self.assertEqual(producer.in_flight_remote(), 4)  # consumer hasn't acked
        got = consumer.poll()
        consumer.mark_consumed(len(got))
        self.assertEqual(producer.consumed_remote(), 4)
        self.assertEqual(producer.in_flight_remote(), 0)

    def test_restart_seeds_consumed_counter_before_new_acks(self):
        first = StreamingRefChannel(self.path)
        first.mark_consumed(3)
        restarted = StreamingRefChannel(self.path)
        self.assertEqual(restarted.seed_consumed(), 3)
        restarted.mark_consumed(2)
        self.assertEqual(first.consumed_remote(), 5)

    def test_stream_idle_timeout_when_producer_silent(self):
        r = StreamingRefChannel(self.path)
        t = {"now": 0.0}

        def clock():
            return t["now"]

        def sleep(dt):
            t["now"] += 10.0  # jump past the timeout on the first idle sleep

        it = r.stream(poll_s=0.1, idle_timeout_s=5.0, clock=clock, sleep=sleep)
        with self.assertRaises(TimeoutError):
            next(it)

    def test_producer_failure_is_not_normal_eof(self):
        producer = StreamingRefChannel(self.path)
        consumer = StreamingRefChannel(self.path)
        producer.publish(_ref("s0"))
        producer.fail("rollout exploded")

        stream = consumer.stream(poll_s=0.0)
        self.assertEqual(next(stream).sample_id, "s0")
        with self.assertRaisesRegex(RuntimeError, "rollout exploded"):
            next(stream)
        self.assertFalse(producer.is_closed())

    def test_queue_failure_wins_over_closed_sentinel(self):
        producer = StreamingRefChannel(self.path)
        consumer = StreamingRefChannel(self.path)
        producer.close()
        producer.fail("partial rollout")

        queue = StreamingRefQueue(consumer, poll_s=0.0)
        with self.assertRaisesRegex(RuntimeError, "partial rollout"):
            queue.get(1)

    def test_consumer_outcome_roundtrip(self):
        producer = StreamingRefChannel(self.path)
        consumer = StreamingRefChannel(self.path)
        self.assertFalse(producer.consumer_stopped())
        consumer.mark_consumer_failed("optimizer failed")
        self.assertTrue(producer.consumer_stopped())
        self.assertEqual(producer.consumer_failure(), "optimizer failed")

        other_path = os.path.join(self.dir, "done.jsonl")
        producer = StreamingRefChannel(other_path)
        consumer = StreamingRefChannel(other_path)
        consumer.mark_consumer_done()
        self.assertTrue(producer.consumer_stopped())
        self.assertIsNone(producer.consumer_failure())

    def test_consumer_quantum_is_single_claimed_attempt_contract(self):
        consumer = StreamingRefChannel(self.path)
        producer = StreamingRefChannel(self.path)
        self.assertIsNone(producer.consumer_quantum())
        consumer.publish_consumer_quantum(32)
        self.assertEqual(producer.consumer_quantum(), 32)
        with self.assertRaisesRegex(ValueError, "fresh reference channel"):
            consumer.publish_consumer_quantum(32)

    def test_resume_accepts_only_the_existing_consumer_quantum(self):
        consumer = StreamingRefChannel(self.path)
        consumer.publish_consumer_quantum(8)
        consumer.publish_consumer_quantum(8, allow_existing=True)
        with self.assertRaisesRegex(ValueError, "changed across resume"):
            consumer.publish_consumer_quantum(16, allow_existing=True)

    def test_queue_get_does_not_advance_consumed_before_explicit_ack(self):
        producer = StreamingRefChannel(self.path)
        producer.publish(_ref("s0"))
        queue = StreamingRefQueue(StreamingRefChannel(self.path), poll_s=0.0)

        refs = queue.get(1)
        self.assertEqual([ref.sample_id for ref in refs], ["s0"])
        self.assertEqual(producer.consumed_remote(), 0)
        self.assertEqual(queue.in_flight_ids(), ["s0"])

        queue.ack_ids(["s0"])
        self.assertEqual(producer.consumed_remote(), 1)
        self.assertEqual(queue.in_flight_ids(), [])


if __name__ == "__main__":
    unittest.main()
