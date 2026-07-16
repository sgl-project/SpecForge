# coding=utf-8
"""Tests for StreamingRefChannel: the cross-process online ref stream."""

import os
import tempfile
import threading
import time
import unittest
from unittest import mock

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


class _FailingPublishChannel(StreamingRefChannel):
    def __init__(self, path, *, fail_after):
        super().__init__(path)
        self.fail_after = fail_after

    def publish(self, ref):
        if self.published >= self.fail_after:
            raise OSError("injected publish failure")
        super().publish(ref)


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

    def test_publish_transaction_exposes_the_whole_batch_before_first_failure(self):
        channel = _FailingPublishChannel(self.path, fail_after=0)
        transaction = channel.begin_publish([_ref("s0"), _ref("s1")])

        with self.assertRaisesRegex(OSError, "injected publish failure"):
            transaction.commit()

        self.assertEqual(transaction.published_refs, ())
        self.assertEqual(
            [ref.sample_id for ref in transaction.unpublished_refs], ["s0", "s1"]
        )

    def test_publish_transaction_exposes_only_the_partial_publish_suffix(self):
        channel = _FailingPublishChannel(self.path, fail_after=1)
        transaction = channel.begin_publish([_ref("s0"), _ref("s1"), _ref("s2")])

        with self.assertRaisesRegex(OSError, "injected publish failure"):
            transaction.commit()

        self.assertEqual([ref.sample_id for ref in transaction.published_refs], ["s0"])
        self.assertEqual(
            [ref.sample_id for ref in transaction.unpublished_refs], ["s1", "s2"]
        )
        self.assertEqual(
            [ref.sample_id for ref in StreamingRefChannel(self.path).poll()], ["s0"]
        )

    def test_fsync_failure_preserves_a_complete_visible_line(self):
        channel = StreamingRefChannel(self.path)
        transaction = channel.begin_publish([_ref("s0"), _ref("s1")])

        with (
            mock.patch(
                "specforge.runtime.data_plane.streaming_ref_channel.os.fsync",
                side_effect=OSError("injected fsync failure"),
            ),
            self.assertRaisesRegex(OSError, "injected fsync failure"),
        ):
            transaction.commit()

        self.assertEqual([ref.sample_id for ref in transaction.published_refs], ["s0"])
        self.assertEqual(
            [ref.sample_id for ref in transaction.unpublished_refs], ["s1"]
        )
        self.assertEqual(
            [ref.sample_id for ref in StreamingRefChannel(self.path).poll()], ["s0"]
        )

    def test_batch_fsync_failure_preserves_all_complete_visible_lines(self):
        channel = StreamingRefChannel(self.path)

        with (
            mock.patch(
                "specforge.runtime.data_plane.streaming_ref_channel.os.fsync",
                side_effect=OSError("injected fsync failure"),
            ),
            self.assertRaisesRegex(OSError, "injected fsync failure"),
        ):
            channel.publish_batch([_ref("s0"), _ref("s1")])

        self.assertEqual(channel.published, 2)
        self.assertEqual(
            [ref.sample_id for ref in StreamingRefChannel(self.path).poll()],
            ["s0", "s1"],
        )

    def test_batch_partial_write_failure_tracks_complete_prefix(self):
        channel = StreamingRefChannel(self.path)
        real_write = os.write
        write_calls = 0

        def write_one_record_then_fail(fd, payload):
            nonlocal write_calls
            write_calls += 1
            if write_calls == 1:
                first_record_end = payload.index(b"\n") + 1
                return real_write(fd, payload[:first_record_end])
            raise OSError("injected batch write failure")

        with (
            mock.patch(
                "specforge.runtime.data_plane.streaming_ref_channel.os.write",
                side_effect=write_one_record_then_fail,
            ),
            self.assertRaisesRegex(OSError, "injected batch write failure"),
        ):
            channel.publish_batch([_ref("s0"), _ref("s1")])

        self.assertEqual(channel.published, 1)
        self.assertEqual(
            [ref.sample_id for ref in StreamingRefChannel(self.path).poll()], ["s0"]
        )

    def test_partial_write_failure_leaves_current_ref_unpublished(self):
        channel = StreamingRefChannel(self.path)
        transaction = channel.begin_publish([_ref("s0"), _ref("s1")])
        real_write = os.write
        write_calls = 0

        def write_partial_then_fail(fd, payload):
            nonlocal write_calls
            write_calls += 1
            if write_calls == 1:
                return real_write(fd, payload[: len(payload) // 2])
            raise OSError("injected partial write failure")

        with (
            mock.patch(
                "specforge.runtime.data_plane.streaming_ref_channel.os.write",
                side_effect=write_partial_then_fail,
            ),
            self.assertRaisesRegex(OSError, "injected partial write failure"),
        ):
            transaction.commit()

        self.assertEqual(transaction.published_refs, ())
        self.assertEqual(
            [ref.sample_id for ref in transaction.unpublished_refs], ["s0", "s1"]
        )
        self.assertEqual(StreamingRefChannel(self.path).poll(), [])

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

    def test_queue_interruptible_get_stops_while_channel_is_open(self):
        queue = StreamingRefQueue(
            StreamingRefChannel(self.path),
            poll_s=0.01,
        )
        stop = threading.Event()
        result = []
        worker = threading.Thread(
            target=lambda: result.append(queue.get_interruptible(1, stop_event=stop))
        )
        worker.start()
        time.sleep(0.02)
        stop.set()
        worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(result, [[]])
        self.assertEqual(queue.in_flight_ids(), [])

    def test_retryable_fail_requeues_without_advancing_consumed_sidecar(self):
        producer = StreamingRefChannel(self.path)
        producer.publish(_ref("s0"))
        queue = StreamingRefQueue(StreamingRefChannel(self.path), poll_s=0.0)

        refs = queue.get(1)
        queue.fail(refs, reason="loader closed", retryable=True)

        self.assertEqual(producer.consumed_remote(), 0)
        self.assertEqual(queue.in_flight_ids(), [])
        self.assertEqual([ref.sample_id for ref in queue.get(1)], ["s0"])


if __name__ == "__main__":
    unittest.main()
