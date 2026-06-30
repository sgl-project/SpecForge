# coding=utf-8
"""Tests for StreamingRefChannel: the cross-process online ref stream."""

import os
import tempfile
import unittest

from specforge.runtime.contracts import FeatureSpec, SampleRef
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel


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


if __name__ == "__main__":
    unittest.main()
