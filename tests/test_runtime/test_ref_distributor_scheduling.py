# coding=utf-8
"""RefDistributor with cost-aware scheduling: planned windows, fallbacks,
and unchanged end-of-stream/ack accounting (CPU only)."""

import os
import tempfile
import unittest

from specforge.runtime.contracts import FeatureSpec, SampleRef
from specforge.runtime.control_plane.controller import DataFlowController
from specforge.runtime.data_plane.ref_distributor import InboxChannel, RefDistributor
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from specforge.runtime.data_plane.window_plan import (
    VALID_ANCHORS_METADATA_KEY,
    SchedulingPolicy,
)


def _ref(sid, num_tokens, valid_anchors=None):
    spec = FeatureSpec(name="hidden_state", shape=(4, 8), dtype="float32")
    metadata = {}
    if valid_anchors is not None:
        metadata[VALID_ANCHORS_METADATA_KEY] = valid_anchors
    return SampleRef(
        sample_id=sid,
        run_id="run0",
        source_task_id=None,
        feature_store_uri=f"mooncake://run0/{sid}",
        feature_keys={"hidden_state": f"{sid}/hidden_state"},
        feature_specs={"hidden_state": spec},
        strategy="dspark",
        num_tokens=num_tokens,
        metadata=metadata,
    )


def _pump_until_quiet(dist, max_rounds=20):
    for _ in range(max_rounds):
        if not dist.pump():
            return
    raise AssertionError("distributor never went quiet")


def _inbox_ids(inbox_dir, rank):
    reader = StreamingRefChannel(RefDistributor.inbox_path(inbox_dir, rank))
    return [r.sample_id for r in reader.poll()]


class _AbortStore:
    def __init__(self):
        self.aborted = []

    def abort(self, sample_id, reason=""):
        self.aborted.append((sample_id, reason))


class TestRefDistributorScheduling(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="refdist_sched_")
        self.src_path = os.path.join(self.dir, "refs.jsonl")
        self.inbox_dir = os.path.join(self.dir, "inboxes")
        self.producer = StreamingRefChannel(self.src_path)
        self.source = StreamingRefChannel(self.src_path)
        self.feature_store = _AbortStore()

    def _distributor(
        self, policy, dp_size=2, refs_per_rank_step=2, refs_per_rank_batch=1
    ):
        return RefDistributor(
            self.source,
            DataFlowController("run0"),
            self.inbox_dir,
            dp_size,
            feature_store=self.feature_store,
            refs_per_rank_step=refs_per_rank_step,
            refs_per_rank_batch=refs_per_rank_batch,
            scheduling=policy,
        )

    def test_window_is_sorted_into_contiguous_rank_slices(self):
        dist = self._distributor(SchedulingPolicy("length"))
        for sid, n in (("s0", 1), ("s1", 8), ("s2", 3), ("s3", 6)):
            self.producer.publish(_ref(sid, n))
        _pump_until_quiet(dist)
        # sorted by length: s1(8) s3(6) | s2(3) s0(1) -> round 0 = s1/s3, round 1 = s2/s0
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s1", "s2"])
        self.assertEqual(_inbox_ids(self.inbox_dir, 1), ["s3", "s0"])
        self.assertEqual(dist.stats["dispatched"], 4)
        self.assertEqual(dist.stats["planned_windows"], 1)
        self.assertEqual(dist.stats["fifo_windows"], 0)

    def test_nothing_is_dispatched_before_the_whole_window_is_committed(self):
        dist = self._distributor(SchedulingPolicy("length"))
        for sid, n in (("s0", 1), ("s1", 8), ("s2", 3)):
            self.producer.publish(_ref(sid, n))
        _pump_until_quiet(dist)
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), [])
        self.assertEqual(dist.stats["dispatched"], 0)
        self.producer.publish(_ref("s3", 6))
        _pump_until_quiet(dist)
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s1", "s2"])
        self.assertEqual(_inbox_ids(self.inbox_dir, 1), ["s3", "s0"])

    def test_anchor_key_uses_metadata_with_sample_id_tiebreak(self):
        dist = self._distributor(SchedulingPolicy("anchors", num_anchors=512))
        for sid, n, anchors in (
            ("s0", 8000, 100),
            ("s1", 1000, 512),
            ("s2", 5000, 300),
            ("s3", 2000, 700),  # capped to 512: ties with s1, s1 < s3 by id
        ):
            self.producer.publish(_ref(sid, n, anchors))
        _pump_until_quiet(dist)
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s1", "s2"])
        self.assertEqual(_inbox_ids(self.inbox_dir, 1), ["s3", "s0"])

    def test_missing_anchor_metadata_falls_back_to_length(self):
        dist = self._distributor(SchedulingPolicy("anchors"))
        for sid, n in (("s0", 1), ("s1", 8), ("s2", 3), ("s3", 6)):
            self.producer.publish(_ref(sid, n))
        _pump_until_quiet(dist)
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s1", "s2"])
        self.assertEqual(_inbox_ids(self.inbox_dir, 1), ["s3", "s0"])
        self.assertEqual(dist.stats["length_fallback_windows"], 1)
        self.assertEqual(dist.stats["planned_windows"], 1)

    def test_refs_without_any_cost_keep_arrival_order(self):
        dist = self._distributor(SchedulingPolicy("length"))
        for sid in ("s0", "s1", "s2", "s3"):
            self.producer.publish(_ref(sid, 0))
        _pump_until_quiet(dist)
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s0", "s2"])
        self.assertEqual(_inbox_ids(self.inbox_dir, 1), ["s1", "s3"])
        self.assertEqual(dist.stats["fifo_windows"], 1)
        self.assertEqual(dist.stats["planned_windows"], 0)

    def test_batches_of_two_and_second_window_planned_independently(self):
        dist = self._distributor(
            SchedulingPolicy("length"), refs_per_rank_step=4, refs_per_rank_batch=2
        )
        lengths = [5, 1, 8, 7, 2, 6, 4, 3, 30, 10, 20, 40, 11, 12, 13, 14]
        for i, n in enumerate(lengths):
            self.producer.publish(_ref(f"s{i}", n))
        _pump_until_quiet(dist)
        # window 1 sorted: 8 7 6 5 | 4 3 2 1 ; window 2: 40 30 20 14 | 13 12 11 10
        self.assertEqual(
            _inbox_ids(self.inbox_dir, 0),
            ["s2", "s3", "s6", "s7", "s11", "s8", "s14", "s13"],
        )
        self.assertEqual(
            _inbox_ids(self.inbox_dir, 1),
            ["s5", "s0", "s4", "s1", "s10", "s15", "s12", "s9"],
        )
        self.assertEqual(dist.stats["planned_windows"], 2)
        self.assertEqual(dist.stats["dispatched"], 16)

    def test_end_of_stream_tail_is_settled_like_the_fifo_path(self):
        dist = self._distributor(SchedulingPolicy("length"))
        for i in range(5):
            self.producer.publish(_ref(f"s{i}", 10 - i))
        _pump_until_quiet(dist)
        self.assertEqual(dist.stats["dispatched"], 4)
        self.producer.close()
        dist._run_guarded()
        self.assertIsNone(dist.error)
        self.assertTrue(dist.finished)
        self.assertEqual(dist.stats["dropped"], 1)
        self.assertEqual([item[0] for item in self.feature_store.aborted], ["s4"])
        for rank in range(2):
            reader = InboxChannel(RefDistributor.inbox_path(self.inbox_dir, rank))
            self.assertTrue(reader.is_closed())


if __name__ == "__main__":
    unittest.main()
