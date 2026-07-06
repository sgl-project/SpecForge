# coding=utf-8
"""Tests for RefDistributor + DPAckController: the DP online consumer's
single-authority dispatch/book-keeping path (CPU only, no torch.distributed)."""

import os
import tempfile
import unittest

from specforge.runtime.contracts import FeatureSpec, SampleRef
from specforge.runtime.control_plane.controller import DataFlowController
from specforge.runtime.control_plane.dp_ack import DPAckController, gather_id_union
from specforge.runtime.control_plane.metadata_store import SQLiteMetadataStore
from specforge.runtime.data_plane.ref_distributor import InboxChannel, RefDistributor
from specforge.runtime.data_plane.streaming_ref_channel import (
    StreamingRefChannel,
    StreamingRefQueue,
)


def _ref(sid):
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
    )


def _pump_until_quiet(dist, max_rounds=20):
    for _ in range(max_rounds):
        if not dist.pump():
            return
    raise AssertionError("distributor never went quiet")


def _inbox_ids(inbox_dir, rank):
    reader = StreamingRefChannel(RefDistributor.inbox_path(inbox_dir, rank))
    return [r.sample_id for r in reader.poll()]


class TestRefDistributor(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="refdist_")
        self.src_path = os.path.join(self.dir, "refs.jsonl")
        self.inbox_dir = os.path.join(self.dir, "inboxes")
        self.producer = StreamingRefChannel(self.src_path)
        self.source = StreamingRefChannel(self.src_path)  # consumer-side view

    def _distributor(self, dp_size=2, controller=None, **kwargs):
        controller = controller or DataFlowController("run0")
        return RefDistributor(
            self.source, controller, self.inbox_dir, dp_size, **kwargs
        )

    def test_round_robin_equal_counts_and_unaligned_tail_dropped(self):
        dist = self._distributor(dp_size=2)
        for i in range(7):
            self.producer.publish(_ref(f"s{i}"))
        _pump_until_quiet(dist)
        # 3 full windows of dp_size=2 dispatched; s6 held (not a full window)
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s0", "s2", "s4"])
        self.assertEqual(_inbox_ids(self.inbox_dir, 1), ["s1", "s3", "s5"])
        self.assertFalse(dist.finished)
        self.producer.close()
        _pump_until_quiet(dist)
        self.assertTrue(dist.finished)
        self.assertEqual(dist.stats["dropped"], 1)  # s6: replayed on restart
        for rank in range(2):
            reader = StreamingRefChannel(
                RefDistributor.inbox_path(self.inbox_dir, rank)
            )
            self.assertTrue(reader.is_closed())

    def test_inbox_readers_get_disjoint_shards_via_queue(self):
        dist = self._distributor(dp_size=2)
        for i in range(4):
            self.producer.publish(_ref(f"s{i}"))
        self.producer.close()
        _pump_until_quiet(dist)
        q0 = StreamingRefQueue(
            StreamingRefChannel(RefDistributor.inbox_path(self.inbox_dir, 0))
        )
        q1 = StreamingRefQueue(
            StreamingRefChannel(RefDistributor.inbox_path(self.inbox_dir, 1))
        )
        ids0 = {r.sample_id for r in q0.get(2)}
        ids1 = {r.sample_id for r in q1.get(2)}
        self.assertEqual(ids0 & ids1, set())
        self.assertEqual(ids0 | ids1, {"s0", "s1", "s2", "s3"})
        # closed-and-drained: both readers terminate identically
        self.assertEqual(q0.get(1), [])
        self.assertEqual(q1.get(1), [])

    def test_duplicate_publication_dispatched_once(self):
        dist = self._distributor(dp_size=1)
        self.producer.publish(_ref("s0"))
        self.producer.publish(_ref("s0"))  # producer-restart republication
        self.producer.publish(_ref("s1"))
        self.producer.close()
        _pump_until_quiet(dist)
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s0", "s1"])
        self.assertEqual(dist.stats["duplicates"], 1)

    def test_skip_ids_never_dispatch_and_are_not_recounted(self):
        # Released refs were already counted by the prior run's acks (the
        # sidecar seed); the skip filter must not count them again.
        prior = StreamingRefChannel(self.src_path)
        prior.mark_consumed(2)  # run 1 counted s0, s1
        dist = self._distributor(dp_size=2, skip_ids={"s0", "s1"})
        for i in range(4):
            self.producer.publish(_ref(f"s{i}"))
        self.producer.close()
        _pump_until_quiet(dist)
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s2"])
        self.assertEqual(_inbox_ids(self.inbox_dir, 1), ["s3"])
        self.assertEqual(dist.stats["skipped"], 2)
        self.assertEqual(self.producer.consumed_remote(), 2)  # seed only, no double

    def test_inbox_acks_forward_to_source_counter(self):
        dist = self._distributor(dp_size=2)
        for i in range(4):
            self.producer.publish(_ref(f"s{i}"))
        _pump_until_quiet(dist)
        self.assertEqual(self.producer.consumed_remote(), 0)  # dispatch != consumed
        # rank readers ack their inboxes per micro-batch (loader behavior)
        for rank in range(2):
            q = StreamingRefQueue(
                StreamingRefChannel(RefDistributor.inbox_path(self.inbox_dir, rank))
            )
            q.ack(q.get(2))
        _pump_until_quiet(dist)
        self.assertEqual(self.producer.consumed_remote(), 4)

    def test_consumed_counter_survives_restart(self):
        # First consumer attempt marks 3 consumed, then dies.
        first = StreamingRefChannel(self.src_path)
        first.mark_consumed(3)
        self.assertEqual(self.producer.consumed_remote(), 3)
        # A restarted distributor must seed from the sidecar, not rewind it:
        # one new inbox ack lands on top of the seed.
        dist = self._distributor(dp_size=1)
        self.producer.publish(_ref("s9"))
        _pump_until_quiet(dist)
        q = StreamingRefQueue(
            StreamingRefChannel(RefDistributor.inbox_path(self.inbox_dir, 0))
        )
        q.ack(q.get(1))
        _pump_until_quiet(dist)
        self.assertEqual(self.producer.consumed_remote(), 4)

    def test_stale_inbox_files_recreated_fresh(self):
        os.makedirs(self.inbox_dir, exist_ok=True)
        stale = RefDistributor.inbox_path(self.inbox_dir, 0)
        StreamingRefChannel(stale).publish(_ref("stale"))
        open(stale + ".closed", "w").close()
        dist = self._distributor(dp_size=1)
        self.producer.publish(_ref("fresh"))
        self.producer.close()
        _pump_until_quiet(dist)
        reader = StreamingRefChannel(stale)
        self.assertEqual([r.sample_id for r in reader.poll()], ["fresh"])

    def test_resume_requeues_committed_unacked_and_skips_released(self):
        db = os.path.join(self.dir, "run.db")
        # Prior attempt: committed s0..s3; s0,s1 durably acked (released).
        prior = DataFlowController("run0", metadata_store=SQLiteMetadataStore(db))
        prior.commit_samples("w0", [_ref(f"s{i}") for i in range(4)])
        prior.ack_train_refs("t0", ["s0", "s1"], global_step=1, optimizer_durable=True)
        prior.store.close()
        # Restart: reconcile requeues committed-unacked (s2, s3) into the new
        # controller's queue; released (s0, s1) become skip_ids.
        controller = DataFlowController("run0", metadata_store=SQLiteMetadataStore(db))
        reconciled = controller.reconcile_on_restart()
        self.assertEqual(sorted(reconciled["released"]), ["s0", "s1"])
        self.assertEqual(sorted(reconciled["requeued"]), ["s2", "s3"])
        dist = self._distributor(
            dp_size=2, controller=controller, skip_ids=set(reconciled["released"])
        )
        # The channel replays everything from offset 0 (restart re-read).
        for i in range(4):
            self.producer.publish(_ref(f"s{i}"))
        self.producer.close()
        _pump_until_quiet(dist)
        ids0, ids1 = _inbox_ids(self.inbox_dir, 0), _inbox_ids(self.inbox_dir, 1)
        # exactly the unacked tail trains again, split evenly, no duplicates
        self.assertEqual(sorted(ids0 + ids1), ["s2", "s3"])
        self.assertEqual(len(ids0), len(ids1))
        self.assertEqual(dist.stats["skipped"], 2)
        controller.store.close()

    def test_end_of_stream_partial_window_releases_leases(self):
        controller = DataFlowController("run0")
        dist = self._distributor(dp_size=2, controller=controller)
        for i in range(3):  # one full window + one leftover
            self.producer.publish(_ref(f"s{i}"))
        self.producer.close()
        _pump_until_quiet(dist)
        self.assertTrue(dist.finished)
        self.assertEqual(dist.stats["dropped"], 1)
        # the dropped ref's lease is released, not leaked
        self.assertEqual(controller.sample_queue.in_flight(), 2)  # dispatched only
        self.assertEqual(controller.sample_queue.depth(), 0)

    def test_distributor_death_poisons_inboxes(self):
        clock = {"t": 0.0}
        dist = self._distributor(
            dp_size=2,
            idle_timeout_s=5.0,
            clock=lambda: clock["t"],
            sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
        )
        dist._run_guarded()  # idle-timeout kills it (producer silent)
        self.assertIsInstance(dist.error, TimeoutError)
        for rank in range(2):
            reader = InboxChannel(RefDistributor.inbox_path(self.inbox_dir, rank))
            with self.assertRaisesRegex(RuntimeError, "ref-distributor died"):
                StreamingRefQueue(reader).get(1)

    def test_idle_timeout_raises_when_producer_silent(self):
        clock = {"t": 0.0}
        dist = self._distributor(
            dp_size=1,
            idle_timeout_s=5.0,
            clock=lambda: clock["t"],
            sleep=lambda s: clock.__setitem__("t", clock["t"] + s),
        )
        with self.assertRaises(TimeoutError):
            dist.run()


class TestDPAckController(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="dpack_")

    def test_authority_records_gathered_union(self):
        gathered = lambda ids: ids + ["s-other-rank"]  # noqa: E731
        controller = DPAckController(
            "run0",
            is_authority=True,
            gather=gathered,
            metadata_store=SQLiteMetadataStore(os.path.join(self.dir, "run.db")),
        )
        controller.commit_samples("w0", [_ref("s0"), _ref("s-other-rank")])
        controller.ack_train_refs("t0", ["s0"], global_step=1, optimizer_durable=True)
        marker = controller.store.durable_marker()
        self.assertEqual(sorted(marker["acked"]), ["s-other-rank", "s0"])
        self.assertTrue(marker["optimizer_durable"])
        controller.store.close()

    def test_non_authority_participates_but_records_nothing(self):
        calls = []

        def gather(ids):
            calls.append(list(ids))
            return ids

        controller = DPAckController("run0", is_authority=False, gather=gather)
        controller.ack_train_refs("t0", ["s0"], global_step=1, optimizer_durable=True)
        self.assertEqual(calls, [["s0"]])  # joined the collective
        self.assertEqual(controller.store.durable_marker()["acked"], set())

    def test_gather_id_union_without_dist_is_identity(self):
        self.assertEqual(gather_id_union(["a", "b"]), ["a", "b"])


if __name__ == "__main__":
    unittest.main()
