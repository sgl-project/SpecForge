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


class _AbortStore:
    def __init__(self):
        self.aborted = []

    def abort(self, sample_id, reason):
        self.aborted.append((sample_id, reason))


class TestRefDistributor(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="refdist_")
        self.src_path = os.path.join(self.dir, "refs.jsonl")
        self.inbox_dir = os.path.join(self.dir, "inboxes")
        self.producer = StreamingRefChannel(self.src_path)
        self.source = StreamingRefChannel(self.src_path)  # consumer-side view
        self.feature_store = _AbortStore()

    def _distributor(self, dp_size=2, controller=None, **kwargs):
        controller = controller or DataFlowController("run0")
        refs_per_rank_step = kwargs.pop("refs_per_rank_step", 1)
        return RefDistributor(
            self.source,
            controller,
            self.inbox_dir,
            dp_size,
            feature_store=self.feature_store,
            refs_per_rank_step=refs_per_rank_step,
            **kwargs,
        )

    def test_round_robin_equal_counts_and_unaligned_tail_fails(self):
        dist = self._distributor(dp_size=2)
        for i in range(7):
            self.producer.publish(_ref(f"s{i}"))
        _pump_until_quiet(dist)
        # 3 full windows of dp_size=2 dispatched; s6 held (not a full window)
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s0", "s2", "s4"])
        self.assertEqual(_inbox_ids(self.inbox_dir, 1), ["s1", "s3", "s5"])
        self.assertFalse(dist.finished)
        self.producer.close()
        dist._run_guarded()
        self.assertIsInstance(dist.error, RuntimeError)
        self.assertEqual(dist.stats["dropped"], 1)
        self.assertEqual([item[0] for item in self.feature_store.aborted], ["s6"])
        for rank in range(2):
            reader = InboxChannel(RefDistributor.inbox_path(self.inbox_dir, rank))
            with self.assertRaisesRegex(RuntimeError, "optimizer window"):
                reader.is_closed()

    def test_inbox_readers_get_disjoint_shards_via_queue(self):
        dist = self._distributor(dp_size=2)
        for i in range(4):
            self.producer.publish(_ref(f"s{i}"))
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
        self.producer.close()
        _pump_until_quiet(dist)
        self.assertEqual(q0.get(1), [])
        self.assertEqual(q1.get(1), [])

    def test_duplicate_publication_dispatched_once(self):
        dist = self._distributor(dp_size=1)
        self.producer.publish(_ref("s0"))
        self.producer.publish(_ref("s0"))  # at-least-once duplicate
        self.producer.publish(_ref("s1"))
        _pump_until_quiet(dist)
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s0", "s1"])
        self.assertEqual(dist.stats["duplicates"], 1)
        self.assertEqual(self.producer.consumed_remote(), 1)

        queue = StreamingRefQueue(
            StreamingRefChannel(RefDistributor.inbox_path(self.inbox_dir, 0))
        )
        refs = queue.get(2)
        queue.ack(refs)
        _pump_until_quiet(dist)
        self.assertEqual(self.producer.consumed_remote(), 3)
        self.assertEqual(self.producer.in_flight_remote(), 0)
        self.producer.close()
        _pump_until_quiet(dist)

    def test_resume_skips_acked_and_dispatches_reconciled_unacked(self):
        ledger_path = os.path.join(self.dir, "resume.sqlite")
        store = SQLiteMetadataStore(ledger_path)
        original = DataFlowController("run0", metadata_store=store)
        refs = [_ref("s0"), _ref("s1")]
        for ref in refs:
            self.producer.publish(ref)
        original.commit_samples("old-distributor", refs)
        original.ack_train_refs(
            "trainer", ["s0"], global_step=1, optimizer_durable=True
        )
        # The prior consumer counted only the durably trained prefix.
        self.source.mark_consumed(1)

        restarted = DataFlowController("run0", metadata_store=store)
        report = restarted.reconcile_on_restart(self.feature_store)
        dist = self._distributor(
            dp_size=1,
            controller=restarted,
            skip_ids=report["released"],
            requeued_ids=report["requeued"],
        )
        _pump_until_quiet(dist)

        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s1"])
        self.assertEqual(dist.stats["skipped"], 1)
        self.assertEqual(dist.stats["duplicates"], 1)
        queue = StreamingRefQueue(
            StreamingRefChannel(RefDistributor.inbox_path(self.inbox_dir, 0))
        )
        queue.ack(queue.get(1))
        _pump_until_quiet(dist)
        self.assertEqual(self.producer.consumed_remote(), 2)
        self.producer.close()
        _pump_until_quiet(dist)
        store.close()

    def test_resume_repairs_ack_committed_before_inbox_counter(self):
        ledger_path = os.path.join(self.dir, "resume-counter.sqlite")
        store = SQLiteMetadataStore(ledger_path)
        original = DataFlowController("run0", metadata_store=store)
        refs = [_ref("s0"), _ref("s1")]
        for ref in refs:
            self.producer.publish(ref)
        original.commit_samples("old-distributor", refs)
        original.ack_train_refs(
            "trainer", ["s0"], global_step=1, optimizer_durable=True
        )
        # Crash here: the SQLite marker exists, but the rank-local inbox ack did
        # not yet advance the source sidecar.
        self.assertEqual(self.producer.consumed_remote(), 0)

        restarted = DataFlowController("run0", metadata_store=store)
        report = restarted.reconcile_on_restart(self.feature_store)
        dist = self._distributor(
            dp_size=1,
            controller=restarted,
            skip_ids=report["released"],
            requeued_ids=report["requeued"],
        )
        self.assertEqual(self.producer.consumed_remote(), 1)
        _pump_until_quiet(dist)
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s1"])
        store.close()

    def test_reconciled_unacked_refs_redistribute_across_consumer_dp_sizes(self):
        """A fresh authority may repartition replayable refs after a DP change.

        This is deliberately a control/ref-plane guarantee. It does not load or
        reshard an FSDP model/optimizer checkpoint, whose world-size contract is
        validated separately by the Trainer resume path.
        """
        ledger_path = os.path.join(self.dir, "consumer-dp-restart.sqlite")
        original_store = SQLiteMetadataStore(ledger_path)
        original = DataFlowController("run0", metadata_store=original_store)
        for index in range(6):
            self.producer.publish(_ref(f"s{index}"))

        before_restart = self._distributor(dp_size=2, controller=original)
        _pump_until_quiet(before_restart)
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s0", "s2", "s4"])
        self.assertEqual(_inbox_ids(self.inbox_dir, 1), ["s1", "s3", "s5"])
        self.assertEqual(before_restart.stats["dispatched"], 6)

        # Crash before any optimizer-durable ack. Reopen both durable ledger and
        # source reader as a new consumer attempt would; ephemeral DP=2 inboxes
        # must not define the replay partition.
        original_store.close()
        reopened_store = SQLiteMetadataStore(ledger_path)
        restarted = DataFlowController("run0", metadata_store=reopened_store)
        report = restarted.reconcile_on_restart(self.feature_store)
        self.assertEqual(set(report["requeued"]), {f"s{i}" for i in range(6)})
        self.assertEqual(report["released"], [])

        self.source = StreamingRefChannel(self.src_path)
        after_restart = self._distributor(
            dp_size=3,
            controller=restarted,
            skip_ids=report["released"],
            requeued_ids=report["requeued"],
        )
        _pump_until_quiet(after_restart)

        shards = [set(_inbox_ids(self.inbox_dir, rank)) for rank in range(3)]
        self.assertEqual(shards, [{"s0", "s3"}, {"s1", "s4"}, {"s2", "s5"}])
        self.assertTrue(all(len(shard) == 2 for shard in shards))
        self.assertEqual(set.union(*shards), {f"s{i}" for i in range(6)})
        for index, left in enumerate(shards):
            for right in shards[index + 1 :]:
                self.assertTrue(left.isdisjoint(right))
        self.assertEqual(after_restart.stats["dispatched"], 6)
        self.assertEqual(after_restart.stats["duplicates"], 6)

        # The new three-rank partition settles the original producer stream once.
        for rank in range(3):
            queue = StreamingRefQueue(
                StreamingRefChannel(RefDistributor.inbox_path(self.inbox_dir, rank))
            )
            refs = queue.get(2)
            self.assertEqual(len(refs), 2)
            queue.ack(refs)
        _pump_until_quiet(after_restart)
        self.assertEqual(self.producer.consumed_remote(), 6)
        self.producer.close()
        _pump_until_quiet(after_restart)
        reopened_store.close()

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

    def test_end_of_stream_partial_window_releases_leases_and_features(self):
        controller = DataFlowController("run0")
        dist = self._distributor(dp_size=2, controller=controller)
        for i in range(3):  # one full window + one leftover
            self.producer.publish(_ref(f"s{i}"))
        self.producer.close()
        dist._run_guarded()
        self.assertIsInstance(dist.error, RuntimeError)
        self.assertEqual(dist.stats["dropped"], 1)
        # the dropped ref's lease is released, not leaked
        self.assertEqual(controller.sample_queue.in_flight(), 2)  # dispatched only
        self.assertEqual(controller.sample_queue.depth(), 0)
        self.assertEqual([item[0] for item in self.feature_store.aborted], ["s2"])
        self.assertEqual(self.producer.consumed_remote(), 1)

    def test_dispatches_only_complete_optimizer_step_windows(self):
        dist = self._distributor(dp_size=2, refs_per_rank_step=4)
        for i in range(8):
            self.producer.publish(_ref(f"s{i}"))
        _pump_until_quiet(dist)
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s0", "s2", "s4", "s6"])
        self.assertEqual(_inbox_ids(self.inbox_dir, 1), ["s1", "s3", "s5", "s7"])

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

    def test_producer_failure_immediately_poisons_all_inboxes(self):
        dist = self._distributor(dp_size=2)
        self.producer.fail("capture server died")
        dist._run_guarded()
        self.assertIsInstance(dist.error, RuntimeError)
        self.assertIn("capture server died", str(dist.error))
        for rank in range(2):
            reader = InboxChannel(RefDistributor.inbox_path(self.inbox_dir, rank))
            with self.assertRaisesRegex(RuntimeError, "capture server died"):
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

    def test_feature_abort_happens_after_durable_marker(self):
        observations = []
        controller = None

        class FeatureStore:
            def abort(self, sample_id, *, reason):
                marker = controller.store.durable_marker()
                observations.append((sample_id, marker, reason))

        controller = DPAckController(
            "run0",
            is_authority=True,
            feature_store=FeatureStore(),
            metadata_store=SQLiteMetadataStore(os.path.join(self.dir, "order.db")),
        )
        controller.commit_samples("w0", [_ref("s0")])
        controller.ack_train_refs("t0", ["s0"], global_step=3, optimizer_durable=True)

        self.assertEqual(len(observations), 1)
        sample_id, marker, reason = observations[0]
        self.assertEqual(sample_id, "s0")
        self.assertEqual(marker["global_step"], 3)
        self.assertTrue(marker["optimizer_durable"])
        self.assertIn("s0", marker["acked"])
        self.assertEqual(reason, "optimizer-boundary-durable-ack")
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
