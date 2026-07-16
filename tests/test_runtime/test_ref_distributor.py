# coding=utf-8
"""Tests for RefDistributor + DPAckController: the DP online consumer's
single-authority dispatch/book-keeping path (CPU only, no torch.distributed)."""

import os
import sqlite3
import tempfile
import unittest
from unittest import mock

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
        self.adopted = []

    def adopt(self, ref):
        self.adopted.append(ref.sample_id)

    def abort(self, sample_id, reason):
        self.aborted.append((sample_id, reason))


class _FailingAbortStore(_AbortStore):
    def abort(self, sample_id, reason):
        super().abort(sample_id, reason)
        raise RuntimeError("injected abort failure")


class _CountForbiddenSQLiteMetadataStore(SQLiteMetadataStore):
    def committed_count(self):
        raise AssertionError("RefDistributor must use the commit freshness result")


class _CloseAfterFirstEmptyPollChannel(StreamingRefChannel):
    """Inject a final publication between a consumer poll and close check."""

    def __init__(self, path, producer, final_ref):
        super().__init__(path)
        self._producer = producer
        self._final_ref = final_ref
        self._injected = False

    def poll(self, max_n=None):
        refs = super().poll(max_n)
        if not self._injected and not refs:
            self._injected = True
            self._producer.publish(self._final_ref)
            self._producer.close()
        return refs


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

    def test_round_robin_equal_counts_and_unaligned_tail_closes_cleanly(self):
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
        self.assertIsNone(dist.error)
        self.assertTrue(dist.finished)
        self.assertEqual(dist.stats["dropped"], 1)
        self.assertEqual([item[0] for item in self.feature_store.aborted], ["s6"])
        for rank in range(2):
            reader = InboxChannel(RefDistributor.inbox_path(self.inbox_dir, rank))
            self.assertTrue(reader.is_closed())

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

    def test_dedup_uses_commit_result_without_counting_the_ledger(self):
        store = _CountForbiddenSQLiteMetadataStore(
            os.path.join(self.dir, "no-count.sqlite")
        )
        self.addCleanup(store.close)
        controller = DataFlowController("run0", metadata_store=store)
        dist = self._distributor(dp_size=1, controller=controller)
        self.producer.publish(_ref("s0"))
        self.producer.publish(_ref("s0"))
        self.producer.close()

        _pump_until_quiet(dist)

        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s0"])
        self.assertEqual(dist.stats["duplicates"], 1)

    def test_poll_batches_one_db_commit_and_one_inbox_fsync_per_rank_window(self):
        store = SQLiteMetadataStore(os.path.join(self.dir, "batched.sqlite"))
        self.addCleanup(store.close)
        controller = DataFlowController("run0", metadata_store=store)
        dist = self._distributor(
            dp_size=2,
            controller=controller,
            refs_per_rank_step=2,
        )
        # Eight fresh refs form two global windows. The repeated s0 exercises
        # same-poll ledger dedup without adding an inbox publication.
        for sample_id in ["s0", "s0", *[f"s{i}" for i in range(1, 8)]]:
            self.producer.publish(_ref(sample_id))

        sql = []
        store._conn.set_trace_callback(sql.append)
        batch_calls = []
        for rank, inbox in enumerate(dist._inboxes):
            publish_batch = inbox.publish_batch

            def record_batch(refs, *, rank=rank, publish_batch=publish_batch):
                batch_calls.append((rank, [ref.sample_id for ref in refs]))
                publish_batch(refs)

            inbox.publish_batch = record_batch

        with mock.patch(
            "specforge.runtime.data_plane.streaming_ref_channel.os.fsync"
        ) as fsync:
            self.assertTrue(dist.pump())

        commits = [statement for statement in sql if statement.strip() == "COMMIT"]
        self.assertEqual(len(commits), 1)
        self.assertEqual(
            batch_calls,
            [
                (0, ["s0", "s2"]),
                (1, ["s1", "s3"]),
                (0, ["s4", "s6"]),
                (1, ["s5", "s7"]),
            ],
        )
        self.assertEqual(fsync.call_count, 4)
        self.assertEqual(dist.stats["duplicates"], 1)
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s0", "s2", "s4", "s6"])
        self.assertEqual(_inbox_ids(self.inbox_dir, 1), ["s1", "s3", "s5", "s7"])

    def test_sqlite_bulk_commit_rolls_back_the_whole_poll_on_insert_failure(self):
        store = SQLiteMetadataStore(os.path.join(self.dir, "rollback.sqlite"))
        self.addCleanup(store.close)
        store._conn.execute(
            "CREATE TRIGGER reject_s1 BEFORE INSERT ON committed "
            "WHEN NEW.sample_id = 's1' BEGIN "
            "SELECT RAISE(ABORT, 'injected insert failure'); END"
        )
        store._conn.commit()

        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected insert failure"):
            store.commit_samples([_ref("s0"), _ref("s1")])

        self.assertEqual(store.all_committed_ids(), [])

    def test_close_observed_after_empty_poll_gets_a_final_drain(self):
        final_ref = _ref("last")
        self.source = _CloseAfterFirstEmptyPollChannel(
            self.src_path, self.producer, final_ref
        )
        dist = self._distributor(dp_size=1)

        self.assertTrue(dist.pump())
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["last"])
        self.assertFalse(dist.finished)

        self.assertFalse(dist.pump())
        self.assertTrue(dist.finished)

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

    def test_resume_never_settles_released_duplicates_twice(self):
        # At-least-once delivery can leave TWO records of one sample on the
        # channel file. The prior attempt settled both (ack for the fresh one,
        # immediate settle for the duplicate) into the consumed sidecar. A
        # resume that re-polls the file must not settle either record again:
        # consumed would advance past the producer's published accounting and
        # crash its byte reconciliation.
        ledger_path = os.path.join(self.dir, "resume-dup.sqlite")
        store = SQLiteMetadataStore(ledger_path)
        original = DataFlowController("run0", metadata_store=store)
        ref = _ref("s0")
        self.producer.publish(ref)
        self.producer.publish(ref)  # at-least-once duplicate record
        original.commit_samples("old-distributor", [ref])
        original.ack_train_refs(
            "trainer", ["s0"], global_step=1, optimizer_durable=True
        )
        # The prior attempt settled the ack AND its polled duplicate.
        self.source.mark_consumed(2)

        restarted = DataFlowController("run0", metadata_store=store)
        report = restarted.reconcile_on_restart(self.feature_store)
        self.assertEqual(report["released"], ["s0"])
        dist = self._distributor(
            dp_size=1,
            controller=restarted,
            skip_ids=report["released"],
            requeued_ids=report["requeued"],
        )
        _pump_until_quiet(dist)

        self.assertEqual(dist.stats["skipped"], 2)
        # published == consumed: nothing over-advanced, in-flight is exactly 0.
        self.assertEqual(self.producer.consumed_remote(), 2)
        self.assertEqual(self.producer.in_flight_remote(), 0)
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), [])
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
        self.assertIsNone(dist.error)
        self.assertTrue(dist.finished)
        self.assertEqual(dist.stats["dropped"], 1)
        # the dropped ref's lease is released, not leaked
        self.assertEqual(controller.sample_queue.in_flight(), 2)  # dispatched only
        self.assertEqual(controller.sample_queue.depth(), 0)
        self.assertEqual([item[0] for item in self.feature_store.aborted], ["s2"])
        self.assertEqual(self.producer.consumed_remote(), 1)
        for rank in range(2):
            reader = InboxChannel(RefDistributor.inbox_path(self.inbox_dir, rank))
            self.assertTrue(reader.is_closed())

    def test_dispatches_only_complete_optimizer_step_windows(self):
        dist = self._distributor(dp_size=2, refs_per_rank_step=4)
        for i in range(8):
            self.producer.publish(_ref(f"s{i}"))
        _pump_until_quiet(dist)
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s0", "s2", "s4", "s6"])
        self.assertEqual(_inbox_ids(self.inbox_dir, 1), ["s1", "s3", "s5", "s7"])

    def test_rounds_withheld_until_full_optimizer_window_is_secured(self):
        # Dispatching the first round of a window obligates every rank to a
        # full accumulation window, and acks only advance in whole windows —
        # so a round is released only once its whole window is committed.
        dist = self._distributor(
            dp_size=2,
            refs_per_rank_step=4,
            refs_per_rank_batch=1,
        )
        for i in range(2):  # one round of the 8-ref global window
            self.producer.publish(_ref(f"s{i}"))

        _pump_until_quiet(dist)

        self.assertEqual(_inbox_ids(self.inbox_dir, 0), [])
        self.assertEqual(_inbox_ids(self.inbox_dir, 1), [])
        self.assertEqual(dist.stats["dispatched"], 0)
        self.assertEqual(dist._window_dispatched, 0)

        for i in range(2, 8):  # complete the window
            self.producer.publish(_ref(f"s{i}"))
        _pump_until_quiet(dist)

        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s0", "s2", "s4", "s6"])
        self.assertEqual(_inbox_ids(self.inbox_dir, 1), ["s1", "s3", "s5", "s7"])
        self.assertEqual(dist.stats["dispatched"], 8)
        self.assertEqual(dist._window_dispatched, 0)

    def test_streamed_partial_optimizer_window_settles_cleanly_at_eof(self):
        # End-of-stream below one optimizer window terminates the run cleanly
        # (drop_last-style): a raise here would be unrecoverable, because the
        # leftover-below-one-window count is invariant across resume attempts.
        dist = self._distributor(
            dp_size=2,
            refs_per_rank_step=4,
            refs_per_rank_batch=1,
        )
        for i in range(2):
            self.producer.publish(_ref(f"s{i}"))
        self.producer.close()

        dist._run_guarded()

        self.assertIsNone(dist.error)
        self.assertTrue(dist.finished)
        self.assertEqual(dist.stats["dispatched"], 0)
        self.assertEqual(dist.stats["dropped"], 2)
        self.assertEqual(self.feature_store.adopted, ["s0", "s1"])
        self.assertEqual(
            [sample_id for sample_id, _ in self.feature_store.aborted],
            ["s0", "s1"],
        )
        self.assertEqual(self.producer.consumed_remote(), 2)
        for rank in range(2):
            reader = InboxChannel(RefDistributor.inbox_path(self.inbox_dir, rank))
            self.assertTrue(reader.is_closed())
            self.assertEqual(reader.poll(), [])

    def test_exact_window_multiple_streams_rounds_and_closes_without_drops(self):
        dist = self._distributor(
            dp_size=2,
            refs_per_rank_step=4,
            refs_per_rank_batch=1,
        )
        for i in range(16):  # exactly two global windows
            self.producer.publish(_ref(f"s{i}"))
        self.producer.close()

        dist._run_guarded()

        self.assertIsNone(dist.error)
        self.assertTrue(dist.finished)
        self.assertEqual(dist.stats["dispatched"], 16)
        self.assertEqual(dist.stats["dropped"], 0)
        self.assertEqual(len(_inbox_ids(self.inbox_dir, 0)), 8)
        self.assertEqual(len(_inbox_ids(self.inbox_dir, 1)), 8)

    def test_requeued_leftover_below_one_window_settles_on_resume(self):
        # Resume scenario: reconciliation requeued committed-unacked refs from
        # a prior attempt, the producer has nothing further, and the leftover
        # cannot fill one optimizer window. The attempt must terminate cleanly
        # instead of failing identically on every retry.
        controller = DataFlowController("run0")
        leftover = [_ref(f"s{i}") for i in range(3)]
        controller.sample_queue.put(leftover)  # as reconcile_on_restart does
        dist = self._distributor(
            dp_size=2,
            refs_per_rank_step=2,  # global window = 4 > 3 leftover
            controller=controller,
        )
        self.producer.close()

        dist._run_guarded()

        self.assertIsNone(dist.error)
        self.assertTrue(dist.finished)
        self.assertEqual(dist.stats["dropped"], 3)
        self.assertEqual(controller.sample_queue.depth(), 0)
        self.assertEqual(controller.sample_queue.in_flight(), 0)
        self.assertEqual(self.producer.consumed_remote(), 3)
        for rank in range(2):
            reader = InboxChannel(RefDistributor.inbox_path(self.inbox_dir, rank))
            self.assertTrue(reader.is_closed())

    def test_partial_optimizer_quantum_settles_tail_after_aligned_prefix(self):
        dist = self._distributor(dp_size=2, refs_per_rank_step=4)
        for i in range(10):
            self.producer.publish(_ref(f"s{i}"))
        self.producer.close()

        dist._run_guarded()

        self.assertIsNone(dist.error)
        self.assertTrue(dist.finished)
        self.assertEqual(_inbox_ids(self.inbox_dir, 0), ["s0", "s2", "s4", "s6"])
        self.assertEqual(_inbox_ids(self.inbox_dir, 1), ["s1", "s3", "s5", "s7"])
        self.assertEqual(dist.stats["dropped"], 2)
        self.assertEqual(self.feature_store.adopted, ["s8", "s9"])
        self.assertEqual(
            [sample_id for sample_id, _ in self.feature_store.aborted],
            ["s8", "s9"],
        )
        self.assertEqual(self.producer.consumed_remote(), 2)

    def test_partial_window_cleanup_failure_stays_loud(self):
        self.feature_store = _FailingAbortStore()
        dist = self._distributor(dp_size=2)
        self.producer.publish(_ref("s0"))
        self.producer.close()

        dist._run_guarded()

        self.assertIsInstance(dist.error, RuntimeError)
        self.assertIn("cleanup errors", str(dist.error))
        self.assertEqual(self.feature_store.adopted, ["s0"])
        for rank in range(2):
            reader = InboxChannel(RefDistributor.inbox_path(self.inbox_dir, rank))
            with self.assertRaisesRegex(RuntimeError, "injected abort failure"):
                reader.is_closed()

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
