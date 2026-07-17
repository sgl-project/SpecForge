# coding=utf-8
"""O1.1: shared, durable, cross-process control plane for online disaggregation.

The online producer and consumer run in *separate processes*. Before O1.1 each
built its own in-process ``InMemoryMetadataStore``, so commit / dedup / ack were
not shared and a restart could not reconcile: the producer held the committed
set, the consumer held the ack marker, in two different process-local stores.
These CPU tests prove a shared ``SQLiteMetadataStore`` makes commit/dedup/ack
cross-process, and that the streaming consumer skips already-trained samples on
restart instead of re-reading the append-only channel from offset zero.

No GPU / torch needed: this exercises the control plane + streaming queue
directly (the FSDP training half is the GPU launcher test).
"""

import os
import tempfile
import unittest

from specforge.runtime.contracts import SampleRef
from specforge.runtime.control_plane.controller import DataFlowController
from specforge.runtime.control_plane.metadata_store import SQLiteMetadataStore
from specforge.runtime.data_plane.streaming_ref_channel import (
    StreamingRefChannel,
    StreamingRefQueue,
)


def _ref(i: int, run_id: str = "run0") -> SampleRef:
    """A minimal tensor-free SampleRef (the control plane carries metadata only)."""
    sid = f"{run_id}:s{i}"
    return SampleRef(
        sample_id=sid,
        run_id=run_id,
        source_task_id=f"task-{i}",
        feature_store_uri=f"mem://{run_id}",
        feature_keys={"hidden_state": f"{sid}/hidden_state"},
        feature_specs={},
        strategy="eagle3",
        metadata={"target_repr": "logits", "num_tokens": 4},
    )


class _RecordingStore:
    """Minimal FeatureStore stub: records abort() (the reconcile-time free)."""

    def __init__(self) -> None:
        self.aborted = []

    def abort(self, sample_id: str, reason: str = "") -> None:
        self.aborted.append(sample_id)


def _db() -> str:
    return os.path.join(tempfile.mkdtemp(prefix="o11_"), "meta.db")


class TestSharedControlPlane(unittest.TestCase):
    def test_commit_and_dedup_cross_process(self):
        db = _db()
        # two controllers over ONE db file == two processes sharing the store
        producer = DataFlowController("run0", metadata_store=SQLiteMetadataStore(db))
        consumer = DataFlowController("run0", metadata_store=SQLiteMetadataStore(db))

        refs = [_ref(i) for i in range(4)]
        producer.commit_samples("rollout-0", refs)

        # consumer (separate store instance, same file) sees the producer's commits
        for r in refs:
            self.assertTrue(consumer.store.is_committed(r.sample_id))
            got = consumer.store.get_committed(r.sample_id)
            self.assertIsNotNone(got)
            self.assertEqual(got.sample_id, r.sample_id)
        self.assertEqual(consumer.store.committed_count(), 4)

        # at-least-once dedup: re-committing the same refs adds no duplicate rows
        producer.commit_samples("rollout-0", refs)
        self.assertEqual(consumer.store.committed_count(), 4)

    def test_durable_ack_visible_across_process(self):
        db = _db()
        producer = DataFlowController("run0", metadata_store=SQLiteMetadataStore(db))
        consumer = DataFlowController("run0", metadata_store=SQLiteMetadataStore(db))
        refs = [_ref(i) for i in range(3)]
        producer.commit_samples("rollout-0", refs)

        tid = consumer.register_trainer({"role": "trainer"})
        consumer.ack_train_refs(
            tid,
            [refs[0].sample_id, refs[1].sample_id],
            global_step=5,
            optimizer_durable=True,
        )

        # a fresh controller reopened on the same file sees the durable marker
        reopened = DataFlowController("run0", metadata_store=SQLiteMetadataStore(db))
        marker = reopened.store.durable_marker()
        self.assertEqual(marker["acked"], {refs[0].sample_id, refs[1].sample_id})
        self.assertEqual(marker["global_step"], 5)
        self.assertTrue(marker["optimizer_durable"])

    def test_reconcile_on_restart_releases_acked_requeues_unacked(self):
        db = _db()
        producer = DataFlowController("run0", metadata_store=SQLiteMetadataStore(db))
        consumer = DataFlowController("run0", metadata_store=SQLiteMetadataStore(db))
        refs = [_ref(i) for i in range(5)]
        producer.commit_samples("rollout-0", refs)
        tid = consumer.register_trainer({"role": "trainer"})
        acked = [refs[0].sample_id, refs[1].sample_id, refs[2].sample_id]
        consumer.ack_train_refs(tid, acked, global_step=3, optimizer_durable=True)

        # crash + restart: a fresh controller over the same db reconciles
        restarted = DataFlowController("run0", metadata_store=SQLiteMetadataStore(db))
        fs = _RecordingStore()
        result = restarted.reconcile_on_restart(fs)

        self.assertEqual(set(result["released"]), set(acked))
        self.assertEqual(
            set(result["requeued"]), {refs[3].sample_id, refs[4].sample_id}
        )
        # released samples were idempotently freed from the feature store
        self.assertEqual(set(fs.aborted), set(acked))
        # the unacked tail is requeued for re-training (no data loss)
        self.assertEqual(restarted.sample_queue.depth(), 2)

    def test_reconcile_without_optimizer_durable_requeues_all(self):
        # acked but the optimizer step never durably committed -> replay all (no
        # sample counted as trained), per the B4 release rule.
        db = _db()
        producer = DataFlowController("run0", metadata_store=SQLiteMetadataStore(db))
        consumer = DataFlowController("run0", metadata_store=SQLiteMetadataStore(db))
        refs = [_ref(i) for i in range(3)]
        producer.commit_samples("rollout-0", refs)
        tid = consumer.register_trainer({"role": "trainer"})
        consumer.ack_train_refs(
            tid,
            [r.sample_id for r in refs],
            global_step=None,
            optimizer_durable=False,
        )

        restarted = DataFlowController("run0", metadata_store=SQLiteMetadataStore(db))
        result = restarted.reconcile_on_restart(_RecordingStore())
        self.assertEqual(result["released"], [])
        self.assertEqual(set(result["requeued"]), {r.sample_id for r in refs})


class TestStreamingQueueRestartSkip(unittest.TestCase):
    def _chan_path(self) -> str:
        return os.path.join(tempfile.mkdtemp(prefix="o11_chan_"), "refs.jsonl")

    def test_skip_ids_drops_trained_refs_on_reread(self):
        path = self._chan_path()
        # producer published s0..s4 then closed (the durable channel survives)
        wchan = StreamingRefChannel(path)
        refs = [_ref(i) for i in range(5)]
        wchan.publish_many(refs)
        wchan.close()

        # restart: s0,s1,s2 were durably trained -> skip them on re-read
        trained = {refs[0].sample_id, refs[1].sample_id, refs[2].sample_id}
        rchan = StreamingRefChannel(path)
        queue = StreamingRefQueue(rchan, poll_s=0.0, skip_ids=trained)

        got = queue.get(2)  # only the untrained tail survives the skip filter
        self.assertEqual(
            [r.sample_id for r in got], [refs[3].sample_id, refs[4].sample_id]
        )
        queue.ack(got)

        # backpressure stays exact: 3 skipped + 2 acked == 5 consumed, 0 in flight
        self.assertEqual(wchan.consumed_remote(), 5)
        self.assertEqual(wchan.in_flight_remote(), 0)

    def test_skip_all_terminates_on_close(self):
        # every published ref was already trained: the queue must drain to empty
        # (loop end) once the channel is closed, not spin.
        path = self._chan_path()
        wchan = StreamingRefChannel(path)
        refs = [_ref(i) for i in range(3)]
        wchan.publish_many(refs)
        wchan.close()

        queue = StreamingRefQueue(
            StreamingRefChannel(path),
            poll_s=0.0,
            skip_ids={r.sample_id for r in refs},
        )
        self.assertEqual(queue.get(1), [])
        self.assertEqual(wchan.consumed_remote(), 3)  # all counted consumed

    def test_no_skip_ids_is_unchanged(self):
        # regression: skip_ids=None preserves the pre-O1.1 full-stream behavior
        path = self._chan_path()
        wchan = StreamingRefChannel(path)
        refs = [_ref(i) for i in range(3)]
        wchan.publish_many(refs)
        wchan.close()

        queue = StreamingRefQueue(StreamingRefChannel(path), poll_s=0.0)
        got = []
        while True:
            batch = queue.get(1)
            if not batch:
                break
            got.extend(batch)
            queue.ack(batch)
        self.assertEqual([r.sample_id for r in got], [r.sample_id for r in refs])
        self.assertEqual(wchan.consumed_remote(), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
