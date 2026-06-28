# coding=utf-8
"""Durable recovery (B4): SQLite round-trip + restart reconciliation.

The headline test is ``test_controller_dies_between_ack_and_release`` — the 3am
crash window where a controller dies after the durable ack but before the queue
release is recorded. A correct restart must neither duplicate-train the acked
sample nor lose any committed-unacked sample. No torch needed.
"""

import os
import tempfile
import unittest

from specforge.runtime.contracts import FeatureSpec, SampleRef
from specforge.runtime.control_plane.controller import DataFlowController
from specforge.runtime.control_plane.metadata_store import (
    SQLiteMetadataStore,
    sample_ref_from_json,
    sample_ref_to_json,
)
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue


def _ref(sid: str) -> SampleRef:
    return SampleRef(
        sample_id=sid,
        run_id="run",
        source_task_id=f"task-{sid}",
        feature_store_uri=f"mem://st/{sid}",
        feature_keys={"hidden_state": f"{sid}/hidden_state"},
        feature_specs={
            "hidden_state": FeatureSpec(
                name="hidden_state",
                shape=(1, 8, 4),
                dtype="float32",
                target_repr="hidden_state",
                target_meta={"vocab_map_version": "v1"},
            )
        },
        strategy="eagle3",
        num_tokens=8,
        estimated_bytes=128,
    )


class _RecordingFeatureStore:
    """Tracks abort() calls so reconciliation's idempotent free is observable."""

    def __init__(self):
        self.aborted = []

    def abort(self, sample_id, *, reason="aborted", **kw):
        self.aborted.append(sample_id)


class TestSampleRefSerialization(unittest.TestCase):
    def test_round_trip_preserves_nested_specs(self):
        ref = _ref("s0")
        back = sample_ref_from_json(sample_ref_to_json(ref))
        self.assertEqual(back, ref)  # frozen dataclass equality, incl. nested spec
        self.assertEqual(back.feature_specs["hidden_state"].shape, (1, 8, 4))  # tuple
        self.assertEqual(
            back.feature_specs["hidden_state"].target_meta, {"vocab_map_version": "v1"}
        )


class TestSQLiteMetadataStore(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "meta.db")

    def test_commit_is_idempotent_and_durable(self):
        store = SQLiteMetadataStore(self.path)
        self.assertTrue(store.commit_sample(_ref("s0")))
        self.assertFalse(store.commit_sample(_ref("s0")))  # duplicate
        self.assertEqual(store.committed_count(), 1)
        store.close()
        # reopen: state survives
        store2 = SQLiteMetadataStore(self.path)
        self.assertTrue(store2.is_committed("s0"))
        self.assertEqual(store2.get_committed("s0").sample_id, "s0")
        store2.close()

    def test_ack_marker_is_one_durable_fact(self):
        store = SQLiteMetadataStore(self.path)
        store.commit_sample(_ref("s0"))
        store.record_train_ack(["s0"], global_step=7, optimizer_durable=True)
        store.close()
        store2 = SQLiteMetadataStore(self.path)
        m = store2.durable_marker()
        self.assertIn("s0", m["acked"])
        self.assertEqual(m["global_step"], 7)
        self.assertTrue(m["optimizer_durable"])
        store2.close()


class TestRestartReconciliation(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "meta.db")

    def test_controller_dies_between_ack_and_release(self):
        # --- pre-crash controller ---
        store = SQLiteMetadataStore(self.path)
        ctrl = DataFlowController(
            "run", sample_queue=SampleRefQueue(), metadata_store=store
        )
        ctrl.commit_samples("w0", [_ref(f"s{i}") for i in range(4)])
        leased = ctrl.lease_train_refs("trainer", 4)
        self.assertEqual(len(leased), 4)
        # The trainer finished an optimizer step covering s0,s1 and the durable
        # ack landed -> then the process CRASHES before queue.ack/release. We
        # simulate that by recording the ack directly on the durable store and
        # NOT completing the in-process release.
        store.record_train_ack(["s0", "s1"], global_step=1, optimizer_durable=True)
        store.close()  # process gone; in-process queue state lost

        # --- restart: fresh controller + queue, same durable DB ---
        store2 = SQLiteMetadataStore(self.path)
        ctrl2 = DataFlowController(
            "run", sample_queue=SampleRefQueue(), metadata_store=store2
        )
        fs = _RecordingFeatureStore()
        report = ctrl2.reconcile_on_restart(feature_store=fs)

        # no duplicate train: acked+stepped samples are NOT requeued
        self.assertNotIn("s0", report["requeued"])
        self.assertNotIn("s1", report["requeued"])
        self.assertEqual(set(report["released"]), {"s0", "s1"})
        self.assertEqual(set(fs.aborted), {"s0", "s1"})  # idempotent free issued
        # no data loss: every committed-unacked sample is replayable
        self.assertEqual(set(report["requeued"]), {"s2", "s3"})
        replayed = ctrl2.lease_train_refs("trainer", 8)
        self.assertEqual({r.sample_id for r in replayed}, {"s2", "s3"})
        store2.close()

    def test_crash_before_any_ack_replays_everything(self):
        store = SQLiteMetadataStore(self.path)
        ctrl = DataFlowController(
            "run", sample_queue=SampleRefQueue(), metadata_store=store
        )
        ctrl.commit_samples("w0", [_ref(f"s{i}") for i in range(3)])
        ctrl.lease_train_refs("trainer", 3)
        store.close()  # crash with no durable ack

        store2 = SQLiteMetadataStore(self.path)
        ctrl2 = DataFlowController(
            "run", sample_queue=SampleRefQueue(), metadata_store=store2
        )
        report = ctrl2.reconcile_on_restart()
        self.assertEqual(set(report["requeued"]), {"s0", "s1", "s2"})  # nothing lost
        self.assertEqual(report["released"], [])
        store2.close()

    def test_durable_ack_without_global_step_still_releases(self):
        # A durable ack (optimizer_durable=True) with global_step omitted must
        # still release the acked samples on restart — not replay them.
        store = SQLiteMetadataStore(self.path)
        ctrl = DataFlowController(
            "run", sample_queue=SampleRefQueue(), metadata_store=store
        )
        ctrl.commit_samples("w0", [_ref(f"s{i}") for i in range(2)])
        ctrl.lease_train_refs("trainer", 2)
        store.record_train_ack(["s0", "s1"], global_step=None, optimizer_durable=True)
        store.close()

        store2 = SQLiteMetadataStore(self.path)
        ctrl2 = DataFlowController(
            "run", sample_queue=SampleRefQueue(), metadata_store=store2
        )
        report = ctrl2.reconcile_on_restart()
        self.assertEqual(
            set(report["released"]), {"s0", "s1"}
        )  # released, not replayed
        self.assertEqual(report["requeued"], [])
        store2.close()

    def test_reconcile_is_idempotent(self):
        # Reconciling twice on a fresh (post-crash) queue must not double-enqueue;
        # queue.put is idempotent on sample_id.
        store = SQLiteMetadataStore(self.path)
        pre = DataFlowController(
            "run", sample_queue=SampleRefQueue(), metadata_store=store
        )
        pre.commit_samples("w0", [_ref(f"s{i}") for i in range(3)])
        store.record_train_ack(["s0"], global_step=1, optimizer_durable=True)
        store.close()

        # restart with a fresh, empty queue (the crash dropped the old one)
        store2 = SQLiteMetadataStore(self.path)
        ctrl = DataFlowController(
            "run", sample_queue=SampleRefQueue(), metadata_store=store2
        )
        ctrl.reconcile_on_restart()
        ctrl.reconcile_on_restart()  # second time must be a no-op for the queue
        out = ctrl.lease_train_refs("trainer", 8)
        self.assertEqual({r.sample_id for r in out}, {"s1", "s2"})  # no s0, no dups
        store2.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
