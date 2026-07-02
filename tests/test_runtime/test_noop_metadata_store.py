# coding=utf-8
"""NoOpMetadataStore contract and DeploymentMode control-plane assembly."""

import os
import tempfile
import unittest

from specforge.runtime.control_plane import (
    InMemoryMetadataStore,
    NoOpMetadataStore,
    build_control_plane_for_mode,
)
from specforge.runtime.control_plane.metadata_store import (
    MetadataStore,
    SQLiteMetadataStore,
)


class TestNoOpMetadataStore(unittest.TestCase):
    def test_contract_retains_nothing(self):
        s = NoOpMetadataStore()
        self.assertIsInstance(s, MetadataStore)
        # commit reports "new" so the caller enqueues once, but nothing is kept.
        self.assertTrue(s.commit_sample(object()))
        self.assertFalse(s.is_committed("x"))
        self.assertIsNone(s.get_committed("x"))
        self.assertEqual(s.committed_count(), 0)
        self.assertEqual(s.all_committed_ids(), [])

    def test_durable_marker_is_empty(self):
        s = NoOpMetadataStore()
        s.record_train_ack(["a", "b"], global_step=5, optimizer_durable=True)
        self.assertEqual(
            s.durable_marker(),
            {"acked": set(), "global_step": None, "optimizer_durable": False},
        )
        self.assertEqual(s.committed_count(), 0)


class TestBuildControlPlaneForMode(unittest.TestCase):
    def test_local_colocated_is_lightweight(self):
        controller, durable_ack = build_control_plane_for_mode("local_colocated", "run")
        self.assertFalse(durable_ack)
        self.assertIsInstance(controller.store, NoOpMetadataStore)

    def test_other_modes_keep_durable_store(self):
        for mode in ("dataflow_colocated", "disaggregated"):
            controller, durable_ack = build_control_plane_for_mode(mode, "run")
            self.assertTrue(durable_ack, mode)
            self.assertIsInstance(controller.store, InMemoryMetadataStore)

    def test_metadata_db_path_selects_sqlite(self):
        db = os.path.join(tempfile.mkdtemp(prefix="rcp_"), "meta.sqlite")
        controller, durable_ack = build_control_plane_for_mode(
            "disaggregated", "run", metadata_db_path=db
        )
        try:
            self.assertTrue(durable_ack)
            self.assertIsInstance(controller.store, SQLiteMetadataStore)
        finally:
            controller.store.close()
        # local_colocated deliberately ignores the path: it retains nothing.
        controller, durable_ack = build_control_plane_for_mode(
            "local_colocated", "run", metadata_db_path=db
        )
        self.assertFalse(durable_ack)
        self.assertIsInstance(controller.store, NoOpMetadataStore)

    def test_noop_ack_path_is_safe(self):
        # ack_train_refs reconstructs refs via get_committed (None under NoOp) and
        # must neither raise nor retain state — the colocated invariant.
        controller, _ = build_control_plane_for_mode("local_colocated", "run")
        controller.ack_train_refs("t", ["s0"], global_step=1, optimizer_durable=True)
        self.assertEqual(controller.store.committed_count(), 0)
        self.assertEqual(controller.status()["samples_committed"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
