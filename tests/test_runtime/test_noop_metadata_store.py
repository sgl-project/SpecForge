# coding=utf-8
"""NoOpMetadataStore contract for colocated paths."""

import unittest

from specforge.runtime.control_plane import (
    DataFlowController,
    NoOpMetadataStore,
)
from specforge.runtime.control_plane.metadata_store import MetadataStore


class TestNoOpMetadataStore(unittest.TestCase):
    def test_contract_retains_nothing(self):
        s = NoOpMetadataStore()
        self.assertIsInstance(s, MetadataStore)
        # commit reports "new" so the caller enqueues once, but nothing is kept.
        self.assertTrue(s.commit_sample(object()))
        self.assertFalse(s.is_committed("x"))
        self.assertIsNone(s.get_committed("x"))
        self.assertEqual(s.committed_count(), 0)

    def test_durable_marker_is_empty(self):
        s = NoOpMetadataStore()
        s.record_train_ack(["a", "b"], global_step=5, optimizer_durable=True)
        self.assertEqual(
            s.durable_marker(),
            {"acked": set(), "global_step": None, "optimizer_durable": False},
        )
        self.assertEqual(s.committed_count(), 0)


class TestNoOpController(unittest.TestCase):
    def test_noop_ack_path_is_safe(self):
        controller = DataFlowController(
            "run",
            metadata_store=NoOpMetadataStore(),
            enable_sample_queue=False,
        )
        controller.ack_train_refs("t", ["s0"], global_step=1, optimizer_durable=True)
        self.assertEqual(controller.store.committed_count(), 0)
        self.assertEqual(controller.status()["samples_committed"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
