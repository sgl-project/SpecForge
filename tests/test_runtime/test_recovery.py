# coding=utf-8
"""Durable online recovery: replay unacked refs and skip durable ones."""

import os
import json
import tempfile
import unittest

from specforge.runtime.contracts import FeatureSpec, SampleRef
from specforge.runtime.control_plane.controller import DataFlowController
from specforge.runtime.control_plane.metadata_store import SQLiteMetadataStore
from specforge.runtime.data_plane.ref_serialization import ref_from_dict, ref_to_dict


def _ref(sample_id: str) -> SampleRef:
    return SampleRef(
        sample_id=sample_id,
        run_id="run0",
        source_task_id=f"task-{sample_id}",
        feature_store_uri=f"mooncake://run0/{sample_id}",
        feature_keys={"hidden_state": f"{sample_id}/hidden_state"},
        feature_specs={
            "hidden_state": FeatureSpec(
                name="hidden_state", shape=(1, 8, 4), dtype="float32"
            )
        },
        strategy="eagle3",
        num_tokens=8,
    )


class _RecordingFeatureStore:
    def __init__(self) -> None:
        self.aborted = []

    def abort(self, sample_id, *, reason="aborted") -> None:
        self.aborted.append((sample_id, reason))


class TestDurableRecovery(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="recovery_")
        self.path = os.path.join(self.work, "metadata.sqlite")

    def test_sample_ref_round_trip_preserves_feature_specs(self):
        ref = _ref("s0")
        restored = ref_from_dict(json.loads(json.dumps(ref_to_dict(ref))))
        self.assertEqual(restored, ref)
        self.assertEqual(restored.feature_specs["hidden_state"].shape, (1, 8, 4))

    def test_crash_before_ack_replays_every_committed_sample(self):
        store = SQLiteMetadataStore(self.path)
        before = DataFlowController("run0", metadata_store=store)
        before.commit_samples("producer", [_ref("s0"), _ref("s1")])
        before.sample_queue.get(2)
        store.close()

        reopened = SQLiteMetadataStore(self.path)
        restarted = DataFlowController("run0", metadata_store=reopened)
        report = restarted.reconcile_on_restart(_RecordingFeatureStore())
        self.assertEqual(report["released"], [])
        self.assertEqual(set(report["requeued"]), {"s0", "s1"})
        replay = restarted.sample_queue.get(2)
        self.assertEqual({ref.sample_id for ref in replay}, {"s0", "s1"})
        reopened.close()

    def test_crash_after_durable_ack_skips_and_releases_only_acked_prefix(self):
        store = SQLiteMetadataStore(self.path)
        before = DataFlowController("run0", metadata_store=store)
        before.commit_samples("producer", [_ref("s0"), _ref("s1"), _ref("s2")])
        before.sample_queue.get(3)
        # Simulate death after SQLite commit and before transient queue/counter
        # acknowledgement.
        store.record_train_ack(["s0", "s1"], global_step=1, optimizer_durable=True)
        store.close()

        reopened = SQLiteMetadataStore(self.path)
        restarted = DataFlowController("run0", metadata_store=reopened)
        features = _RecordingFeatureStore()
        report = restarted.reconcile_on_restart(features)
        self.assertEqual(set(report["released"]), {"s0", "s1"})
        self.assertEqual(report["requeued"], ["s2"])
        self.assertEqual({item[0] for item in features.aborted}, {"s0", "s1"})
        replay = restarted.sample_queue.get(3)
        self.assertEqual([ref.sample_id for ref in replay], ["s2"])
        reopened.close()


if __name__ == "__main__":
    unittest.main()
