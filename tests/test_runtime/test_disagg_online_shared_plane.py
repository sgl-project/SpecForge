# coding=utf-8
"""Shared-plane consumer resume through the canonical disaggregated builder."""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from specforge.runtime.contracts import FeatureSpec, SampleRef
from specforge.runtime.control_plane.controller import DataFlowController
from specforge.runtime.control_plane.metadata_store import SQLiteMetadataStore
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from specforge.training.checkpoint import STATE_FILE


def _ref(sample_id: str) -> SampleRef:
    return SampleRef(
        sample_id=sample_id,
        run_id="run0",
        source_task_id=f"task-{sample_id}",
        feature_store_uri=f"mooncake://run0/{sample_id}",
        feature_keys={"hidden_state": f"{sample_id}/hidden_state"},
        feature_specs={
            "hidden_state": FeatureSpec(
                name="hidden_state", shape=(2, 4), dtype="float32"
            )
        },
        strategy="eagle3",
        metadata={"target_repr": "hidden_state"},
    )


class _RetainingFeatureStore:
    retain_on_release = True

    def __init__(self) -> None:
        self.aborted = []

    def abort(self, sample_id, *, reason="aborted") -> None:
        self.aborted.append((sample_id, reason))


class _FakeDistributor:
    latest = None

    def __init__(self, *_args, **kwargs) -> None:
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        type(self).latest = self

    @staticmethod
    def inbox_path(inbox_dir: str, dp_rank: int) -> str:
        return os.path.join(inbox_dir, f"inbox-rank{dp_rank}.jsonl")

    def start(self):
        self.started = True
        return self

    def stop(self) -> None:
        self.stopped = True


class TestSharedPlaneConsumerResume(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="shared_plane_resume_")
        self.db = os.path.join(self.work, "metadata.sqlite")
        self.channel = StreamingRefChannel(os.path.join(self.work, "refs.jsonl"))
        self.features = _RetainingFeatureStore()

    def _checkpoint(self, step: int) -> str:
        path = os.path.join(self.work, f"run0-step{step}")
        os.makedirs(path, exist_ok=True)
        torch.save({"global_step": step}, os.path.join(path, STATE_FILE))
        return path

    def _seed_ledger(self, *, acked=(), step=None) -> None:
        store = SQLiteMetadataStore(self.db)
        controller = DataFlowController("run0", metadata_store=store)
        controller.commit_samples("producer", [_ref("s0"), _ref("s1")])
        if acked:
            store.record_train_ack(
                list(acked), global_step=step, optimizer_durable=True
            )
        store.close()

    def _build(self, *, resume_from=None):
        from specforge.launch import build_disagg_online_consumer

        captured = {}

        def assemble(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace()

        if resume_from is not None and self.channel.consumer_quantum() is None:
            self.channel.publish_consumer_quantum(1)
        with (
            mock.patch("torch.distributed.is_initialized", return_value=False),
            mock.patch(
                "specforge.runtime.data_plane.ref_distributor.RefDistributor",
                _FakeDistributor,
            ),
            mock.patch("specforge.launch._assemble_trainer", side_effect=assemble),
        ):
            trainer = build_disagg_online_consumer(
                feature_store=self.features,
                channel=self.channel,
                draft_model=object(),
                optimizer_factory=object(),
                run_id="run0",
                output_dir=os.path.join(self.work, "output"),
                collate_fn=lambda features: {},
                metadata_db_path=self.db,
                inbox_dir=os.path.join(self.work, "inboxes"),
                resume_from=resume_from,
            )
        return trainer, captured, _FakeDistributor.latest

    def test_crash_before_ack_requeues_every_committed_ref(self):
        self._seed_ledger()
        checkpoint = self._checkpoint(0)

        trainer, captured, distributor = self._build(resume_from=checkpoint)

        self.assertTrue(trainer.ref_distributor.started)
        self.assertEqual(distributor.kwargs["skip_ids"], set())
        self.assertEqual(distributor.kwargs["requeued_ids"], {"s0", "s1"})
        self.assertEqual(self.features.aborted, [])
        self.assertEqual(captured["resume_from"], checkpoint)
        self.assertTrue(captured["ref_source"]["prepositioned"])
        self.assertTrue(captured["ref_source"]["defer_ack_until_durable"])

    def test_crash_after_ack_skips_durable_ref_and_requeues_tail(self):
        self._seed_ledger(acked=("s0",), step=1)
        checkpoint = self._checkpoint(1)

        _trainer, _captured, distributor = self._build(resume_from=checkpoint)

        self.assertEqual(distributor.kwargs["skip_ids"], {"s0"})
        self.assertEqual(distributor.kwargs["requeued_ids"], {"s1"})
        self.assertEqual([item[0] for item in self.features.aborted], ["s0"])

    def test_marker_ahead_of_checkpoint_is_rejected(self):
        self._seed_ledger(acked=("s0",), step=2)
        checkpoint = self._checkpoint(1)

        with self.assertRaisesRegex(RuntimeError, "ahead of.*checkpoint"):
            self._build(resume_from=checkpoint)

    def test_fresh_attempt_rejects_nonempty_ledger(self):
        self._seed_ledger()
        with self.assertRaisesRegex(ValueError, "fresh online attempt cannot reuse"):
            self._build()

    def test_consumer_requires_release_retention(self):
        self.features.retain_on_release = False
        with self.assertRaisesRegex(ValueError, "retain_on_release=True"):
            self._build()


if __name__ == "__main__":
    unittest.main()
