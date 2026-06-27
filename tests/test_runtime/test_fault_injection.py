# coding=utf-8
"""Fault-injection: rollout + data-plane failure paths and cleanup (M5).

Covers the scenarios in ``testing/fault_injection.md`` that Phase 1 must exercise
(rollout/data-plane failures + idempotent cleanup). Trainer- and controller-death
recovery is covered by ``test_recovery.py`` (durable reconciliation). Expected
behavior asserted here, per that doc:

* retryable failures replay within budget; terminal failures are visible with
  reason + sample_id;
* no tensors cross the control plane, even on a failure path;
* feature-store cleanup is idempotent;
* a half-failed ``put`` leaves the store unchanged (atomic).
"""

import os
import tempfile
import unittest

import torch

from specforge.runtime.contracts import FeatureSpec, SampleRef, assert_no_tensors
from specforge.runtime.control_plane.controller import DataFlowController
from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue


def _committed_ref(store: LocalFeatureStore, sid: str) -> SampleRef:
    return store.put(
        {"hidden_state": torch.randn(1, 4, 8)},
        sample_id=sid,
        metadata={"target_repr": "hidden_state", "strategy": "eagle3"},
    )


class TestRolloutFailures(unittest.TestCase):
    def test_rollout_dies_before_write_requeues_prompt(self):
        # No feature write happened -> the prompt attempt is marked failed and,
        # if retryable, requeued with attempt+1 (replay within budget).
        ctrl = DataFlowController("run")
        ctrl.ingest_prompts([{"task_id": "t0", "payload": {}}])
        [task] = ctrl.lease_prompt_tasks("w0", 1)
        self.assertEqual(task.attempt, 0)
        ctrl.fail_prompt_tasks("w0", ["t0"], reason="worker_died", retryable=True)
        [retry] = ctrl.lease_prompt_tasks("w0", 1)
        self.assertEqual(retry.attempt, 1)  # replayed, attempt incremented

    def test_rollout_terminal_failure_is_visible(self):
        ctrl = DataFlowController("run")
        ctrl.ingest_prompts([{"task_id": "t0", "payload": {}}])
        ctrl.lease_prompt_tasks("w0", 1)
        ctrl.fail_prompt_tasks("w0", ["t0"], reason="bad_prompt", retryable=False)
        self.assertEqual(ctrl.status()["prompts_failed"], 1)
        self.assertEqual(ctrl.lease_prompt_tasks("w0", 1), [])  # not requeued

    def test_rollout_dies_after_write_before_commit_is_idempotent(self):
        # Feature written, process dies before commit ack. On retry rollout
        # re-puts (same sample_id, store bumps generation) and commits; a
        # duplicate commit is a no-op. No double-enqueue, no orphan leak.
        store = LocalFeatureStore("st")
        ctrl = DataFlowController("run")
        ref = _committed_ref(store, "s0")  # write succeeded
        # ... crash before commit_samples. Retry: re-put + commit (idempotent).
        ref_retry = _committed_ref(store, "s0")
        ctrl.commit_samples("w0", [ref_retry])
        ctrl.commit_samples("w0", [ref_retry])  # duplicate ack -> no-op
        self.assertEqual(ctrl.status()["samples_committed"], 1)
        self.assertEqual(ctrl.status()["queue_depth"], 1)
        self.assertEqual(store.health()["resident_samples"], 1)  # no orphan
        _ = ref


class TestFeatureStoreFailures(unittest.TestCase):
    def test_put_over_budget_is_atomic(self):
        # A put that trips the budget raises and leaves residency unchanged.
        store = LocalFeatureStore("st", max_resident_bytes=64)
        store.put({"x": torch.zeros(1, 8)}, sample_id="s0", metadata={})  # 32 bytes
        before = store.health()["resident_bytes"]
        with self.assertRaises(MemoryError):
            store.put(
                {"x": torch.zeros(1, 16)}, sample_id="s1", metadata={}
            )  # +64 > 64
        self.assertEqual(store.health()["resident_bytes"], before)  # no partial write
        self.assertEqual(store.health()["resident_samples"], 1)

    def test_get_missing_key_fails_sample_terminally(self):
        # Feature evicted out from under a committed ref -> loader get raises,
        # the queue fails the ref non-retryably with a reason, and it is dropped.
        store = LocalFeatureStore("st")
        queue = SampleRefQueue()
        ref = _committed_ref(store, "s0")
        queue.put([ref])
        store.abort("s0", reason="evicted")  # tensor gone, ref still queued
        loader = FeatureDataLoader(store, queue, batch_size=1, drop_last=False)
        with self.assertRaises(KeyError):
            list(loader)  # materialize fails
        self.assertEqual(queue.depth(), 0)  # terminal -> dropped, not requeued
        self.assertEqual(queue.in_flight(), 0)

    def test_release_and_abort_cleanup_is_idempotent(self):
        store = LocalFeatureStore("st")
        ref = _committed_ref(store, "s0")
        _, h = store.get(ref)
        store.release(h)
        store.release(h)  # idempotent
        store.abort("s0", reason="late")  # already freed -> no-op, no raise
        store.abort("s0", reason="late")
        self.assertEqual(store.health()["resident_samples"], 0)


class TestOfflineManifestFaults(unittest.TestCase):
    def test_bad_dtype_non_tensor_value_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            torch.save(
                {
                    "input_ids": torch.arange(4),
                    "loss_mask": torch.ones(4),
                    "hidden_state": "not-a-tensor",  # corrupt
                    "aux_hidden_state": torch.randn(1, 4, 8),
                },
                os.path.join(d, "bad.ckpt"),
            )
            with self.assertRaises(TypeError):
                OfflineManifestReader(d, run_id="off").read()

    def test_mixed_schema_version_batch_rejected(self):
        store = LocalFeatureStore("st")
        spec = {"x": FeatureSpec(name="x", shape=(1, 4), dtype="float32")}

        def mk(sid, ver):
            return SampleRef(
                sample_id=sid,
                run_id="r",
                source_task_id=None,
                feature_store_uri=f"file://{sid}",
                feature_keys={"x": "x"},
                feature_specs=spec,
                strategy="eagle3",
                schema_version=ver,
                metadata={"target_repr": "hidden_state"},
            )

        refs = [mk("s0", 1), mk("s1", 2)]  # schema versions disagree
        loader = FeatureDataLoader(store, refs=refs, batch_size=2, drop_last=False)
        with self.assertRaises(ValueError):
            list(loader)


class TestControlPlaneStaysTensorFree(unittest.TestCase):
    def test_commit_rejects_tensor_in_metadata(self):
        ctrl = DataFlowController("run")
        bad = SampleRef(
            sample_id="s0",
            run_id="r",
            source_task_id=None,
            feature_store_uri="mem://st/s0",
            feature_keys={},
            feature_specs={},
            strategy="eagle3",
            metadata={"smuggled": torch.randn(2)},  # tensor in control-plane record
        )
        with self.assertRaises(TypeError):
            ctrl.commit_samples("w0", [bad])

    def test_assert_no_tensors_on_failure_record(self):
        # A failure record routed through the control plane is metadata only.
        record = {"reason": "evicted", "sample_id": "s0", "component": "loader"}
        assert_no_tensors(record)  # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
