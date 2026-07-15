# coding=utf-8
"""CPU-only gates for bounded colocated-online rollout."""

import types
import unittest

from specforge.runtime.contracts import SampleRef
from specforge.runtime.control_plane import DataFlowController
from specforge.runtime.data_plane import LocalRolloutStream


class _FakeFeatureStore:
    def __init__(self):
        self.resident = {}

    def put(self, *, sample_id, source_task_id):
        self.resident[sample_id] = object()
        return SampleRef(
            sample_id=sample_id,
            run_id="run",
            source_task_id=source_task_id,
            feature_store_uri=f"fake://run/{sample_id}",
            feature_keys={"feature": f"{sample_id}/feature"},
            feature_specs={},
            strategy="eagle3",
        )

    def release(self, sample_id):
        self.resident.pop(sample_id, None)

    def abort(self, sample_id, *, reason):
        self.resident.pop(sample_id, None)

    def abort_all(self, *, reason):
        del reason
        evicted = len(self.resident)
        self.resident.clear()
        return evicted

    def health(self):
        return {"resident_samples": len(self.resident)}


class _FakeRolloutWorker:
    def __init__(self, controller, store, *, error=None, error_after_commit=False):
        self.controller = controller
        self.store = store
        self.error = error
        self.error_after_commit = error_after_commit
        self.worker_id = controller.register_rollout_worker(
            {"worker_id": "fake-rollout", "role": "rollout"}
        )
        self.started = False
        self.stop_reason = None
        self.calls = 0

    def start(self):
        self.started = True

    def stop(self, reason="stopped"):
        self.stop_reason = reason

    def run_once(self, max_tasks):
        self.calls += 1
        if self.error is not None and not self.error_after_commit:
            raise self.error
        tasks = self.controller.lease_prompt_tasks(self.worker_id, max_tasks)
        refs = []
        for task in tasks:
            ref = self.store.put(
                sample_id=f"run:{task.task_id}",
                source_task_id=task.task_id,
            )
            refs.append(ref)
        self.controller.commit_samples(self.worker_id, refs)
        if self.error is not None:
            raise self.error
        return refs


def _runtime(num_prompts, *, batch_size=2, error=None, error_after_commit=False):
    controller = DataFlowController("run")
    controller.ingest_prompts(
        [
            {"task_id": f"task-{index}", "payload": {"input_ids": [index]}}
            for index in range(num_prompts)
        ]
    )
    store = _FakeFeatureStore()
    worker = _FakeRolloutWorker(
        controller,
        store,
        error=error,
        error_after_commit=error_after_commit,
    )
    stream = LocalRolloutStream(
        controller=controller,
        workers=[worker],
        feature_store=store,
        max_resident_samples=batch_size,
    )
    return controller, store, worker, stream


class TestLocalRolloutStream(unittest.TestCase):
    def test_rollout_is_interleaved_and_residency_is_batch_bounded(self):
        controller, store, worker, stream = _runtime(10, batch_size=2)
        consumed = []

        with stream:
            while True:
                refs = stream.get(2, timeout_s=0.0)
                if not refs:
                    break
                self.assertLessEqual(store.health()["resident_samples"], 2)
                for ref in refs:
                    store.release(ref.sample_id)
                    consumed.append(ref.sample_id)
                stream.ack(refs)

        self.assertEqual(len(consumed), 10)
        self.assertEqual(stream.produced_count, 10)
        self.assertEqual(stream.peak_resident_samples, 2)
        self.assertEqual(store.health()["resident_samples"], 0)
        self.assertEqual(controller.sample_queue.depth(), 0)
        self.assertEqual(controller.sample_queue.in_flight(), 0)
        self.assertTrue(worker.started)
        self.assertEqual(worker.stop_reason, "training_finished")

    def test_trainer_early_stop_stops_rollout_and_evicts_unconsumed_batch(self):
        controller, store, worker, stream = _runtime(10, batch_size=2)
        loader = types.SimpleNamespace(queue=stream)

        class _EarlyStopTrainer:
            def fit(self, data):
                self.refs = data.queue.get(2)
                return 1

        trainer = _EarlyStopTrainer()
        with stream:
            step = trainer.fit(loader)

        self.assertEqual(step, 1)
        self.assertEqual(len(trainer.refs), 2)
        self.assertEqual(stream.produced_count, 2)
        self.assertEqual(controller.status()["prompts_pending"], 8)
        self.assertEqual(store.health()["resident_samples"], 0)
        self.assertEqual(worker.stop_reason, "training_finished")

    def test_producer_exception_propagates_and_stops_worker(self):
        error = RuntimeError("capture exploded")
        _, store, worker, stream = _runtime(3, batch_size=1, error=error)

        with self.assertRaises(RuntimeError) as raised:
            with stream:
                stream.get(1)

        self.assertIs(raised.exception, error)
        self.assertEqual(worker.stop_reason, "training_failed")
        self.assertEqual(store.health()["resident_samples"], 0)

    def test_partial_commit_before_producer_error_is_reclaimed(self):
        error = RuntimeError("second capture failed")
        _, store, worker, stream = _runtime(
            2,
            batch_size=2,
            error=error,
            error_after_commit=True,
        )

        with self.assertRaises(RuntimeError) as raised:
            with stream:
                stream.get(2)

        self.assertIs(raised.exception, error)
        self.assertEqual(worker.stop_reason, "training_failed")
        self.assertEqual(store.health()["resident_samples"], 0)

    def test_partial_tail_is_bounded_and_close_reclaims_it(self):
        _, store, _, stream = _runtime(3, batch_size=2)

        with stream:
            first = stream.get(2)
            for ref in first:
                store.release(ref.sample_id)
            stream.ack(first)
            tail = stream.get(2)
            self.assertEqual(len(tail), 1)
            self.assertEqual(store.health()["resident_samples"], 1)

        self.assertEqual(store.health()["resident_samples"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
