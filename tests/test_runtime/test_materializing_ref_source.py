# coding=utf-8
"""Local feature generation -> persisted ref adapter tests."""

import tempfile
import unittest

import torch

from specforge.inference.adapters.materializing import (
    MaterializationFailure,
    MaterializingRefSource,
)
from specforge.inference.capture import CaptureConfig
from specforge.runtime.contracts import PromptTask, SampleRef
from specforge.runtime.data_plane.disaggregated import SharedDirFeatureStore


def _capture():
    return CaptureConfig.from_strategy(
        required_features={"input_ids", "hidden_states", "loss_mask"},
        aux_hidden_state_layer_ids=(2, 18, 33),
        target_repr="hidden_state",
        target_hidden_size=8,
    )


def _task(index: int) -> PromptTask:
    return PromptTask(
        task_id=f"task-{index}",
        run_id="run",
        source_id="fixture",
        payload={"input_ids": [index, index + 1], "loss_mask": [1, 1]},
        metadata={"num_tokens": 2},
        max_length=8,
    )


class _FeatureSource:
    def __init__(self, *, missing_loss_mask: int = -1) -> None:
        self.missing_loss_mask = missing_loss_mask
        self.generated = []

    def generate_features(self, tasks, *, capture):
        del capture
        self.generated = []
        for index, _task_value in enumerate(tasks):
            features = {
                "input_ids": torch.tensor([[index, index + 1]]),
                "hidden_states": torch.full((1, 2, 24), float(index)),
                "loss_mask": torch.ones(1, 2, dtype=torch.long),
            }
            if index == self.missing_loss_mask:
                features.pop("loss_mask")
            self.generated.append(features)
        return self.generated


class _FailingPutStore:
    def __init__(self) -> None:
        self.aborted = []

    def put(self, tensors, *, sample_id, metadata):
        del tensors, sample_id, metadata
        raise OSError("store full")

    def abort(self, sample_id, *, reason):
        self.aborted.append((sample_id, reason))


class _ShortFeatureSource:
    def generate_features(self, tasks, *, capture):
        del tasks, capture
        return []


class TestMaterializingRefSource(unittest.TestCase):
    def test_persists_ordered_refs_with_stable_identity_and_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            store = SharedDirFeatureStore(root, store_id="features")
            generated = _FeatureSource()
            source = MaterializingRefSource(
                generated,
                store,
                run_id="run",
                strategy="dflash",
                target_model_version="target-v1",
                tokenizer_version="tokenizer-v1",
            )

            results = source.produce_refs([_task(0), _task(1)], capture=_capture())

            self.assertTrue(all(isinstance(result, SampleRef) for result in results))
            self.assertEqual(
                [result.sample_id for result in results],
                ["run:task-0", "run:task-1"],
            )
            self.assertEqual(
                [result.source_task_id for result in results],
                ["task-0", "task-1"],
            )
            self.assertEqual(results[0].strategy, "dflash")
            self.assertEqual(results[0].target_model_version, "target-v1")
            self.assertEqual(results[0].tokenizer_version, "tokenizer-v1")
            self.assertEqual(results[0].metadata["generation"], 1)
            tensors, handle = store.get(results[1])
            self.assertEqual(tensors["hidden_states"].sum().item(), 48.0)
            store.release(handle)
            self.assertIn("loss_mask", generated.generated[0])

    def test_capture_mismatch_is_non_retryable_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as root:
            store = SharedDirFeatureStore(root, store_id="features")
            source = MaterializingRefSource(
                _FeatureSource(missing_loss_mask=1),
                store,
                run_id="run",
                strategy="dflash",
            )

            first, second = source.produce_refs(
                [_task(0), _task(1)], capture=_capture()
            )

            self.assertIsInstance(first, SampleRef)
            self.assertIsInstance(second, MaterializationFailure)
            self.assertEqual(second.task_id, "task-1")
            self.assertFalse(second.retryable)
            self.assertEqual(store.health()["resident_samples"], 1)

    def test_put_failure_is_aligned_retryable_and_aborted(self):
        store = _FailingPutStore()
        source = MaterializingRefSource(
            _FeatureSource(), store, run_id="run", strategy="dflash"
        )

        (result,) = source.produce_refs([_task(0)], capture=_capture())

        self.assertEqual(
            result,
            MaterializationFailure(
                task_id="task-0", reason="put_failed:store full", retryable=True
            ),
        )
        self.assertEqual(store.aborted, [("run:task-0", "put_failed:store full")])

    def test_wrong_feature_count_fails_every_task_without_retry(self):
        store = _FailingPutStore()
        source = MaterializingRefSource(
            _ShortFeatureSource(), store, run_id="run", strategy="dflash"
        )

        results = source.produce_refs([_task(0), _task(1)], capture=_capture())

        self.assertEqual(
            [result.task_id for result in results], ["task-0", "task-1"]
        )
        self.assertTrue(all(not result.retryable for result in results))
        self.assertTrue(
            all(
                "returned 0 feature records for 2 tasks" in result.reason
                for result in results
            )
        )
        self.assertEqual(store.aborted, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
