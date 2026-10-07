"""Packing preserves logical samples while removing only batch padding (CPU)."""

import tempfile
import unittest
from pathlib import Path

import torch

from specforge.algorithms.eagle3.data import (
    DataCollatorWithPacking,
    build_offline_normalizer,
    build_packed_collator,
)
from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue


def _feature(length, offset=0):
    return {
        "input_ids": (torch.arange(length) + offset).unsqueeze(0),
        "attention_mask": torch.ones(1, length, dtype=torch.long),
        "loss_mask": (torch.arange(length) % 2).unsqueeze(0),
        "hidden_state": torch.arange(length * 6).reshape(1, length, 6) + offset,
        "target": torch.arange(length * 2).reshape(1, length, 2) + offset,
    }


class SequencePackingCollatorTest(unittest.TestCase):
    def test_concatenates_features_with_document_positions_and_padded_denominator(self):
        features = [_feature(2, 10), _feature(5, 20), _feature(1, 30)]
        originals = [{key: value.clone() for key, value in f.items()} for f in features]

        batch = build_packed_collator()(features)

        for key in features[0]:
            with self.subTest(key=key):
                torch.testing.assert_close(
                    batch[key], torch.cat([f[key] for f in originals], dim=1)
                )
        self.assertEqual(batch["input_ids"].shape, (1, 8))
        self.assertEqual(batch["position_ids"].tolist(), [[0, 1, 0, 1, 2, 3, 4, 0]])
        self.assertEqual(batch["sequence_lengths"].tolist(), [2, 5, 1])
        self.assertEqual(batch["loss_denominator"].item(), 3 * 5)
        self.assertEqual(batch["sequence_lengths"].dtype, torch.long)
        self.assertEqual(batch["sequence_lengths"].device.type, "cpu")
        self.assertEqual(batch["loss_denominator"].device.type, "cpu")
        for feature, original in zip(features, originals):
            self.assertEqual(feature.keys(), original.keys())
            for key in original:
                torch.testing.assert_close(feature[key], original[key])

    def test_accepts_standard_positions_and_single_sample(self):
        feature = _feature(3)
        feature["position_ids"] = torch.arange(3).unsqueeze(0)

        batch = DataCollatorWithPacking()([feature])

        self.assertEqual(batch["position_ids"].tolist(), [[0, 1, 2]])
        self.assertEqual(batch["sequence_lengths"].tolist(), [3])
        self.assertEqual(batch["loss_denominator"].item(), 3)

    def test_rejects_empty_batch_and_empty_or_batched_sample(self):
        with self.assertRaisesRegex(ValueError, "empty feature batch"):
            DataCollatorWithPacking()([])
        for ids in (torch.empty(1, 0), torch.zeros(2, 3), torch.zeros(3)):
            feature = _feature(3)
            feature["input_ids"] = ids
            with (
                self.subTest(shape=ids.shape),
                self.assertRaisesRegex(ValueError, "nonempty"),
            ):
                DataCollatorWithPacking()([feature])

    def test_rejects_missing_or_misaligned_features(self):
        for key in _feature(3):
            feature = _feature(3)
            del feature[key]
            with self.subTest(missing=key), self.assertRaisesRegex(KeyError, key):
                DataCollatorWithPacking()([feature])
        for key in ("loss_mask", "attention_mask", "hidden_state", "target"):
            for wrong in (torch.zeros(1, 2), torch.zeros(3), torch.zeros(2, 3, 6)):
                feature = _feature(3)
                feature[key] = wrong
                with (
                    self.subTest(key=key, shape=wrong.shape),
                    self.assertRaisesRegex(ValueError, key),
                ):
                    DataCollatorWithPacking()([feature])

    def test_rejects_padding_and_nonstandard_positions(self):
        feature = _feature(3)
        feature["attention_mask"][0, -1] = 0
        with self.assertRaisesRegex(ValueError, "unpadded"):
            DataCollatorWithPacking()([feature])
        for positions in (
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[0, 0, 1]]),
            torch.arange(3),
            torch.arange(3).reshape(1, 1, 3),
        ):
            feature = _feature(3)
            feature["position_ids"] = positions
            with (
                self.subTest(shape=positions.shape),
                self.assertRaisesRegex(ValueError, "standard text position_ids"),
            ):
                DataCollatorWithPacking()([feature])

    def test_offline_loader_preserves_sample_ids_order_and_partial_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            for index, length in enumerate((2, 5, 3)):
                torch.save(
                    {
                        "input_ids": torch.arange(length) + index * 10,
                        "loss_mask": torch.ones(length, dtype=torch.long),
                        "hidden_state": torch.full((1, length, 2), float(index)),
                        "aux_hidden_state": torch.full((1, length, 6), float(index)),
                    },
                    Path(directory) / f"{index:03d}.ckpt",
                )
            refs = OfflineManifestReader(directory, run_id="packing-test").read()
            loader = FeatureDataLoader(
                LocalFeatureStore("packing-test"),
                refs=refs,
                batch_size=2,
                collate_fn=build_packed_collator(),
                per_sample_transform=build_offline_normalizer(4),
                drop_last=False,
            )
            batches = list(loader)
            repeated = list(loader)

        expected_ids = [[r.sample_id for r in refs[:2]], [refs[2].sample_id]]
        self.assertEqual([b.sample_ids for b in batches], expected_ids)
        self.assertEqual([b.sample_ids for b in repeated], expected_ids)
        self.assertEqual(batches[0].tensors["sequence_lengths"].tolist(), [2, 4])
        self.assertEqual(
            batches[0].tensors["input_ids"].tolist(), [[0, 1, 10, 11, 12, 13]]
        )
        self.assertEqual(batches[0].tensors["loss_mask"].tolist(), [[1, 0, 1, 1, 1, 0]])
        self.assertEqual(batches[0].tensors["loss_denominator"].item(), 8)
        self.assertEqual(batches[1].tensors["sequence_lengths"].tolist(), [3])
        self.assertEqual(batches[1].tensors["loss_denominator"].item(), 3)


class StreamingPackingDataTest(unittest.TestCase):
    @staticmethod
    def _dflash(length, offset=0, teacher=True):
        sample = _feature(length, offset)
        result = {
            "input_ids": sample["input_ids"],
            "loss_mask": sample["loss_mask"],
            "hidden_states": sample["hidden_state"],
        }
        if teacher:
            result["target_last_hidden_states"] = sample["target"]
        return result

    def test_dflash_preserves_optional_teacher_features_without_loss_denominator(self):
        from specforge.algorithms.common.hidden_states_data import build_packed_collator

        for teacher in (False, True):
            features = [self._dflash(3, 10, teacher), self._dflash(5, 20, teacher)]
            originals = [{k: v.clone() for k, v in f.items()} for f in features]
            batch = build_packed_collator()(features)
            self.assertEqual(batch["sequence_lengths"].tolist(), [3, 5])
            self.assertNotIn("loss_denominator", batch)
            self.assertEqual("target_last_hidden_states" in batch, teacher)
            for key in originals[0]:
                torch.testing.assert_close(
                    batch[key], torch.cat([f[key] for f in originals], dim=1)
                )
                for actual, original in zip(features, originals):
                    torch.testing.assert_close(actual[key], original[key])

    def test_dflash_rejects_inconsistent_teacher_features_and_bad_shapes(self):
        from specforge.algorithms.common.hidden_states_data import build_packed_collator

        collate = build_packed_collator()
        with self.assertRaises(KeyError):
            collate([self._dflash(3, teacher=True), self._dflash(3, teacher=False)])
        for key in self._dflash(3):
            sample = self._dflash(3)
            sample[key] = sample[key][:, :2]
            with self.subTest(key=key), self.assertRaises(ValueError):
                collate([sample])

    def test_online_queue_keeps_logical_batch_size_identity_and_ack_count(self):
        from specforge.algorithms.builtin import builtin_algorithm_registry

        for name in ("eagle3", "dflash"):
            algorithm = builtin_algorithm_registry().resolve(name)
            store = LocalFeatureStore(f"packing-queue-{name}")
            refs = []
            for index, length in enumerate((3, 7, 4, 6)):
                sample = (
                    _feature(length, index * 10)
                    if name == "eagle3"
                    else self._dflash(length, index * 10)
                )
                refs.append(
                    store.put(
                        sample,
                        sample_id=f"sample-{index}",
                        metadata={
                            "run_id": "queue-test",
                            "strategy": name,
                            "target_repr": "hidden_state",
                        },
                    )
                )
            queue = SampleRefQueue()
            queue.put(refs)
            loader = FeatureDataLoader(
                store,
                queue,
                batch_size=2,
                strategy=name,
                collate_fn=algorithm.providers.server_streaming_for(
                    "text"
                ).build_packed_collator(),
            )
            batches = list(loader)
            self.assertEqual(
                [batch.sample_ids for batch in batches],
                [["sample-0", "sample-1"], ["sample-2", "sample-3"]],
            )
            self.assertEqual(
                [batch.tensors["sequence_lengths"].tolist() for batch in batches],
                [[3, 7], [4, 6]],
            )
            self.assertEqual(queue.in_flight(), 0)
            self.assertEqual(queue.depth(), 0)


if __name__ == "__main__":
    unittest.main()
