from __future__ import annotations

import unittest
from unittest import mock

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.providers import OfflineCaptureLayout
from specforge.offline_capture import OfflineSGLangCapture


class OfflineCaptureLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = builtin_algorithm_registry()

    def test_builtin_offline_layouts_materialize_exact_storage_schemas(self):
        expected_sources = {
            "eagle3": {
                "input_ids": "input_ids",
                "loss_mask": "loss_mask",
                "aux_hidden_state": "aux_hidden_states",
                "hidden_state": "last_hidden_states",
            },
            "dflash": {
                "input_ids": "input_ids",
                "loss_mask": "loss_mask",
                "hidden_states": "aux_hidden_states",
            },
            "domino": {
                "input_ids": "input_ids",
                "loss_mask": "loss_mask",
                "hidden_states": "aux_hidden_states",
            },
            "dspark": {
                "input_ids": "input_ids",
                "loss_mask": "loss_mask",
                "hidden_states": "aux_hidden_states",
                "target_last_hidden_states": "last_hidden_states",
            },
        }
        expected_capture_methods = {
            "eagle3": "eagle3",
            "dflash": "dflash",
            "domino": "dflash",
            "dspark": "dspark",
        }
        sources = {
            "input_ids": torch.tensor([1, 2, 3]),
            "loss_mask": torch.tensor([1, 1, 0]),
            "aux_hidden_states": torch.randn(1, 3, 5 * 8),
            "last_hidden_states": torch.randn(1, 3, 8),
        }

        for strategy, feature_sources in expected_sources.items():
            with self.subTest(strategy=strategy):
                registration = self.registry.resolve(strategy)
                provider = registration.providers.offline_for("text")
                record = provider.capture_layout.materialize(sources)

                self.assertEqual(
                    expected_capture_methods[strategy],
                    provider.capture_layout.capture_method,
                )

                self.assertEqual(set(feature_sources), set(record))
                self.assertEqual(
                    registration.spec.feature_contract(
                        "offline", "text"
                    ).storage.required_tensors,
                    set(record),
                )
                for feature_name, source_name in feature_sources.items():
                    self.assertIs(sources[source_name], record[feature_name])

                ready = provider.build_normalizer(3)(record)
                self.assertTrue(
                    registration.spec.feature_contract(
                        "offline", "text"
                    ).required_tensors.issubset(ready)
                )

    def test_materialize_preserves_arbitrary_auxiliary_layer_counts(self):
        for strategy in ("dflash", "domino", "dspark"):
            layout = (
                self.registry.resolve(strategy)
                .providers.offline_for("text")
                .capture_layout
            )
            for layer_count in (1, 4, 7):
                aux_hidden_states = torch.randn(1, 3, layer_count * 8)
                sources = {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "loss_mask": torch.ones(3, dtype=torch.long),
                    "aux_hidden_states": aux_hidden_states,
                    "last_hidden_states": torch.randn(1, 3, 8),
                }
                with self.subTest(strategy=strategy, layer_count=layer_count):
                    record = layout.materialize(sources)
                    self.assertIs(aux_hidden_states, record["hidden_states"])
                    self.assertEqual(
                        layer_count * 8,
                        record["hidden_states"].shape[-1],
                    )

    def test_dflash_family_normalizers_require_adjacent_supervision(self):
        raw = {
            "input_ids": torch.tensor([1, 2, 3]),
            "loss_mask": torch.tensor([1, 0, 1]),
            "hidden_states": torch.randn(1, 3, 8),
            "target_last_hidden_states": torch.randn(1, 3, 8),
        }
        for strategy in ("dflash", "domino", "dspark"):
            with self.subTest(strategy=strategy):
                normalizer = (
                    self.registry.resolve(strategy)
                    .providers.offline_for("text")
                    .build_normalizer(3)
                )
                with self.assertRaisesRegex(ValueError, "two consecutive"):
                    normalizer(raw)

    def test_duplicate_output_names_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate.*hidden_states"):
            OfflineCaptureLayout(
                capture_method="dflash",
                aux_feature="hidden_states",
                last_hidden_feature=None,
                passthrough=(("hidden_states", "input_ids"),),
            )

    def test_missing_mapped_source_reports_source_and_output_names(self):
        layout = OfflineCaptureLayout(
            capture_method="dflash",
            aux_feature="hidden_states",
            last_hidden_feature="target_last_hidden_states",
            passthrough=(
                ("input_ids", "input_ids"),
                ("loss_mask", "loss_mask"),
            ),
        )
        sources = {
            "input_ids": torch.tensor([1]),
            "loss_mask": torch.tensor([1]),
            "aux_hidden_states": torch.zeros(1, 1, 1),
        }

        with self.assertRaises(KeyError) as raised:
            layout.materialize(sources)

        message = str(raised.exception)
        self.assertIn("last_hidden_states", message)
        self.assertIn("target_last_hidden_states", message)

    def test_local_capture_forwards_the_algorithm_capture_method(self):
        backend = mock.Mock()
        capture = OfflineSGLangCapture(backend)

        for capture_method in ("dflash", "dspark", "hspec"):
            with self.subTest(capture_method=capture_method):
                backend.reset_mock()
                capture.set_capture_layers(
                    [1, 9, 17, 25, 33],
                    capture_method=capture_method,
                )

                self.assertEqual(capture_method, capture.capture_method)
                backend.set_capture_layers.assert_called_once_with(
                    [1, 9, 17, 25, 33],
                    capture_method=capture_method,
                )

    def test_hspec_kv_layers_are_only_set_when_given_explicitly(self):
        backend = mock.Mock()
        capture = OfflineSGLangCapture(backend)

        capture.set_capture_layers(
            [1, 9, 17, 25, 33], capture_method="hspec"
        )
        backend.set_hspec_kv_layer_ids.assert_not_called()

        capture.set_hspec_kv_layer_ids([17])
        backend.set_hspec_kv_layer_ids.assert_called_once_with([17])

    def test_hspec_capture_stacks_per_sample_features(self):
        backend = mock.Mock()
        seq, width = 3, 8
        features = []
        for _ in range(2):
            features.append(
                {
                    "input_ids": torch.arange(seq).unsqueeze(0),
                    "loss_mask": torch.ones(1, seq, dtype=torch.long),
                    "prefix_masks": torch.ones(1, seq, dtype=torch.long),
                    "hidden_states": torch.randn(seq, width),
                    "target_last_hidden_states": torch.randn(seq, width),
                    "selected_target_k": torch.randn(seq, width),
                    "selected_target_v": torch.randn(seq, width),
                }
            )
        backend.capture_hspec.return_value = features
        capture = OfflineSGLangCapture(backend, capture_method="hspec")

        batch = capture.capture(
            input_ids=torch.zeros(2, seq, dtype=torch.long),
            attention_mask=torch.ones(2, seq, dtype=torch.long),
            loss_mask=torch.ones(2, seq, dtype=torch.long),
        )

        self.assertEqual(batch.hidden_states.shape, (2, seq, width))
        self.assertEqual(batch.last_hidden_states.shape, (2, seq, width))
        self.assertEqual(batch.selected_target_k.shape, (2, seq, width))
        self.assertEqual(batch.selected_target_v.shape, (2, seq, width))
        self.assertEqual(batch.prefix_masks.shape, (2, 1, seq))
        rows = list(batch.feature_rows())
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["hidden_states"].shape, (seq, width))
            self.assertEqual(row["selected_target_k"].shape, (seq, width))
            self.assertEqual(row["prefix_masks"].shape, (1, seq))


if __name__ == "__main__":
    unittest.main()
