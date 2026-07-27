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
                    "eagle3" if strategy == "eagle3" else "dflash",
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

        capture.set_capture_layers(
            [1, 9, 17, 25, 33],
            capture_method="dflash",
        )

        self.assertEqual("dflash", capture.capture_method)
        backend.set_capture_layers.assert_called_once_with(
            [1, 9, 17, 25, 33],
            capture_method="dflash",
        )


if __name__ == "__main__":
    unittest.main()
