"""Capture/data contracts are derived without selecting model constructors."""

import unittest

import torch

from tests.test_algorithms.fixtures import builtin_algorithm_registry


class CaptureContractTest(unittest.TestCase):
    def test_all_capture_families_preserve_unshifted_features(self):
        registry = builtin_algorithm_registry()
        for registration in registry:
            with self.subTest(algorithm=registration.name):
                offline = registration.providers.offline_for("text")
                sources = dict(
                    input_ids=torch.arange(5),
                    loss_mask=torch.ones(5),
                    aux_hidden_states=torch.randn(1, 5, 24),
                    last_hidden_states=torch.randn(1, 5, 8),
                )
                record = offline.capture_layout.materialize(sources)
                normalized = offline.build_normalizer(4)(record)
                batch = offline.build_collator()([normalized, normalized])
                self.assertEqual(batch["input_ids"].tolist(), [[0, 1, 2, 3]] * 2)
                self.assertEqual(batch["loss_mask"].tolist(), [[1, 1, 1, 1]] * 2)
                self.assertEqual(
                    set(record),
                    set(
                        registration.spec.feature_contract(
                            "offline", "text"
                        ).storage.required_tensors
                    ),
                )
                server = registration.providers.server_streaming_for("text")
                self.assertEqual(
                    server.capture_method, offline.capture_layout.capture_method
                )
                self.assertFalse(hasattr(registration.providers.model, "build_draft"))
                self.assertFalse(
                    hasattr(registration.providers.model, "build_training_model")
                )

    def test_mismatched_feature_lengths_fail_before_training(self):
        provider = (
            builtin_algorithm_registry().resolve("dspark").providers.offline_for("text")
        )
        record = dict(
            input_ids=torch.arange(3),
            loss_mask=torch.ones(3),
            hidden_states=torch.zeros(1, 3, 8),
            target_last_hidden_states=torch.zeros(1, 2, 8),
        )
        with self.assertRaisesRegex(ValueError, "identical sequence lengths"):
            provider.build_normalizer(8)(record)
