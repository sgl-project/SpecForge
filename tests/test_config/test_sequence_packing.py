"""Reject unsupported packing combinations before building any model."""

import unittest
from dataclasses import replace

from pydantic import ValidationError

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.eagle3.data import DataCollatorWithPacking
from specforge.application import resolve_run
from specforge.config import Config
from specforge.config.schema import TrainingConfig


def _offline_config(**training):
    return Config.model_validate(
        {
            "model": {
                "target_model_path": "target",
                "draft_model_config": "draft.json",
                "vocab_mapping_path": "mapping.pt",
            },
            "data": {"hidden_states_path": "features"},
            "training": training,
        }
    )


class SequencePackingConfigTest(unittest.TestCase):
    def test_default_retains_padded_execution(self):
        resolved = resolve_run(_offline_config())
        self.assertFalse(resolved.config.training.sequence_packing)

    def test_offline_eagle3_packing_is_available(self):
        resolved = resolve_run(_offline_config(sequence_packing=True, batch_size=4))
        self.assertTrue(resolved.config.training.sequence_packing)
        provider = resolved.algorithm.providers.offline_for("text")
        self.assertIsInstance(provider.build_packed_collator(), DataCollatorWithPacking)

    def test_other_algorithms_reject_packing(self):
        for strategy in ("domino", "dspark"):
            with (
                self.subTest(strategy=strategy),
                self.assertRaisesRegex(
                    ValueError, "does not support training.sequence_packing"
                ),
            ):
                resolve_run(_offline_config(strategy=strategy, sequence_packing=True))

    def test_rejects_non_flex_attention(self):
        for attention_backend in ("eager", "sdpa", "fa", "usp"):
            with (
                self.subTest(backend=attention_backend),
                self.assertRaisesRegex(
                    ValidationError, "sequence_packing requires flex_attention"
                ),
            ):
                TrainingConfig(
                    sequence_packing=True, attention_backend=attention_backend
                )

    def test_rejects_unimplemented_objective_combinations(self):
        for extra in (
            {"compact_teacher": True},
            {"trim_loss_positions": True},
        ):
            with (
                self.subTest(extra=extra),
                self.assertRaisesRegex(
                    ValidationError, "sequence_packing currently requires"
                ),
            ):
                TrainingConfig(sequence_packing=True, **extra)

    def test_lk_packing_is_algorithm_specific(self):
        for lk_loss_type in ("lambda", "alpha", "tv"):
            with self.subTest(lk_loss_type=lk_loss_type):
                with self.assertRaisesRegex(ValueError, "LK|lk_loss"):
                    resolve_run(
                        _offline_config(
                            sequence_packing=True, lk_loss_type=lk_loss_type
                        )
                    )
                resolve_run(
                    _offline_config(
                        strategy="dflash",
                        sequence_packing=True,
                        lk_loss_type=lk_loss_type,
                    )
                )

    def test_supports_dflash_offline_and_dflash2_architecture(self):
        resolved = resolve_run(
            _offline_config(strategy="dflash", sequence_packing=True)
        )
        self.assertTrue(
            callable(
                resolved.algorithm.providers.offline_for("text").build_packed_collator
            )
        )
        self.assertIn(
            "DFlash2DraftModel", resolved.algorithm.spec.draft.compatible_architectures
        )

    def test_supports_online_text_features(self):
        payload = _offline_config(sequence_packing=True).model_dump()
        payload["data"] = {"train_data_path": "train.jsonl"}
        payload["training"]["max_steps"] = 1
        payload["training"]["role"] = "auto"
        payload["deployment"] = {
            "mode": "disaggregated",
            "disaggregated": {
                "control_dir": "outputs/packing-test/control",
                "backend": "mooncake",
                "server_urls": ["http://127.0.0.1:30000"],
                "mooncake_metadata_server": "http://127.0.0.1:35880/metadata",
                "mooncake_master_server_addr": "127.0.0.1:35551",
            },
        }
        for strategy in ("eagle3", "dflash"):
            payload["training"]["strategy"] = strategy
            with self.subTest(strategy=strategy):
                resolved = resolve_run(Config.model_validate(payload))
                provider = resolved.algorithm.providers.server_streaming_for("text")
                self.assertTrue(callable(provider.build_packed_collator))

    def test_provider_rejects_noncallable_packing_factory(self):
        provider = (
            builtin_algorithm_registry().resolve("eagle3").providers.offline_for("text")
        )
        with self.assertRaisesRegex(
            TypeError, "build_packed_collator must be callable"
        ):
            replace(provider, build_packed_collator=True)

        streaming = (
            builtin_algorithm_registry()
            .resolve("eagle3")
            .providers.server_streaming_for("text")
        )
        with self.assertRaisesRegex(
            TypeError, "build_packed_collator must be callable"
        ):
            replace(streaming, build_packed_collator=True)


if __name__ == "__main__":
    unittest.main()
