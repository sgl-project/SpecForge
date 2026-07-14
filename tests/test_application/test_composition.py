from __future__ import annotations

import unittest

from specforge.application import resolve_run
from specforge.config import Config, TrainingConfig


def _payload(
    algorithm: str = "eagle3",
    *,
    mode: str = "online",
    modality: str = "text",
) -> dict:
    model = {
        "target_model_path": "target",
        "draft_model_config": "draft.json",
        "input_modality": modality,
    }
    if algorithm in {"eagle3", "peagle"} and mode == "online":
        model["vocab_mapping_path"] = "mapping.pt"
    training = {
        "strategy": algorithm,
        "max_steps": 1,
        "attention_backend": "flex_attention",
        "batch_size": 1,
    }
    if mode == "offline":
        return {
            "model": model,
            "data": {"hidden_states_path": "features"},
            "training": training,
        }
    return {
        "model": model,
        "data": {"train_data_path": "prompts.jsonl"},
        "training": training,
        "deployment": {
            "mode": "disaggregated",
            "disaggregated": {
                "control_dir": "outputs/test/control",
                "backend": "mooncake",
                "server_urls": ["http://capture:30000"],
            },
        },
    }


class ApplicationCompositionTest(unittest.TestCase):
    def test_resolve_run_pairs_config_with_one_registration(self):
        config = Config.model_validate(_payload())

        resolved = resolve_run(config)

        self.assertIs(config, resolved.config)
        self.assertEqual("eagle3", resolved.algorithm.name)

    def test_unknown_algorithm_fails_at_the_composition_root(self):
        config = Config.model_validate(_payload("unknown"))

        with self.assertRaisesRegex(ValueError, "registered algorithms"):
            resolve_run(config)

    def test_training_domain_has_no_duplicate_deployment_fields(self):
        fields = set(TrainingConfig.model_fields)

        self.assertTrue(
            {"deployment_mode", "server_urls", "metadata_db_path"}.isdisjoint(fields)
        )

    def test_unsupported_modality_fails_through_generic_provider_boundary(self):
        config = Config.model_validate(_payload(modality="future_vision_language"))

        with self.assertRaisesRegex(
            ValueError,
            "no streaming feature contract and provider",
        ):
            resolve_run(config)

    def test_mode_constraints_are_derived_from_feature_contracts(self):
        for algorithm in ("peagle", "dspark"):
            with self.subTest(algorithm=algorithm):
                config = Config.model_validate(_payload(algorithm, mode="offline"))
                with self.assertRaisesRegex(ValueError, "no offline feature contract"):
                    resolve_run(config)

    def test_peagle_remains_server_streaming_capable(self):
        resolved = resolve_run(Config.model_validate(_payload("peagle")))

        stream = resolved.algorithm.providers.server_streaming_for("text")
        self.assertEqual("eagle3", stream.capture_method)

    def test_algorithm_draft_override_constraints_are_planned(self):
        payload = _payload("eagle3")
        payload["model"]["draft_num_hidden_layers"] = 2

        with self.assertRaisesRegex(
            ValueError, "requires model.draft_num_hidden_layers=1"
        ):
            resolve_run(Config.model_validate(payload))

        domino = _payload("domino")
        domino["model"]["draft_block_size"] = 8
        with self.assertRaisesRegex(
            ValueError, "does not support model.draft_block_size"
        ):
            resolve_run(Config.model_validate(domino))

    def test_disaggregated_vocab_mapping_is_provider_owned(self):
        payload = _payload("peagle")
        payload["model"].pop("vocab_mapping_path")

        with self.assertRaisesRegex(ValueError, "require model.vocab_mapping_path"):
            resolve_run(Config.model_validate(payload))


if __name__ == "__main__":
    unittest.main()
