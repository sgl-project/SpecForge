"""Training sources and architecture overrides use canonical fields only."""

import unittest

from pydantic import ValidationError

from specforge.config import Config


def payload(**extra):
    return {
        "model": {"target_model_path": "target/model"},
        "data": {"hidden_states_path": "/features"},
        **extra,
    }


class ModelLoadingSchemaTest(unittest.TestCase):
    def test_every_algorithm_requires_an_explicit_source(self):
        with self.assertRaisesRegex(ValidationError, "draft"):
            Config.model_validate(payload())
        for source in (
            {"draft": "draft.yaml"},
            {"initialization": {"checkpoint": "org/draft"}},
            {"resume": "/run/checkpoint"},
        ):
            with self.subTest(source=source):
                Config.model_validate(payload(**source))

    def test_architecture_fields_cannot_override_the_draft(self):
        for field in (
            "draft_model_config",
            "draft_checkpoint_path",
            "draft_num_hidden_layers",
            "draft_block_size",
            "mask_token_id",
            "aux_hidden_state_layer_ids",
        ):
            raw = payload(draft="draft.yaml")
            raw["model"][field] = 1
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValidationError, field),
            ):
                Config.model_validate(raw)

    def test_warm_start_and_full_resume_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValidationError, "mutually exclusive"):
            Config.model_validate(
                payload(
                    initialization={"checkpoint": "/base/checkpoint"},
                    resume="/run/checkpoint",
                )
            )

    def test_producer_keeps_the_same_checkpoint_source(self):
        raw = payload(initialization={"checkpoint": "org/draft"})
        raw.update(
            data={"train_data_path": "/train.jsonl"},
            training={"role": "producer", "total_steps": 1},
            deployment={
                "mode": "disaggregated",
                "disaggregated": {
                    "control_dir": "/control",
                    "backend": "mooncake",
                    "server_urls": ["http://127.0.0.1:30000"],
                },
            },
        )
        config = Config.model_validate(raw)
        self.assertEqual(config.initialization.checkpoint, "org/draft")
