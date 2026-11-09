"""Draft sources and objective options have one canonical training contract."""

import unittest

from pydantic import ValidationError

from specforge.application import resolve_run
from specforge.config import Config

BASE = {
    "model": {"target_model_path": "target"},
    "data": {"hidden_states_path": "features"},
    "draft": "specforge/config/draft/presets/qwen3-8b-dflash.yaml",
}


class ModelLoadingSchemaTest(unittest.TestCase):
    def test_structure_is_required_and_legacy_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "requires draft or a checkpoint"):
            resolve_run(Config.model_validate({**BASE, "draft": None}))
        for key in (
            "draft_model_config",
            "draft_checkpoint_path",
            "draft_num_hidden_layers",
            "mask_token_id",
        ):
            with self.subTest(field=key), self.assertRaises(ValidationError):
                Config.model_validate({**BASE, "model": {**BASE["model"], key: 1}})
        for key in ("strategy", "num_anchors", "fused_logits_loss", "compact_teacher"):
            with self.subTest(field=key), self.assertRaises(ValidationError):
                Config.model_validate({**BASE, "training": {key: True}})

    def test_objective_options_and_structure_remain_separate(self):
        cfg = Config.model_validate(
            {
                **BASE,
                "training": {
                    "objective_options": {"num_anchors": 8, "fused_logits_loss": False}
                },
            }
        )
        run = resolve_run(cfg)
        self.assertEqual(run.config.training.objective, "dflash")
        self.assertEqual(run.config.training.objective_options["num_anchors"], 8)
        with self.assertRaises(TypeError):
            resolve_run(
                Config.model_validate(
                    {**BASE, "training": {"objective_options": {"unknown": 1}}}
                )
            )

    def test_warm_start_and_resume_are_exclusive(self):
        with self.assertRaisesRegex(ValidationError, "mutually exclusive"):
            Config.model_validate(
                {**BASE, "initialization": {"checkpoint": "base"}, "resume": "run"}
            )
        with self.assertRaisesRegex(ValidationError, "requires draft structure"):
            Config.model_validate(
                {**BASE, "draft": None, "initialization": {"from_target": {"a": "b"}}}
            )


if __name__ == "__main__":
    unittest.main()
