"""Canonical sources select structure before any training modules are built."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from specforge.config import Config
from specforge.config.draft.codec import encode_config
from specforge.training.model_loading import resolve_draft_config
from tests.test_config.draft_fixtures import draft_config


class DraftConfigResolutionTest(unittest.TestCase):
    def test_inline_and_file_share_the_same_schema(self):
        draft = draft_config()
        payload = encode_config(draft)
        with TemporaryDirectory() as temp:
            # A preset filename cannot choose a model family or override fields.
            path = Path(temp) / "deepseek.yaml"
            path.write_text(yaml.safe_dump(payload))
            for source in (payload, str(path)):
                cfg = Config.model_validate(
                    dict(
                        draft=source,
                        model=dict(target_model_path="qwen"),
                        data=dict(hidden_states_path="features"),
                    )
                )
                self.assertEqual(resolve_draft_config(cfg), draft)

    def test_structure_is_required_without_checkpoint(self):
        with self.assertRaisesRegex(ValueError, "Training requires draft"):
            Config.model_validate(
                dict(
                    model=dict(target_model_path="target"),
                    data=dict(hidden_states_path="features"),
                )
            )
