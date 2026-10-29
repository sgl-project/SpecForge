"""Component validation and compatibility paths for draft models."""

import importlib
import unittest
from unittest.mock import patch

from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.config import AutoDraftConfig
from tests.test_modeling.test_all_draft_configs import architecture_payloads


class DraftComponentsTest(unittest.TestCase):
    def test_unimplemented_components_fail_before_allocation(self):
        for name, payload in architecture_payloads().items():
            for field, kind in [
                ("attention_config", "dsa"),
                ("attention_config", "qsa"),
                ("attention_config", "kda"),
                ("residual_config", "gr"),
                ("residual_config", "mhc"),
                ("ffn_config", "moe"),
            ]:
                with self.subTest(architecture=name, component=kind):
                    config = AutoDraftConfig.from_dict(
                        {**payload, field: {"type": kind}}
                    )
                    with patch(
                        "torch.nn.Linear", side_effect=AssertionError("allocated")
                    ):
                        with self.assertRaisesRegex(ValueError, "not implemented"):
                            AutoDraftModel.from_config(config)

    def test_component_typos_are_not_silently_ignored(self):
        payload = architecture_payloads()["DFlashDraftModel"]
        for field in ["attention_config", "residual_config", "ffn_config"]:
            with self.subTest(field=field):
                config = AutoDraftConfig.from_dict({**payload, field: {"typo": 1}})
                with self.assertRaisesRegex(ValueError, "Unknown"):
                    AutoDraftConfig.finalize(config)

    def test_legacy_imports_refer_to_the_same_implementations(self):
        for old, new in [
            ("dflash", "models.dflash"),
            ("dspark", "models.dspark"),
            ("llama3_eagle", "models.eagle3"),
            ("mtp.qwen3_5", "models.specialized.qwen3_5_mtp.model"),
        ]:
            with self.subTest(module=old):
                self.assertIs(
                    importlib.import_module(f"specforge.modeling.draft.{old}"),
                    importlib.import_module(f"specforge.modeling.draft.{new}"),
                )
