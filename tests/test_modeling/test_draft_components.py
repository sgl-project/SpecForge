"""Component validation for draft models."""

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
