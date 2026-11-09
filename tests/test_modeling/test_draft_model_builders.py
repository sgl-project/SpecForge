"""Canonical model dispatch preserves one golden configuration and boundaries."""

import subprocess
import sys
import unittest
from abc import ABC, abstractmethod
from dataclasses import replace
from unittest.mock import patch

import torch
from torch import nn

from specforge.config.draft.registry import _SPECS, architecture_spec, register_spec
from specforge.modeling import build_draft_model
from specforge.modeling.draft.hf_config import SpecForgeDraftConfig
from specforge.modeling.draft.models import register_draft_model, resolve_draft_model
from specforge.modeling.draft.models.registry import _MODELS
from specforge.modeling.draft.pretrained import DraftPreTrainedModel
from tests.test_config.draft_fixtures import draft_config


class DFlashDraftModel(DraftPreTrainedModel):
    """Test implementation; no production architecture is switched in this PR."""

    def __init__(self, config, *, device=None, dtype=None):
        super().__init__(config)
        self.proj = nn.Linear(config.draft.hidden_size, 4, device=device, dtype=dtype)
        self.post_init()

    def forward(self, hidden):
        return self.proj(hidden)


class ModelBuilderTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(_MODELS, clear=True))
        self.enterContext(patch.dict(_SPECS))

    def test_build_keeps_the_golden_node_and_passes_runtime_options(self):
        self.assertIs(register_draft_model(DFlashDraftModel), DFlashDraftModel)
        config = draft_config(target_bindings=(), target_logit_transform=None)
        model = build_draft_model(config, dtype=torch.float64)
        self.assertIsInstance(model, DFlashDraftModel)
        self.assertIs(model.config.draft, config)
        self.assertEqual(model.proj.weight.dtype, torch.float64)
        model(torch.randn(2, config.hidden_size, dtype=torch.float64)).sum().backward()
        self.assertIsNotNone(model.proj.weight.grad)
        self.assertFalse(model.is_backend_compatible())
        meta = build_draft_model(config, device="meta")
        self.assertTrue(all(parameter.is_meta for parameter in meta.parameters()))

    def test_registration_is_idempotent_and_conflicts_fail(self):
        register_draft_model(DFlashDraftModel)
        register_draft_model(DFlashDraftModel)
        self.assertIs(resolve_draft_model("DFlashDraftModel"), DFlashDraftModel)
        conflicting = type("DFlashDraftModel", (DraftPreTrainedModel,), {})
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_draft_model(conflicting)

    def test_extension_schema_and_model_register_separately(self):
        class ExternalDraft(DraftPreTrainedModel):
            pass

        with self.assertRaisesRegex(ValueError, "unknown architecture"):
            register_draft_model(ExternalDraft)
        spec = architecture_spec("DFlashDraftModel")
        register_spec("ExternalDraft", spec.schema, spec.validate)
        register_draft_model(ExternalDraft)
        config = replace(draft_config(), architecture="ExternalDraft")
        self.assertIsInstance(build_draft_model(config), ExternalDraft)

    def test_missing_implementations_and_nonresolved_configs_fail_before_allocation(
        self,
    ):
        with patch("torch.nn.Linear") as allocate:
            with self.assertRaisesRegex(ValueError, "no canonical model"):
                build_draft_model(draft_config())
            for invalid in ({}, "config.json", SpecForgeDraftConfig(draft_config())):
                with self.assertRaisesRegex(TypeError, "resolved DraftConfig"):
                    build_draft_model(invalid)
            with self.assertRaisesRegex(ValueError, "unknown architecture"):
                resolve_draft_model("UnknownDraftModel")
            allocate.assert_not_called()

    def test_wrong_base_shell_serving_flags_and_abstract_models_are_rejected(self):
        for invalid in (object, nn.Linear, lambda: None):
            with self.assertRaises(TypeError):
                register_draft_model(invalid)
        for fields, error in (
            ({"config_class": object}, TypeError),
            ({"_supports_attention_backend": True}, ValueError),
        ):
            invalid = type("DFlashDraftModel", (DraftPreTrainedModel,), fields)
            with self.assertRaises(error):
                register_draft_model(invalid)

        class AbstractDraft(DraftPreTrainedModel, ABC):
            @abstractmethod
            def forward(self, hidden):
                pass

        with self.assertRaisesRegex(TypeError, "concrete"):
            register_draft_model(AbstractDraft)

    def test_a_constructor_cannot_replace_the_golden_configuration(self):
        class DFlashDraftModel(DraftPreTrainedModel):
            def __init__(self, config):
                super().__init__(
                    SpecForgeDraftConfig(replace(config.draft, initializer_range=0.3))
                )

        register_draft_model(DFlashDraftModel)
        with self.assertRaisesRegex(ValueError, "golden DraftConfig"):
            build_draft_model(draft_config())

    def test_public_import_stays_lazy(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; import specforge.modeling; "
                "assert 'torch' not in sys.modules; "
                "assert 'transformers' not in sys.modules; "
                "assert 'specforge.modeling.draft.models' not in sys.modules",
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
