"""Generic HF construction owns initialization and shared-backbone checkpointing."""

import copy
import unittest
from tempfile import TemporaryDirectory

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers import initialization as init

from specforge.modeling.draft.backbone import DraftBackbone, LayerInputs
from specforge.modeling.draft.hf_config import SpecForgeDraftConfig
from specforge.modeling.draft.layers.residual import StandardResidual
from specforge.modeling.draft.pretrained import DraftPreTrainedModel
from tests.test_config.draft_fixtures import draft_config
from tests.test_modeling.test_draft_backbone import TestLayer, TestRotary


class OffsetNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(width))

    def reset_draft_parameters(self):
        init.zeros_(self.weight)


class DFlashDraftModel(DraftPreTrainedModel):
    """Small test model; the production DFlash implementation is unchanged."""

    def __init__(self, config):
        super().__init__(config)
        draft = config.draft
        self.embed_tokens = nn.Embedding(
            draft.vocab_size, draft.hidden_size, padding_idx=0
        )
        self.lm_head = nn.Linear(draft.hidden_size, draft.draft_vocab_size, bias=False)
        self.special_norm = OffsetNorm(draft.hidden_size)
        self.backbone = DraftBackbone(
            [TestLayer(draft.hidden_size) for _ in draft.layer_types],
            final_norm=nn.LayerNorm(draft.hidden_size),
            rotary=[TestRotary(0.5) for _ in draft.layer_types],
            residual=StandardResidual(draft.residual_config),
        )
        self.post_init()

    def forward(self, hidden, layer_inputs):
        return self.backbone(hidden, layer_inputs)


def model_config(**changes):
    return SpecForgeDraftConfig(
        draft_config(target_bindings=(), target_logit_transform=None, **changes)
    )


def model_inputs(model):
    width = model.config.draft.hidden_size
    return torch.randn(1, 3, width), [
        LayerInputs(
            target_context=torch.randn(1, 3, width),
            position_ids=torch.arange(3).unsqueeze(0),
        )
        for _ in model.config.draft.layer_types
    ]


class DraftPreTrainedTest(unittest.TestCase):
    def test_generic_base_uses_golden_initialization_and_is_not_a_serving_backend(self):
        config = model_config(initializer_range=0)
        before = hash(config.draft)
        model = DFlashDraftModel(config)
        self.assertIsInstance(model, PreTrainedModel)
        self.assertIs(model.config, config)
        self.assertFalse(model.is_backend_compatible())
        self.assertEqual(DraftPreTrainedModel.__bases__, (PreTrainedModel,))
        self.assertFalse(hasattr(config, "initializer_range"))
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                self.assertEqual(torch.count_nonzero(module.weight).item(), 0)
            elif isinstance(module, nn.LayerNorm):
                torch.testing.assert_close(
                    module.weight, torch.ones_like(module.weight)
                )
        torch.testing.assert_close(
            model.special_norm.weight, torch.zeros(config.draft.hidden_size)
        )
        self.assertEqual(hash(config.draft), before)

    def test_custom_initialization_is_required_and_loaded_weights_are_preserved(self):
        model = DFlashDraftModel(model_config())
        with self.assertRaisesRegex(TypeError, "reset_draft_parameters"):
            model._init_weights(nn.Conv1d(4, 4, 1))
        with torch.no_grad():
            model.lm_head.weight.fill_(7)
        model.post_init()
        torch.testing.assert_close(
            model.lm_head.weight, torch.full_like(model.lm_head.weight, 7)
        )
        # A partially restored module must not reinitialize its loaded tensor.
        layer = nn.Linear(4, 4)
        with torch.no_grad():
            layer.weight.fill_(3)
        layer.weight._is_hf_initialized = True
        model._init_weights(layer)
        torch.testing.assert_close(layer.weight, torch.full_like(layer.weight, 3))
        torch.testing.assert_close(layer.bias, torch.zeros_like(layer.bias))

    def test_shared_weights_follow_the_draft_view(self):
        tied = DFlashDraftModel(model_config(tie_word_embeddings=True))
        self.assertIs(tied.lm_head.weight, tied.embed_tokens.weight)
        untied = DFlashDraftModel(model_config())
        self.assertIsNot(untied.lm_head.weight, untied.embed_tokens.weight)

    def test_checkpointing_preserves_frozen_input_gradients_and_structure(self):
        plain = DFlashDraftModel(model_config())
        checked = copy.deepcopy(plain)
        checked.gradient_checkpointing_enable()
        self.assertTrue(checked.is_gradient_checkpointing)
        hidden, inputs = model_inputs(plain)
        expected, actual = plain(hidden, inputs), checked(hidden, inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.square().sum().backward()
        actual.square().sum().backward()
        for left, right in zip(
            plain.backbone.parameters(), checked.backbone.parameters()
        ):
            self.assertIsNotNone(right.grad)
            torch.testing.assert_close(right.grad, left.grad, rtol=0, atol=0)
        self.assertEqual(checked.config.draft, plain.config.draft)
        checked.gradient_checkpointing_disable()
        self.assertFalse(checked.is_gradient_checkpointing)
        for options in ({"use_reentrant": True}, {"preserve_rng_state": False}):
            with self.assertRaisesRegex(ValueError, "use_reentrant=False"):
                checked.gradient_checkpointing_enable(options)
        self.assertFalse(checked.is_gradient_checkpointing)

    def test_hf_model_save_reload_keeps_weights_outputs_and_config(self):
        config = model_config()
        config.auto_map = {"AutoModel": "model.DFlashDraftModel"}
        model = DFlashDraftModel(config).eval()
        hidden, inputs = model_inputs(model)
        with TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = DFlashDraftModel.from_pretrained(directory).eval()
        self.assertEqual(restored.config.draft, config.draft)
        self.assertEqual(restored.config.auto_map, config.auto_map)
        torch.testing.assert_close(
            restored.state_dict(), model.state_dict(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            restored(hidden, inputs), model(hidden, inputs), rtol=0, atol=0
        )

    def test_wrong_config_and_architecture_fail_before_building_layers(self):
        with self.assertRaises(TypeError):
            DFlashDraftModel(draft_config())
        other = model_config()

        # A different class cannot silently save itself under DFlash's selector.
        class OtherModel(DraftPreTrainedModel):
            pass

        with self.assertRaisesRegex(ValueError, "class name"):
            OtherModel(other)


if __name__ == "__main__":
    unittest.main()
