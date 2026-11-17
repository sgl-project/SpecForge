"""P-EAGLE learns feature replacement inside the same forward used for inference."""

import copy
import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from specforge.modeling import build_draft_model
from specforge.modeling.draft.models import PEagleDraftModel
from tests.test_modeling.test_canonical_eagle3 import install_mapping, model_inputs
from tests.test_modeling.test_draft_eagle_backbone import config_for


def model_config(*, normalized=False, bound=False, target_width=48):
    base = config_for("peagle", normalized)
    return replace(
        base,
        target_bindings=("embed_tokens",) if bound else (),
        spec_config=replace(
            base.spec_config,
            target_features=replace(
                base.spec_config.target_features, hidden_size=target_width
            ),
        ),
    )


def feature_mask(device="cpu"):
    return torch.tensor(
        [[False, True, False, True], [True, False, True, False]], device=device
    )


class CanonicalPEagleTest(unittest.TestCase):

    def test_linear_initialization_matches_pytorch_default(self):
        model = build_draft_model(model_config())
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                reference = torch.nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                )
                torch.manual_seed(17)
                reference.reset_parameters()
                torch.manual_seed(17)
                model._init_weights(module)
                torch.testing.assert_close(
                    module.weight, reference.weight, rtol=0, atol=0
                )
                if module.bias is not None:
                    torch.testing.assert_close(
                        module.bias, reference.bias, rtol=0, atol=0
                    )

    def test_custom_linear_initialization_takes_precedence(self):
        class CustomLinear(torch.nn.Linear):
            def reset_draft_parameters(self):
                torch.nn.init.constant_(self.weight, 0.25)

        model = build_draft_model(model_config())
        module = CustomLinear(4, 4, bias=False)
        model._init_weights(module)
        torch.testing.assert_close(module.weight, torch.full_like(module.weight, 0.25))

    def test_checkpointing_binding_and_hf_restore_keep_learned_mask(self):
        config = model_config(bound=True, target_width=32)
        actual = build_draft_model(config)
        install_mapping(actual)
        embedding = torch.randn(64, 32)
        actual.embed_tokens.load_target_weight(embedding)
        with torch.no_grad():
            actual.mask_hidden.fill_(0.37)
        checkpointed = copy.deepcopy(actual)
        checkpointed.gradient_checkpointing_enable()
        args, mask = model_inputs(config), feature_mask()
        output, expected = actual(*args, feature_mask=mask), checkpointed(
            *args, feature_mask=mask
        )
        torch.testing.assert_close(output.logits, expected.logits, rtol=0, atol=0)
        output.logits.sum().backward()
        expected.logits.sum().backward()
        for name, parameter in actual.named_parameters():
            self.assertEqual(parameter.grad is not None, parameter.requires_grad, name)
            if parameter.requires_grad:
                torch.testing.assert_close(
                    parameter.grad,
                    dict(checkpointed.named_parameters())[name].grad,
                    rtol=0,
                    atol=0,
                )
        with TemporaryDirectory() as path:
            actual.save_pretrained(path)
            restored = PEagleDraftModel.from_pretrained(path)
        self.assertFalse(restored.embed_tokens.weight.is_meta)
        restored.to("cpu")
        restored.embed_tokens.load_target_weight(embedding)
        torch.testing.assert_close(
            restored.mask_hidden, actual.mask_hidden, rtol=0, atol=0
        )
        torch.testing.assert_close(
            restored(*args, feature_mask=mask).logits, output.logits, rtol=0, atol=0
        )
        torch.testing.assert_close(
            restored.vocabulary.to_target(torch.arange(32)),
            torch.arange(32) * 2,
            rtol=0,
            atol=0,
        )
        self.assertNotIn("embed_tokens.weight", actual.state_dict())

    def test_mask_application_hidden_only_meta_and_tied_embeddings(self):
        config = model_config()
        actual = build_draft_model(config)
        args = model_inputs(config)
        with patch(
            "specforge.modeling.draft.layers.attention.gqa.AttentionKV",
            side_effect=AssertionError("ordinary attention needs no K/V wrapper"),
        ):
            output = actual(*args, output_logits=False)
        unmasked = actual(
            *args,
            feature_mask=torch.zeros_like(args[0], dtype=torch.bool),
            output_logits=False,
        )
        torch.testing.assert_close(
            output.hidden_states, unmasked.hidden_states, rtol=0, atol=0
        )
        output.hidden_states.sum().backward()
        self.assertIsNotNone(actual.mask_hidden.grad)
        self.assertEqual(actual.mask_hidden.grad.count_nonzero().item(), 0)
        self.assertIsNone(output.logits)
        with self.assertRaisesRegex(RuntimeError, "mapping"):
            actual.compute_logits(output.hidden_states)
        for mask in (torch.ones_like(args[0]), feature_mask()[:1]):
            with self.assertRaisesRegex(ValueError, "feature_mask"):
                actual(*args, feature_mask=mask, output_logits=False)
        meta = build_draft_model(config, device="meta", dtype=torch.bfloat16)
        self.assertTrue(
            all(p.is_meta and p.dtype == torch.bfloat16 for p in meta.parameters())
        )
        self.assertEqual(meta.mask_hidden.shape, config.spec_config.mask_hidden_shape)
        full = build_draft_model(
            replace(config, draft_vocab_size=64, tie_word_embeddings=True)
        )
        self.assertIs(full.embed_tokens.weight, full.lm_head.weight)
        self.assertTrue(full.vocabulary.ready)


if __name__ == "__main__":
    unittest.main()
