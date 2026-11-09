"""Migrated P-EAGLE settings preserve existing parameter shapes and computation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from specforge.algorithms.model_providers import resolve_eagle_capture_layers
from specforge.config.draft.codec import decode_legacy_config
from specforge.config.draft.codec.components import encode_component
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.config import AutoDraftConfig
from specforge.modeling.draft.llama3_eagle import LlamaRotaryEmbedding
from tests.test_config.test_draft_legacy_peagle import peagle_payload, target_for_peagle


def reconstructed_payload(node):
    attention = node.attention_configs["default"]
    return dict(
        architectures=["PEagleDraftModel"],
        hidden_size=node.hidden_size,
        vocab_size=node.vocab_size,
        draft_vocab_size=node.draft_vocab_size,
        num_hidden_layers=node.num_hidden_layers,
        num_attention_heads=attention.num_attention_heads,
        num_key_value_heads=attention.num_key_value_heads,
        head_dim=attention.head_dim,
        intermediate_size=node.mlp_configs["default"].intermediate_size,
        hidden_act=node.mlp_configs["default"].hidden_act,
        rms_norm_eps=node.norm_config.eps,
        max_position_embeddings=node.max_position_embeddings,
        rope_parameters=encode_component("rope", node.rope_parameters),
        norm_before_residual=node.spec_config.norm_before_residual,
        target_hidden_size=node.spec_config.target_features.hidden_size,
        eagle_config={
            "eagle_aux_hidden_state_layer_ids": list(
                node.spec_config.target_features.target_layer_ids
            )
        },
        bos_token_id=None,
        eos_token_id=None,
        pad_token_id=node.spec_config.pad_token_id,
    )


def eager_attention(*, query, key, value, block_mask, enable_gqa, **kwargs):
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=block_mask, enable_gqa=enable_gqa
    )


class PEagleCodecParityTest(unittest.TestCase):
    def test_shapes_outputs_gradients_and_saved_configs_match(self):
        for depth, normalized_residual in ((1, False), (3, False), (3, True)):
            with self.subTest(depth=depth, normalized_residual=normalized_residual):
                payload = peagle_payload() | dict(
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=depth,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=8,
                    vocab_size=64,
                    draft_vocab_size=24,
                    max_position_embeddings=64,
                    norm_before_residual=normalized_residual,
                    bos_token_id=None,
                    eos_token_id=None,
                    pad_token_id=0,
                    rope_scaling={"rope_type": "linear", "factor": 4},
                    partial_rotary_factor=0.5,
                    attention_bias=True,
                    mlp_bias=True,
                    attention_dropout=0.25,
                )
                target = target_for_peagle(payload)
                node = decode_legacy_config(payload, target_config=target)
                old = AutoDraftModel.from_config(AutoDraftConfig.from_dict(payload))
                resolved = AutoDraftModel.from_config(
                    AutoDraftConfig.from_dict(reconstructed_payload(node))
                )
                resolved.load_state_dict(old.state_dict(), strict=True)
                for index, layer in enumerate(old.layers):
                    self.assertEqual(
                        layer.self_attn.q_proj.in_features,
                        node.spec_config.attention_input_width(node.hidden_size, index),
                    )
                self.assertEqual(
                    tuple(old.mask_hidden.shape), node.spec_config.mask_hidden_shape
                )
                self.assertEqual(
                    old.fc.in_features, node.spec_config.target_features.input_width
                )
                self.assertEqual(
                    decode_legacy_config(old.config.to_dict(), target_config=target),
                    node,
                )
                torch.testing.assert_close(
                    old.rotary_emb.inv_freq,
                    resolved.rotary_emb.inv_freq,
                    rtol=0,
                    atol=0,
                )
                features = torch.randn(
                    1, 4, node.spec_config.target_features.input_width
                )
                ids = torch.tensor([[0, 1, 2, 3]])
                mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
                positions = torch.arange(4).unsqueeze(0)
                # Use the same eager attention for both old models, so this CPU
                # check covers configuration semantics rather than GPU kernels.
                with (
                    patch(
                        "specforge.modeling.draft.peagle.compile_friendly_flex_attention",
                        side_effect=eager_attention,
                    ),
                    patch.object(
                        LlamaRotaryEmbedding,
                        "forward",
                        LlamaRotaryEmbedding.forward.__wrapped__,
                    ),
                ):
                    expected = old(
                        features,
                        old.embed_tokens(ids),
                        attention_mask=mask,
                        position_ids=positions,
                    )
                    actual = resolved(
                        features,
                        resolved.embed_tokens(ids),
                        attention_mask=mask,
                        position_ids=positions,
                    )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                old.lm_head(expected).square().sum().backward()
                resolved.lm_head(actual).square().sum().backward()
                for (name, left), (_, right) in zip(
                    old.named_parameters(), resolved.named_parameters()
                ):
                    with self.subTest(parameter=name):
                        if left.grad is None:
                            self.assertIsNone(right.grad)
                        else:
                            torch.testing.assert_close(
                                right.grad, left.grad, rtol=0, atol=0
                            )

    def test_capture_defaults_match_existing_provider(self):
        run = SimpleNamespace(model=SimpleNamespace(aux_hidden_state_layer_ids=None))
        for depth in (4, 28, 36, 64):
            for explicit in (None, [2, 1, 2]):
                payload = peagle_payload()
                if explicit is not None:
                    payload["eagle_config"] = {
                        "eagle_aux_hidden_state_layer_ids": explicit
                    }
                target = target_for_peagle(payload) | {"num_hidden_layers": depth}
                expected = resolve_eagle_capture_layers(
                    run, payload, SimpleNamespace(**target)
                )
                node = decode_legacy_config(payload, target_config=target)
                self.assertEqual(
                    node.spec_config.target_features.target_layer_ids, tuple(expected)
                )


if __name__ == "__main__":
    unittest.main()
