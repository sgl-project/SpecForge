"""Canonical MTP settings reproduce the existing model, after the main #1014 fix."""

import unittest
from types import SimpleNamespace

import torch

from specforge.config.draft.codec import decode_legacy_config
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.config import AutoDraftConfig
from specforge.modeling.draft.mtp.qwen3_5 import PartialRotaryEmbedding
from tests.test_config.test_draft_legacy_mtp import mtp_payload, target_for_mtp


def model_from_node(node):
    attention = node.attention_configs["default"]
    if not attention.qk_norm or node.norm_config.weight_offset != 1:
        raise AssertionError("legacy MTP cannot express this node")
    payload = dict(
        architectures=["Qwen3_5MTPDraftModel"],
        hidden_size=node.hidden_size,
        vocab_size=node.vocab_size,
        num_hidden_layers=node.num_hidden_layers,
        num_attention_heads=attention.num_attention_heads,
        num_key_value_heads=attention.num_key_value_heads,
        head_dim=attention.head_dim,
        attention_bias=attention.attention_bias,
        attention_dropout=attention.attention_dropout,
        attn_output_gate=attention.output_gate,
        intermediate_size=node.mlp_configs["default"].intermediate_size,
        hidden_act=node.mlp_configs["default"].hidden_act,
        rms_norm_eps=node.norm_config.eps,
        max_position_embeddings=node.max_position_embeddings,
        mtp_config={"share_lm_head": "lm_head" in node.target_bindings},
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
    )
    config = AutoDraftConfig.from_dict(payload)
    config._attn_implementation = "eager"
    model = AutoDraftModel.from_config(config)
    # Consume the canonical theta directly, without the HF compatibility bug.
    model.mtp.rotary_emb = PartialRotaryEmbedding(
        SimpleNamespace(
            rope_theta=node.rope_parameters.rope_theta,
            partial_rotary_factor=node.rope_parameters.partial_rotary_factor,
        ),
        attention.head_dim,
    )
    return model


class MTPCodecParityTest(unittest.TestCase):
    def test_original_recipe_layout_rotary_and_hf_resave(self):
        payload = mtp_payload()
        target = target_for_mtp(payload)
        node = decode_legacy_config(payload, target_config=target)
        config = AutoDraftConfig.from_dict(payload)
        with torch.device("meta"):
            old = AutoDraftModel.from_config(config)
            resolved = model_from_node(node)
        self.assertEqual(
            {key: tuple(value.shape) for key, value in old.state_dict().items()},
            {key: tuple(value.shape) for key, value in resolved.state_dict().items()},
        )
        self.assertEqual(node.norm_config.weight_offset, 1)
        self.assertEqual(
            node, decode_legacy_config(config.to_dict(), target_config=target)
        )
        expected = PartialRotaryEmbedding(config, config.head_dim)
        actual = PartialRotaryEmbedding(
            SimpleNamespace(
                rope_theta=node.rope_parameters.rope_theta,
                partial_rotary_factor=node.rope_parameters.partial_rotary_factor,
            ),
            config.head_dim,
        )
        torch.testing.assert_close(actual.inv_freq, expected.inv_freq, rtol=0, atol=0)
        # The declared theta is also what main computes after #1014.
        intended = PartialRotaryEmbedding(
            SimpleNamespace(
                rope_theta=payload["rope_theta"], partial_rotary_factor=0.25
            ),
            config.head_dim,
        )
        torch.testing.assert_close(actual.inv_freq, intended.inv_freq, rtol=0, atol=0)
        positions = torch.arange(32768).unsqueeze(0)
        for dtype in (torch.float32, torch.bfloat16):
            hidden = torch.empty(1, 1, config.hidden_size, dtype=dtype)
            torch.testing.assert_close(
                actual.to(dtype)(hidden, positions),
                expected.to(dtype)(hidden, positions),
                rtol=0,
                atol=0,
            )

    def test_outputs_and_gradients_match_with_binding_and_attention_choices(self):
        for overrides in (
            {},
            {"mtp_config": {"share_lm_head": False}},
            {"attn_output_gate": False},
            {"attention_bias": True},
            {
                "num_key_value_heads": None,
                "rms_norm_eps": 1e-5,
                "hidden_act": "gelu_pytorch_tanh",
            },
            {"partial_rotary_factor": 1.0},
            {"attention_dropout": 0.2},
        ):
            with self.subTest(overrides=overrides):
                payload = (
                    mtp_payload()
                    | dict(
                        hidden_size=32,
                        intermediate_size=64,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        head_dim=16,
                        vocab_size=64,
                        max_position_embeddings=64,
                        pad_token_id=None,
                        bos_token_id=None,
                        eos_token_id=None,
                    )
                    | overrides
                )
                node = decode_legacy_config(
                    payload, target_config=target_for_mtp(payload)
                )
                config = AutoDraftConfig.from_dict(payload)
                config._attn_implementation = "eager"
                old = AutoDraftModel.from_config(config)
                resolved = model_from_node(node)
                resolved.load_state_dict(old.state_dict(), strict=True)
                # Model the existing binding/freeze step without a target checkpoint.
                for model in (old, resolved):
                    model.embed_tokens.requires_grad_(False)
                    model.mtp.lm_head.requires_grad_(
                        "lm_head" not in node.target_bindings
                    )
                ids = torch.randint(0, 64, (2, 4))
                features = torch.randn(2, 4, 32)
                positions = torch.arange(1, 5).expand(2, -1)
                mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
                outputs = []
                for model in (old, resolved):
                    torch.manual_seed(42)
                    hidden = model.forward_hidden(ids, features, mask, positions)
                    logits = model.mtp.lm_head(hidden)
                    logits.square().sum().backward()
                    outputs.append((hidden, logits))
                torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
                for (name, left), (_, right) in zip(
                    old.named_parameters(), resolved.named_parameters()
                ):
                    with self.subTest(parameter=name):
                        if left.requires_grad:
                            self.assertIsNotNone(left.grad)
                            torch.testing.assert_close(
                                right.grad, left.grad, rtol=0, atol=0
                            )
                        else:
                            self.assertIsNone(left.grad)
                            self.assertIsNone(right.grad)


if __name__ == "__main__":
    unittest.main()
