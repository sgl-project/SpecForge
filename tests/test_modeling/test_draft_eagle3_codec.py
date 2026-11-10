"""Reconstruct existing EAGLE3 models from converted settings as a parity oracle."""

import json
import unittest
from unittest.mock import patch

import torch

from specforge.config.draft.codec import decode_legacy_config
from specforge.config.draft.codec.components import encode_component
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft import llama3_eagle
from specforge.modeling.draft.config import AutoDraftConfig
from tests.test_config.test_draft_legacy_dflash import CONFIGS
from tests.test_config.test_draft_legacy_eagle3 import target_for_eagle
from tests.test_modeling.test_draft_eagle_rope_codec import rotary_from_node


def legacy_config(payload):
    # HF 5 rejects the historical Qwen2.5-VL-32B recipe even though EAGLE
    # supports its explicit head_dim. Restore its head count after constructing
    # the config so the oracle still checks the original parameter dimensions.
    heads = payload["num_attention_heads"]
    if payload.get("head_dim") and payload["hidden_size"] % heads:
        config = AutoDraftConfig.from_dict(payload | {"num_attention_heads": 1})
        config.num_attention_heads = heads
        return config
    return AutoDraftConfig.from_dict(payload)


def model_from_node(node):
    attention = node.attention_configs["default"]
    if (attention.qk_norm, attention.attention_bias, attention.output_gate) != (
        False,
        False,
        False,
    ) or node.norm_config.weight_offset != 0:
        raise AssertionError("legacy EAGLE cannot express this node")
    payload = dict(
        architectures=["LlamaForCausalLMEagle3"],
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
        target_hidden_size=node.spec_config.target_features.hidden_size,
        fc_norm=node.spec_config.normalize_features,
        norm_output=node.spec_config.norm_output,
        # Supply the resolved rotary module below, using the existing EAGLE
        # numerical primitives rather than interpreting canonical fields as HF defaults.
        rope_parameters={
            "rope_type": "default",
            "rope_theta": node.rope_parameters.rope_theta,
        },
        bos_token_id=None,
        eos_token_id=None,
    )
    model = AutoDraftModel.from_config(legacy_config(payload))
    model.midlayer.self_attn.rotary_emb = rotary_from_node(
        node.rope_parameters, attention.head_dim, node.max_position_embeddings
    )
    model.midlayer.self_attn.rope_scaling = encode_component(
        "rope", node.rope_parameters
    )
    return model


class Eagle3CodecParityTest(unittest.TestCase):
    def test_defaults_and_explicit_values_match_hf_configuration(self):
        for architecture in (
            "LlamaForCausalLMEagle3",
            "PEagleDraftModel",
            "Qwen3_5MTPDraftModel",
        ):
            for overrides in (
                {},
                dict(
                    rms_norm_eps=1e-5,
                    hidden_act="gelu_pytorch_tanh",
                    initializer_range=0.03,
                    num_key_value_heads=None,
                ),
            ):
                with self.subTest(architecture=architecture, overrides=overrides):
                    payload = (
                        dict(
                            architectures=[architecture],
                            hidden_size=256,
                            vocab_size=64,
                            num_hidden_layers=1,
                            num_attention_heads=32,
                            intermediate_size=64,
                        )
                        | overrides
                    )
                    options = (
                        {"mask_token_id": 1}
                        if architecture == "PEagleDraftModel"
                        else {}
                    )
                    node = decode_legacy_config(
                        payload, target_config=target_for_eagle(payload), **options
                    )
                    hf = AutoDraftConfig.from_dict(payload)
                    attention = node.attention_configs["default"]
                    self.assertEqual(attention.head_dim, hf.head_dim)
                    self.assertEqual(attention.attention_dropout, hf.attention_dropout)
                    self.assertEqual(node.norm_config.eps, hf.rms_norm_eps)
                    self.assertEqual(
                        node.mlp_configs["default"].hidden_act, hf.hidden_act
                    )
                    self.assertEqual(node.initializer_range, hf.initializer_range)
                    self.assertEqual(
                        node.max_position_embeddings, hf.max_position_embeddings
                    )
                    self.assertEqual(
                        node.attention_configs["default"].num_key_value_heads,
                        hf.num_key_value_heads,
                    )
                    self.assertEqual(
                        node.rope_parameters.rope_theta,
                        hf.rope_parameters["rope_theta"],
                    )

    def test_all_recipes_preserve_parameter_names_shapes_and_saved_settings(self):
        count = 0
        for path in CONFIGS.glob("*.json"):
            payload = json.loads(path.read_text())
            if payload.get("architectures") != ["LlamaForCausalLMEagle3"]:
                continue
            count += 1
            with self.subTest(path=path.name):
                target = target_for_eagle(payload)
                node = decode_legacy_config(payload, target_config=target)
                # Keep original recipe dimensions without allocating large weights.
                with torch.device("meta"):
                    old = AutoDraftModel.from_config(legacy_config(payload))
                    resolved = model_from_node(node)
                self.assertEqual(
                    {
                        key: tuple(value.shape)
                        for key, value in old.state_dict().items()
                    },
                    {
                        key: tuple(value.shape)
                        for key, value in resolved.state_dict().items()
                    },
                )
                self.assertEqual(
                    decode_legacy_config(old.config.to_dict(), target_config=target),
                    node,
                )
        self.assertGreaterEqual(count, 27)

    def test_outputs_and_gradients_match_for_normalization_and_rotary_variants(self):
        cases = [
            ("qwen3-8b-eagle3.json", {}),
            ("qwen3-8b-eagle3.json", {"num_hidden_layers": 8}),
            ("qwen3-30B-A3B-eagle3.1.json", {}),
            ("qwen2-5-vl-7b-eagle3.json", {}),
            ("deepseek-v2-lite-eagle3.json", {}),
            ("llama3-70B-ealge3.json", {}),
            (
                "qwen3-8b-eagle3.json",
                {"rms_norm_eps": 1e-5, "hidden_act": "gelu_pytorch_tanh"},
            ),
        ]
        for name, overrides in cases:
            with self.subTest(name=name, overrides=overrides):
                payload = (
                    json.loads((CONFIGS / name).read_text())
                    | dict(
                        hidden_size=32,
                        intermediate_size=64,
                        num_attention_heads=2,
                        num_key_value_heads=1,
                        head_dim=128,
                        vocab_size=64,
                        draft_vocab_size=24,
                        max_position_embeddings=64,
                        bos_token_id=None,
                        eos_token_id=None,
                        pad_token_id=None,
                    )
                    | overrides
                )
                node = decode_legacy_config(
                    payload, target_config=target_for_eagle(payload)
                )
                old = AutoDraftModel.from_config(AutoDraftConfig.from_dict(payload))
                resolved = model_from_node(node)
                resolved.load_state_dict(old.state_dict(), strict=True)
                features, embeddings = torch.randn(1, 4, 96), torch.randn(1, 4, 32)
                positions = torch.arange(4).unsqueeze(0)
                if node.rope_parameters.rope_type == "mrope":
                    positions = torch.arange(12).reshape(3, 1, 4)
                mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
                outputs = []
                with (
                    patch.object(
                        llama3_eagle.LlamaRotaryEmbedding,
                        "forward",
                        llama3_eagle.LlamaRotaryEmbedding.forward.__wrapped__,
                    ),
                    patch.object(
                        llama3_eagle,
                        "apply_rotary_pos_emb",
                        llama3_eagle.apply_rotary_pos_emb.__wrapped__,
                    ),
                ):
                    for model in (old, resolved):
                        hidden = model.backbone(
                            input_embeds=embeddings,
                            hidden_states=model.project_hidden_states(features),
                            cache_hidden=None,
                            attention_mask=mask,
                            position_ids=positions,
                        )
                        logits = model.compute_logits(hidden)
                        logits.square().sum().backward()
                        outputs.append((hidden, logits))
                torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
                for (parameter, left), (_, right) in zip(
                    old.named_parameters(), resolved.named_parameters()
                ):
                    with self.subTest(parameter=parameter):
                        if left.grad is None:
                            self.assertIsNone(right.grad)
                        else:
                            torch.testing.assert_close(
                                right.grad, left.grad, rtol=0, atol=0
                            )


if __name__ == "__main__":
    unittest.main()
