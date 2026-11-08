"""Legacy DFlash configs become complete, immutable canonical structures."""

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

from specforge.config.draft.codec import (
    decode_config,
    decode_legacy_config,
    encode_config,
)
from specforge.config.draft.logits import MuPLogitTransformSpec

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def legacy_payload():
    return json.loads((CONFIGS / "qwen3-8b-dflash.json").read_text())


def target_for(payload):
    return {
        "hidden_size": payload["hidden_size"],
        "vocab_size": payload["vocab_size"],
        "num_hidden_layers": payload["num_target_layers"],
    }


class LegacyDFlashTest(unittest.TestCase):
    def test_every_bundled_dflash_recipe_round_trips(self):
        paths = sorted(CONFIGS.glob("*dflash*.json"))
        self.assertGreaterEqual(len(paths), 11)
        for path in paths:
            with self.subTest(path=path.name):
                payload = json.loads(path.read_text())
                before = copy.deepcopy(payload)
                target = target_for(payload)
                node = decode_legacy_config(payload, target_config=target)
                saved = json.loads(json.dumps(encode_config(node)))
                restored = decode_config(saved, target_config=target)
                self.assertEqual(restored, node)
                self.assertEqual(hash(restored), hash(node))
                self.assertEqual(node.num_hidden_layers, payload["num_hidden_layers"])
                self.assertEqual(
                    node.spec_config.target_features.target_layer_ids,
                    tuple(payload["dflash_config"]["target_layer_ids"]),
                )
                self.assertEqual(payload, before)
                self.assertNotIn("auto_map", saved)
                self.assertNotIn("dflash_config", saved)

    def test_alias_conflicts_and_null_vocab_are_rejected(self):
        payload = legacy_payload()
        target = target_for(payload)
        payload["dflash_config"]["block_size"] = payload["block_size"]
        decode_legacy_config(payload, target_config=target)
        payload["dflash_config"]["block_size"] += 1
        with self.assertRaisesRegex(ValueError, "conflicting.*block_size"):
            decode_legacy_config(payload, target_config=target)
        payload = legacy_payload()
        with self.assertRaisesRegex(ValueError, "draft_vocab_size"):
            decode_legacy_config(
                payload | {"draft_vocab_size": None}, target_config=target
            )
        self.assertEqual(
            decode_legacy_config(payload, target_config=target).draft_vocab_size,
            payload["vocab_size"],
        )

    def test_mixed_layer_causality_is_explicit(self):
        payload = legacy_payload() | {
            "num_hidden_layers": 2,
            "layer_types": ["sliding_attention", "full_attention"],
            "use_sliding_window": True,
            "sliding_window": 128,
        }
        target = target_for(payload)
        for causal in (None, True, False):
            node = decode_legacy_config(
                payload | {"is_causal": causal}, target_config=target
            )
            local, full = (node.attention_configs[name] for name in node.layer_types)
            self.assertEqual(local.is_causal, True if causal is None else causal)
            self.assertEqual(full.is_causal, False if causal is None else causal)
            self.assertEqual(local.sliding_window, 128)
            self.assertIsNone(full.sliding_window)

    def test_feature_defaults_and_bound_target_semantics(self):
        payload = legacy_payload()
        target = target_for(payload) | {"logits_mup_width_multiplier": 8}
        payload["dflash_config"].pop("target_layer_ids")
        payload.pop("num_target_layers")
        node = decode_legacy_config(payload, target_config=target)
        self.assertEqual(
            node.spec_config.target_features.target_layer_ids, (1, 9, 17, 25, 33)
        )
        self.assertEqual(
            node.target_logit_transform, MuPLogitTransformSpec(multiplier=8)
        )
        self.assertEqual(node.target_bindings, ("embed_tokens", "lm_head"))
        with self.assertRaisesRegex(ValueError, "target_config"):
            decode_legacy_config(payload, target_config=None)
        with self.assertRaisesRegex(ValueError, "vocab_size"):
            decode_legacy_config(payload, target_config=target | {"vocab_size": 1})
        payload["dflash_config"].pop("mask_token_id")
        with self.assertRaisesRegex(ValueError, "mask_token_id"):
            decode_legacy_config(payload, target_config=target)

    def test_mha_and_mla_preserve_projection_semantics(self):
        payload = legacy_payload()
        target = target_for(payload)
        payload["dflash_config"]["attention_mode"] = "mha"
        with self.assertRaisesRegex(ValueError, "equal"):
            decode_legacy_config(payload, target_config=target)
        payload["num_key_value_heads"] = payload["num_attention_heads"]
        node = decode_legacy_config(payload, target_config=target)
        self.assertEqual(node.attention_configs["full_attention"].type, "mha")
        payload["dflash_config"]["attention_mode"] = "MLA"
        payload.update(
            q_lora_rank=None,
            kv_lora_rank=16,
            qk_nope_head_dim=16,
            qk_rope_head_dim=8,
            v_head_dim=8,
        )
        node = decode_legacy_config(payload, target_config=target)
        attention = node.attention_configs["full_attention"]
        self.assertEqual(attention.type, "mla")
        self.assertEqual(attention.qk_head_dim, 24)
        self.assertTrue(attention.rope_interleave)
        self.assertFalse(hasattr(attention, "num_key_value_heads"))
        self.assertEqual(decode_config(encode_config(node)), node)

    def test_dflash2_extracts_convolution_and_selector_separately(self):
        payload = json.loads((CONFIGS / "qwen3.8-27b-dflash2.json").read_text())
        node = decode_legacy_config(payload, target_config=target_for(payload))
        self.assertEqual(node.conv_configs["default"].kernel_size, 2)
        self.assertEqual(node.conv_configs["default"].group_size, 16)
        self.assertTrue(
            all(
                layer.sublayer_conv == "default"
                for layer in node.layer_configs.values()
            )
        )
        self.assertEqual(node.spec_config.selector_rank, 256)
        self.assertEqual(node.spec_config.selector_top_k, 16)
        self.assertEqual(node.spec_config.output_multiplier, 1)
        self.assertIsNone(node.spec_config.final_logit_softcapping)

    def test_unknown_or_wrong_architecture_fields_do_not_disappear(self):
        for field in (
            "typo",
            "training",
            "initialization",
            "qk_norm",
            "target_bindings",
        ):
            payload = legacy_payload() | {field: True}
            with self.assertRaisesRegex(ValueError, field):
                decode_legacy_config(payload, target_config=target_for(payload))
        for field in ("selector_rank", "loss_weight", "conv_kernel_size", "typo"):
            payload = legacy_payload()
            payload["dflash_config"][field] = 1
            with self.assertRaisesRegex(ValueError, field):
                decode_legacy_config(payload, target_config=target_for(payload))
        for architectures in (None, [], ["DSpineDraftModel"], [["DFlashDraftModel"]]):
            payload = legacy_payload() | {"architectures": architectures}
            with self.assertRaisesRegex(ValueError, "architectures"):
                decode_legacy_config(payload, target_config=target_for(payload))

    def test_legacy_codec_imports_no_compute_or_hf_packages(self):
        code = """
import sys
from tests.test_config.test_draft_legacy_dflash import legacy_payload, target_for
from specforge.config.draft.codec import decode_legacy_config
payload = legacy_payload()
decode_legacy_config(payload, target_config=target_for(payload))
assert not any(name in sys.modules for name in ('torch', 'transformers', 'pydantic'))
"""
        result = subprocess.run(
            [sys.executable, "-S", "-c", code], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
