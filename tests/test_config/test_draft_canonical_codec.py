"""Canonical configuration round-trips independently of files and model code."""

import copy
import dataclasses
import json
import subprocess
import sys
import unittest

import yaml

from specforge.config.draft.architecture import DFlash2Spec
from specforge.config.draft.attention import MLASpec
from specforge.config.draft.codec import decode_config, encode_config, parse_draft
from specforge.config.draft.conv import GroupedCausalConvSpec
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.logits import MuPLogitTransformSpec
from specforge.config.draft.mlp import DenseFFNSpec
from tests.test_config.draft_fixtures import draft_config


def shorthand():
    payload = encode_config(draft_config())
    for name in (
        "schema_version",
        "model_type",
        "extensions",
        "initializer_range",
        "draft_vocab_size",
    ):
        payload.pop(name)
    payload["attention"] = payload.pop("attention_configs")["full"]
    payload["mlp"] = payload.pop("mlp_configs")["dense"]
    for name in ("conv_configs", "layer_configs", "layer_types"):
        payload.pop(name)
    return payload


class CanonicalCodecTest(unittest.TestCase):
    def test_config_and_canonical_json_round_trip(self):
        node = draft_config()
        before = hash(node)
        encoded = encode_config(node)
        source = json.loads(json.dumps(encoded, allow_nan=False))
        restored = decode_config(source)
        self.assertEqual(restored, node)
        self.assertEqual(hash(restored), before)
        self.assertEqual(encode_config(restored), source)
        self.assertEqual(hash(node), before)
        self.assertNotIn("architecture", encoded)
        self.assertEqual(encoded["num_hidden_layers"], 2)
        self.assertEqual(
            encoded["spec_config"]["target_features"]["target_layer_ids"], [1, 3]
        )

    def test_decode_snapshots_the_source_and_encode_returns_fresh_values(self):
        payload = encode_config(draft_config())
        original = copy.deepcopy(payload)
        node = decode_config(payload)
        self.assertEqual(payload, original)
        payload["spec_config"]["target_features"]["target_layer_ids"].append(9)
        payload["attention_configs"]["full"]["head_dim"] = 32
        encoded = encode_config(node)
        encoded["layer_types"].clear()
        self.assertEqual(encode_config(node), original)

    def test_heterogeneous_attention_and_mlp_definitions_round_trip(self):
        base = draft_config()
        attention = base.attention_configs["full"]
        node = dataclasses.replace(
            base,
            attention_configs={
                "full": attention,
                "local": dataclasses.replace(
                    attention, is_causal=True, sliding_window=128
                ),
                "latent": MLASpec(
                    rope_parameters=None,
                    num_attention_heads=4,
                    q_lora_rank=None,
                    kv_lora_rank=16,
                    qk_nope_head_dim=8,
                    qk_rope_head_dim=8,
                    v_head_dim=8,
                    rope_interleave=True,
                    attention_bias=False,
                    is_causal=False,
                    sliding_window=None,
                    attention_dropout=0.0,
                ),
            },
            mlp_configs={
                **base.mlp_configs,
                "wide": DenseFFNSpec(intermediate_size=128, hidden_act="silu"),
            },
            layer_configs={
                "full": LayerSpec(attention="full", mlp="dense"),
                "local": LayerSpec(attention="local", mlp="dense"),
                "latent": LayerSpec(attention="latent", mlp="wide"),
            },
            layer_types=("local", "latent", "full", "local"),
            target_logit_transform=MuPLogitTransformSpec(multiplier=8),
        )
        payload = encode_config(node)
        self.assertEqual(payload["num_hidden_layers"], 4)
        self.assertEqual(decode_config(json.loads(json.dumps(payload))), node)
        payload["layer_types"][1] = "missing"
        with self.assertRaisesRegex(ValueError, "unknown layer"):
            decode_config(payload)

    def test_yaml_shorthand_matches_its_expanded_configuration(self):
        payload = shorthand()
        before = copy.deepcopy(payload)
        parsed_yaml = yaml.safe_load(yaml.safe_dump({"draft": payload}))
        node = parse_draft(parsed_yaml["draft"], target_config=None)
        self.assertEqual(node.num_hidden_layers, 2)
        self.assertEqual(node.layer_types, ("default", "default"))
        self.assertEqual(node.draft_vocab_size, node.vocab_size)
        self.assertEqual(node.initializer_range, 0.02)
        self.assertEqual(parse_draft(encode_config(node), target_config=None), node)
        self.assertEqual(decode_config(encode_config(node)), node)
        self.assertEqual(payload, before)

    def test_shorthand_never_mixes_with_named_definitions(self):
        for name in (
            "attention_configs",
            "mlp_configs",
            "conv_configs",
            "layer_configs",
            "layer_types",
        ):
            with self.assertRaisesRegex(ValueError, "cannot mix shorthand"):
                parse_draft({**shorthand(), name: {}}, target_config=None)
        for name in ("attention", "mlp", "num_hidden_layers"):
            payload = shorthand()
            payload.pop(name)
            with self.assertRaisesRegex(ValueError, name):
                parse_draft(payload, target_config=None)

    def test_dflash2_round_trip_with_convolution_and_selector(self):
        source = draft_config()
        spec = DFlash2Spec(
            block=source.spec_config.block,
            target_features=source.spec_config.target_features,
            selector_rank=8,
            selector_top_k=4,
            output_multiplier=0.5,
            final_logit_softcapping=30.0,
        )
        node = dataclasses.replace(
            source,
            architecture="DFlash2DraftModel",
            spec_config=spec,
            conv_configs={"conv": GroupedCausalConvSpec(kernel_size=3, group_size=8)},
            layer_configs={
                "full_dense": LayerSpec(
                    attention="full", mlp="dense", sublayer_conv="conv"
                )
            },
        )
        self.assertEqual(
            decode_config(json.loads(json.dumps(encode_config(node)))), node
        )
        compact = shorthand()
        compact["architectures"] = ["DFlash2DraftModel"]
        compact["spec_config"] = encode_config(node)["spec_config"]
        compact["sublayer_conv"] = {
            "type": "grouped_causal",
            "kernel_size": 3,
            "group_size": 8,
        }
        self.assertEqual(parse_draft(compact, target_config=None).spec_config, spec)

    def test_version_architecture_and_depth_are_checked(self):
        for name, values in (
            ("schema_version", (None, True, 1.0, 2)),
            ("model_type", (None, "qwen3")),
            ("architectures", (None, [], ["a", "b"], "DFlashDraftModel", [True])),
            ("num_hidden_layers", (None, True, 1.0, 0, 3)),
        ):
            for value in values:
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    decode_config({**encode_config(draft_config()), name: value})
        with self.assertRaisesRegex(ValueError, "architecture"):
            decode_config(
                {**encode_config(draft_config()), "architecture": "DFlash2DraftModel"}
            )

    def test_missing_and_unknown_fields_are_not_hf_defaults(self):
        for name in (
            "hidden_size",
            "norm_config",
            "rope_parameters",
            "spec_config",
            "layer_types",
            "initializer_range",
            "draft_vocab_size",
        ):
            payload = encode_config(draft_config())
            payload.pop(name)
            with self.assertRaisesRegex(ValueError, name):
                decode_config(payload)
        payload = encode_config(draft_config())
        payload["attention_configs"]["full"]["typo"] = 1
        with self.assertRaisesRegex(ValueError, "attention_configs.full.*typo"):
            decode_config(payload)
        for name in ("dflash_config", "training", "initialization", "hf_typo"):
            with self.assertRaisesRegex(ValueError, name):
                decode_config({**encode_config(draft_config()), name: {}})
        with self.assertRaisesRegex(ValueError, "draft_vocab_size"):
            parse_draft({**shorthand(), "draft_vocab_size": None}, target_config=None)

    def test_nested_architecture_nodes_are_strict(self):
        for field, value in (("block_size", True), ("block_size", "8"), ("extra", 1)):
            payload = encode_config(draft_config())
            payload["spec_config"]["block"][field] = value
            with self.assertRaisesRegex(ValueError, field):
                decode_config(payload)
        payload = encode_config(draft_config())
        payload["spec_config"]["target_features"]["target_layer_ids"] = [True]
        with self.assertRaisesRegex(ValueError, "target_layer_ids"):
            decode_config(payload)

    def test_bound_head_settings_are_required_and_unbound_settings_forbidden(self):
        payload = encode_config(draft_config())
        payload.pop("target_logit_transform")
        with self.assertRaisesRegex(ValueError, "target_logit_transform"):
            decode_config(payload)
        node = draft_config(target_bindings=(), target_logit_transform=None)
        payload = encode_config(node)
        self.assertNotIn("target_logit_transform", payload)
        self.assertEqual(decode_config(payload), node)
        with self.assertRaisesRegex(ValueError, "requires lm_head"):
            decode_config({**payload, "target_logit_transform": None})

    def test_unsupported_extension_prerequisites_are_not_ignored(self):
        for extensions in (None, "package", ["package"]):
            with self.assertRaisesRegex(ValueError, "extensions"):
                decode_config(
                    {**encode_config(draft_config()), "extensions": extensions}
                )

    def test_codec_needs_no_compute_packages(self):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                """
import sys
from specforge.config.draft.codec import decode_config, encode_config
from tests.test_config.draft_fixtures import draft_config
assert decode_config(encode_config(draft_config())) == draft_config()
assert not any(x in sys.modules for x in ('torch', 'transformers', 'pydantic', 'yaml'))
""",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
