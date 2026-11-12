"""Native presets are ordinary YAML; target copying has no family dispatch."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import yaml
from safetensors.torch import save_file

from specforge.checkpoint import save_draft
from specforge.checkpoint.formats.native.qwen3_5 import (
    convert_config as convert_qwen3_5_config,
)
from specforge.checkpoint.formats.sglang.converters import serving_weights
from specforge.checkpoint.tensors import TensorStore
from specforge.config.draft.codec import parse_draft
from specforge.export.to_sglang import export_to_sglang
from specforge.loading import initialize_draft
from specforge.modeling import build_draft_model
from tests.test_config.test_draft_legacy_mtp import mtp_payload

PRESETS = Path(__file__).resolve().parents[2] / "specforge/config/draft/presets/native"
NATIVE_MAPPING = {
    "": {
        "from": "mtp",
        "rename": {
            "layers": "backbone.layers",
            "norm": "backbone.final_norm",
            "fc": "fusion.fc",
            "pre_fc_norm_embedding": "fusion.pre_fc_norm_embedding",
            "pre_fc_norm_hidden": "fusion.pre_fc_norm_hidden",
        },
    }
}


def native_configs(*, small=False):
    for path in sorted(PRESETS.glob("*.yaml")):
        config = parse_draft(yaml.safe_load(path.read_text()), target_config=None)
        if small:
            # Keep native head geometry, gates, normalization and RoPE unchanged.
            config = replace(
                config,
                hidden_size=32,
                vocab_size=64,
                draft_vocab_size=64,
                max_position_embeddings=256,
                attention_configs={
                    name: replace(a, num_attention_heads=4, num_key_value_heads=2)
                    for name, a in config.attention_configs.items()
                },
                mlp_configs={
                    name: replace(m, intermediate_size=64)
                    for name, m in config.mlp_configs.items()
                },
                spec_config=replace(
                    config.spec_config,
                    target_features=replace(
                        config.spec_config.target_features, hidden_size=32
                    ),
                ),
            )
        yield config


class NativePresetTest(unittest.TestCase):
    def test_presets_build_on_meta_and_copy_export_copy_is_exact(self):
        self.assertTrue(list(PRESETS.glob("*.yaml")))
        for config in native_configs():
            model = build_draft_model(config, device="meta")
            self.assertEqual(
                model.backbone.layers[0].self_attn.q_proj.weight.shape, (8192, 2560)
            )
        for config in native_configs(small=True):
            with TemporaryDirectory() as tmp:
                root = Path(tmp)
                target = root / "target"
                target.mkdir()
                original = build_draft_model(config)
                weights = {
                    "mtp." + name: value
                    for name, value in serving_weights(
                        config, original.state_dict()
                    ).items()
                }
                weights["model.language_model.embed_tokens.weight"] = torch.randn(
                    64, 32
                )
                payload = convert_qwen3_5_config(config) | {
                    "model_type": "qwen3_5_text",
                    "tie_word_embeddings": True,
                }
                (target / "config.json").write_text(
                    json.dumps({"text_config": payload})
                )
                save_file(weights, target / "model.safetensors")
                copied = initialize_draft(
                    config, target=target, from_target=NATIVE_MAPPING
                )
                torch.testing.assert_close(
                    copied.state_dict(), original.state_dict(), rtol=0, atol=0
                )
                self.assertFalse(copied.embed_tokens.weight.requires_grad)
                self.assertTrue(copied.fusion.fc.weight.requires_grad)
                save_draft(copied, root / "draft")
                export_to_sglang(
                    str(root / "draft"),
                    None,
                    str(root / "export"),
                    target_model=str(target),
                )
                restored = initialize_draft(
                    config, target=root / "export", from_target=NATIVE_MAPPING
                )
                torch.testing.assert_close(
                    restored.state_dict(), copied.state_dict(), rtol=0, atol=0
                )
                legacy = root / "legacy"
                legacy.mkdir()
                raw = mtp_payload() | {
                    name: payload[name]
                    for name in (
                        "hidden_size",
                        "vocab_size",
                        "intermediate_size",
                        "num_attention_heads",
                        "num_key_value_heads",
                        "head_dim",
                        "max_position_embeddings",
                        "partial_rotary_factor",
                        "rope_theta",
                    )
                }
                (legacy / "config.json").write_text(json.dumps(raw))
                save_file(
                    {k: v for k, v in weights.items() if k.startswith("mtp.")},
                    legacy / "model.safetensors",
                )
                export_to_sglang(
                    str(legacy),
                    None,
                    str(root / "legacy-export"),
                    target_model=str(target),
                )
                exported = json.loads((root / "legacy-export/config.json").read_text())
                self.assertEqual(
                    exported["text_config"]["rope_parameters"],
                    payload["rope_parameters"],
                )
                # Extra, quantized and incomplete native tensors fail before data reads.
                for bad in (
                    weights | {"mtp.scale": torch.ones(1)},
                    {k: v for k, v in weights.items() if k != "mtp.fc.weight"},
                    weights
                    | {"mtp.fc.weight": weights["mtp.fc.weight"].to(torch.int8)},
                ):
                    save_file(bad, target / "model.safetensors")
                    with (
                        self.assertRaises(ValueError),
                        patch.object(TensorStore, "read", side_effect=AssertionError),
                    ):
                        initialize_draft(
                            config, target=target, from_target=NATIVE_MAPPING
                        )


if __name__ == "__main__":
    unittest.main()
