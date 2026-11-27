"""Tests for multimodal tie_word_embeddings resolution and native writer dtype validation."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from specforge.checkpoint.formats.native.writer import export_native
from specforge.checkpoint.formats.sglang.converters import serving_weights
from specforge.checkpoint.formats.sglang.models.qwen3_5 import (
    convert_config as convert_qwen3_5_config,
)
from specforge.checkpoint.resolve import target_configuration
from specforge.checkpoint.tensors import TensorStore
from specforge.modeling import build_draft_model
from tests.test_checkpoint.test_native_presets import native_configs


def _write_multimodal_target(
    root: Path, config, model, *, outer_tie=True, inner_tie=None
):
    target = root / "target"
    target.mkdir()
    dtype = torch.bfloat16
    weights = {
        f"mtp.{k}": v.to(dtype)
        for k, v in serving_weights(config, model.state_dict()).items()
    }
    embed = torch.randn(config.vocab_size, config.hidden_size, dtype=dtype)
    weights["model.language_model.embed_tokens.weight"] = embed
    text_config = convert_qwen3_5_config(config) | {
        "model_type": "qwen3_5_text",
        "dtype": "bfloat16",
    }
    text_config.pop("tie_word_embeddings", None)
    if inner_tie is not None:
        text_config["tie_word_embeddings"] = inner_tie
    payload = {"model_type": "qwen3_5", "text_config": text_config}
    if outer_tie is not None:
        payload["tie_word_embeddings"] = outer_tie
    (target / "config.json").write_text(json.dumps(payload))
    save_file(weights, target / "model.safetensors")
    return target, embed


class NativeWriterTest(unittest.TestCase):
    def test_target_configuration_tie_word_embeddings_precedence(self):
        cases = (
            (True, None, True),
            (True, False, True),
            (False, True, False),
            (None, True, True),
            (None, False, False),
            (None, None, None),
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for outer, inner, expected in cases:
                with self.subTest(outer=outer, inner=inner):
                    text_cfg = {"model_type": "qwen3_5_text"}
                    if inner is not None:
                        text_cfg["tie_word_embeddings"] = inner
                    payload = {"model_type": "qwen3_5", "text_config": text_cfg}
                    if outer is not None:
                        payload["tie_word_embeddings"] = outer
                    (root / "config.json").write_text(json.dumps(payload))
                    self.assertIs(
                        target_configuration(root).get("tie_word_embeddings"), expected
                    )

    def test_multimodal_top_level_tie_word_embeddings_export(self):
        config = next(native_configs(small=True))
        model = build_draft_model(config)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, embed = _write_multimodal_target(
                root, config, model, outer_tie=True, inner_tie=False
            )
            state = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
            output = root / "exported"
            export_native(config, state, target, output, copy_target=True)
            exported = TensorStore(output)
            self.assertTrue(
                torch.equal(exported.read("mtp.embed_tokens.weight"), embed)
            )
            self.assertTrue(torch.equal(exported.read("mtp.lm_head.weight"), embed))
            self.assertTrue(
                all(
                    info.dtype == torch.bfloat16
                    for name, info in exported.metadata.items()
                    if name.startswith("mtp.")
                )
            )

    def test_rejects_mismatched_non_floating_and_quantized_dtypes_before_reads(self):
        config = next(native_configs(small=True))
        model = build_draft_model(config)
        bf16 = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
        fc = bf16["fusion.fc.weight"]
        cases = (
            ("fp32_into_bf16", model.state_dict(), "dtype"),
            (
                "int8",
                bf16 | {"fusion.fc.weight": fc.to(torch.int8)},
                "unquantized floating",
            ),
            (
                "fp8",
                bf16 | {"fusion.fc.weight": fc.to(torch.float8_e4m3fn)},
                "unquantized floating",
            ),
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, _ = _write_multimodal_target(
                root, config, model, outer_tie=None, inner_tie=True
            )
            for label, state, pattern in cases:
                with self.subTest(case=label):
                    out = root / f"bad-{label}"
                    with (
                        patch.object(
                            TensorStore,
                            "read",
                            side_effect=AssertionError("read before preflight"),
                        ),
                        self.assertRaisesRegex(ValueError, pattern),
                    ):
                        export_native(config, state, target, out, copy_target=True)
                    self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
