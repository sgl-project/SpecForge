"""Serving format boundaries, exact tensor layout, and native shard replacement."""

import json
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import torch
from safetensors.torch import load_file, save_file

from specforge.checkpoint import save_draft
from specforge.checkpoint.formats.sglang import (
    _CONVERTERS,
    SGLANG_COMMIT,
    artifact_metadata,
    convert_config,
)
from specforge.checkpoint.formats.sglang.converters import serving_weights
from specforge.checkpoint.tensors import TensorStore
from specforge.cli import main
from specforge.config.draft.logits import NoLogitTransformSpec
from specforge.config.draft.norm import RMSNormSpec
from specforge.config.draft.registry import register_component_schema
from specforge.config.draft.rope import LinearRoPESpec
from specforge.export.to_sglang import export_to_sglang
from specforge.loading import read_draft
from specforge.modeling import build_draft_model
from tests.test_algorithms.test_objectives import configurations
from tests.test_checkpoint.test_loading import write_target
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_mtp_spec import mtp_config
from tests.test_modeling.test_canonical_dflash2 import dflash2_config
from tests.test_modeling.test_canonical_domino import model_config as domino_config
from tests.test_modeling.test_canonical_dspark import model_config as dspark_config
from tests.test_modeling.test_canonical_eagle3 import model_config as eagle_config


def serving_configs():
    return [
        draft_config(),
        dflash2_config(bound=True),
        domino_config(bound=True),
        dspark_config(bound=True),
        eagle_config(bound=True),
    ]


class SGLangExportTest(unittest.TestCase):
    def test_registered_logit_transform_requires_serving_support(self):
        @dataclass(frozen=True, kw_only=True)
        class CustomLogits(NoLogitTransformSpec):
            type: ClassVar[str] = "test_serving_logits"

        register_component_schema("logits", CustomLogits)
        config = replace(draft_config(), target_logit_transform=CustomLogits())
        with self.assertRaisesRegex(ValueError, "target logit transform"):
            convert_config(config)

    def test_registry_and_artifact_boundary(self):
        workflow = Path(__file__).resolve().parents[2] / ".github/workflows/test.yaml"
        self.assertIn("SGLANG_COMMIT: " + SGLANG_COMMIT, workflow.read_text())
        self.assertEqual(set(_CONVERTERS), {c.architecture for c in configurations()})
        for config in serving_configs():
            with (
                self.subTest(architecture=config.architecture),
                TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                model = build_draft_model(config)
                if hasattr(model, "vocabulary"):
                    model.vocabulary.load_mapping(
                        t2d=torch.arange(config.vocab_size) < config.draft_vocab_size,
                        d2t=torch.zeros(config.draft_vocab_size, dtype=torch.long),
                    )
                save_draft(model, root / "draft")
                before = model.config.to_dict()
                self.assertEqual(
                    main(
                        [
                            "export",
                            "--converter",
                            "sglang",
                            "--checkpoint",
                            str(root / "draft"),
                            "--output-dir",
                            str(root / "serving"),
                        ]
                    ),
                    0,
                )
                with self.assertRaisesRegex(ValueError, "serving-only"):
                    read_draft(root / "serving")
                fields = json.loads((root / "serving" / "config.json").read_text())
                self.assertEqual(fields["specforge_artifact"], artifact_metadata())
                self.assertFalse((root / "serving" / "artifact.json").exists())
                from transformers import AutoConfig

                hf = AutoConfig.from_pretrained(root / "serving")
                self.assertEqual(hf.num_hidden_layers, config.num_hidden_layers)
                actual = load_file(root / "serving" / "model.safetensors")
                expected = serving_weights(config, model.state_dict())
                self.assertEqual(actual.keys(), expected.keys())
                for name in expected:
                    self.assertTrue(torch.equal(actual[name], expected[name]), name)
                self.assertEqual(model.config.to_dict(), before)
                if config.spec_config.protocol:
                    self.assertEqual(
                        fields["block_size"],
                        config.spec_config.protocol.num_model_slots,
                    )

    def test_unrepresentable_structures_fail(self):
        base = draft_config()
        a = base.attention_configs["full"]
        for config in [
            replace(base, target_bindings=(), target_logit_transform=None),
            replace(base, attention_configs={"full": replace(a, qk_norm=False)}),
            replace(base, attention_configs={"full": replace(a, output_gate=True)}),
            dspark_config(bound=True, confidence=False),
        ]:
            with (
                self.subTest(config=config),
                self.assertRaisesRegex(ValueError, "SGLang export"),
            ):
                convert_config(config)
        from specforge.config.draft.layer import LayerSpec

        mixed = replace(
            base,
            attention_configs={"full": a, "causal": replace(a, is_causal=True)},
            layer_configs={
                **base.layer_configs,
                "causal": LayerSpec(attention="causal", mlp="dense"),
            },
            layer_types=("full_dense", "causal"),
        )
        with self.assertRaisesRegex(ValueError, "per-layer"):
            convert_config(mixed)
        for config in configurations():
            if config.architecture in {
                "DSpineDraftModel",
                "DMTPDraftModel",
                "PEagleDraftModel",
            }:
                with self.assertRaisesRegex(ValueError, "no serving implementation"):
                    convert_config(config)

    def test_qk_norm_serving_boundary(self):
        for config in serving_configs():
            for changes in (
                {"qk_norm_position": "after_rope"},
                {"qk_norm_scope": "projection"},
                {"qk_norm_config": replace(config.norm_config, has_weight=False)},
            ):
                attentions = {
                    name: replace(node, qk_norm=True, **changes)
                    for name, node in config.attention_configs.items()
                }
                with (
                    self.subTest(architecture=config.architecture, changes=changes),
                    self.assertRaisesRegex(ValueError, "qk_norm|QK normalization"),
                ):
                    convert_config(replace(config, attention_configs=attentions))
            with self.assertRaisesRegex(ValueError, "weightless"):
                convert_config(
                    replace(
                        config,
                        norm_config=replace(config.norm_config, has_weight=False),
                    )
                )

    def test_native_merge_rewrites_stale_shards_and_preserves_target(self):
        config = replace(
            mtp_config(),
            norm_config=RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=1),
        )
        with self.assertRaisesRegex(ValueError, "default RoPE only"):
            convert_config(
                replace(
                    config,
                    attention_configs={
                        "full": replace(
                            config.attention_configs["full"],
                            rope_parameters=LinearRoPESpec(
                                interleaved=False,
                                rope_theta=10000,
                                partial_rotary_factor=1,
                                factor=2,
                            ),
                        )
                    },
                )
            )
        for copy in (False, True):
            with self.subTest(copy=copy), TemporaryDirectory() as tmp:
                root = Path(tmp)
                target = root / "target"
                tables = write_target(target)
                payload = convert_config(config) | {"model_type": "qwen3_5_text"}
                (target / "config.json").write_text(json.dumps(payload))
                (target / "model.safetensors").unlink()
                save_file(tables, target / "tables.safetensors")
                save_file(
                    {"target.weight": torch.ones(2), "mtp.obsolete": torch.zeros(3)},
                    target / "mixed.safetensors",
                )
                index = {name: "tables.safetensors" for name in tables} | {
                    "target.weight": "mixed.safetensors",
                    "mtp.obsolete": "mixed.safetensors",
                }
                (target / "model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": index})
                )
                model = build_draft_model(config)
                save_draft(model, root / "draft")
                export_to_sglang(
                    str(root / "draft"),
                    None,
                    str(root / "served"),
                    target_model=str(target),
                    copy_target=copy,
                )
                restored = TensorStore(root / "served")
                exported = json.loads((root / "served" / "config.json").read_text())
                self.assertEqual(exported["specforge_artifact"], artifact_metadata())
                self.assertFalse((root / "served" / "artifact.json").exists())
                with self.assertRaisesRegex(ValueError, "serving-only"):
                    read_draft(root / "served")
                for key, tensor in serving_weights(config, model.state_dict()).items():
                    self.assertTrue(torch.equal(restored.read("mtp." + key), tensor))
                self.assertTrue(
                    torch.equal(
                        restored.read("mtp.embed_tokens.weight"),
                        tables["model.embed_tokens.weight"],
                    )
                )
                self.assertEqual(
                    restored.files["lm_head.weight"].is_symlink(), not copy
                )
                for shard in (root / "served").glob("*.safetensors"):
                    self.assertNotIn("mtp.obsolete", load_file(shard))
                self.assertIn("mtp.obsolete", load_file(target / "mixed.safetensors"))
                payload["hidden_size"] += 1
                (target / "config.json").write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    export_to_sglang(
                        str(root / "draft"),
                        None,
                        str(root / "bad"),
                        target_model=str(target),
                    )
                self.assertFalse((root / "bad").exists())
