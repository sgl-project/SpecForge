"""Serving format boundaries, exact tensor layout, and native shard replacement."""

import json
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar
from unittest.mock import patch

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
from specforge.checkpoint.formats.sglang.native import export_native
from specforge.checkpoint.tensors import TensorStore
from specforge.cli import main
from specforge.config.draft.logits import NoLogitTransformSpec
from specforge.config.draft.norm import RMSNormSpec
from specforge.config.draft.registry import _SCHEMAS, register_component_schema
from specforge.config.draft.rope import LinearRoPESpec
from specforge.export.to_sglang import export_to_sglang
from specforge.loading import read_draft
from specforge.modeling import build_draft_model
from tests.test_algorithms.test_objectives import configurations
from tests.test_checkpoint.test_loading import write_target
from tests.test_checkpoint.test_sglang_boundaries import multi_axis_config
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

        with patch.dict(_SCHEMAS):
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
                tokens = dict(bos_token_id=1, eos_token_id=[2, 3], pad_token_id=0)
                for name, value in tokens.items():
                    setattr(model.config, name, value)
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
                self.assertEqual({name: fields[name] for name in tokens}, tokens)
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
        with self.assertRaisesRegex(ValueError, "mixed causality"):
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
        with self.assertRaisesRegex(ValueError, "default or multi-axis"):
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
        config = multi_axis_config(config, interleaved=True)
        for copy in (False, True):
            with self.subTest(copy=copy), TemporaryDirectory() as tmp:
                root = Path(tmp)
                target = root / "target"
                tables = write_target(target)
                # Target computation fields are explicit and independent of the converter.
                payload = dict(
                    model_type="qwen3_5_text",
                    hidden_act="silu",
                    attn_output_gate=False,
                    hidden_size=32,
                    vocab_size=64,
                    intermediate_size=64,
                    rms_norm_eps=1e-6,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=8,
                    max_position_embeddings=256,
                    rope_parameters=dict(
                        rope_type="default",
                        rope_theta=10000,
                        partial_rotary_factor=1,
                        mrope_section=[2, 1, 1],
                        mrope_interleaved=True,
                    ),
                )
                (target / "config.json").write_text(json.dumps(payload))
                (target / "model.safetensors").unlink()
                save_file(tables, target / "tables.safetensors")
                model = build_draft_model(config)
                state = serving_weights(config, model.state_dict())
                old_native = {
                    "mtp." + name: torch.zeros_like(value)
                    for name, value in state.items()
                }
                save_file(
                    {"target.weight": torch.ones(2), **old_native},
                    target / "mixed.safetensors",
                )
                (target / "chat_template.jinja").write_text("{{ messages }}")
                (target / "chat_templates").mkdir()
                (target / "chat_templates/tool.jinja").write_text("{{ tools }}")
                index = {name: "tables.safetensors" for name in tables} | {
                    "target.weight": "mixed.safetensors",
                    **{name: "mixed.safetensors" for name in old_native},
                }
                (target / "model.safetensors.index.json").write_text(
                    json.dumps({"weight_map": index})
                )
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
                self.assertEqual(
                    {k: v for k, v in exported.items() if k != "specforge_artifact"},
                    payload,
                )
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
                for shard in (root / "served").glob("target-*.safetensors"):
                    self.assertFalse(
                        any(name.startswith("mtp.") for name in load_file(shard))
                    )
                self.assertEqual(
                    (root / "served/chat_template.jinja").read_text(), "{{ messages }}"
                )
                self.assertEqual(
                    (root / "served/chat_templates/tool.jinja").read_text(),
                    "{{ tools }}",
                )
                self.assertEqual(
                    set(load_file(target / "mixed.safetensors")),
                    {"target.weight", *old_native},
                )
                if not copy:
                    for path in (
                        ("attn_output_gate",),
                        ("hidden_act",),
                        ("rms_norm_eps",),
                        ("rope_parameters", "rope_theta"),
                        ("rope_parameters", "partial_rotary_factor"),
                    ):
                        changed = json.loads(json.dumps(payload))
                        parent = changed if len(path) == 1 else changed[path[0]]
                        del parent[path[-1]]
                        (target / "config.json").write_text(json.dumps(changed))
                        with (
                            self.subTest(missing=path),
                            patch.object(
                                TensorStore,
                                "read",
                                side_effect=AssertionError("read before preflight"),
                            ),
                            self.assertRaisesRegex(ValueError, path[-1]),
                        ):
                            export_native(
                                config,
                                convert_config(config),
                                state,
                                target,
                                root / "bad",
                            )
                    (target / "config.json").write_text(json.dumps(payload))
                    for changed, error in (
                        (
                            {k: v for k, v in state.items() if k != "fc.weight"},
                            "coverage",
                        ),
                        (state | {"renamed.weight": state["fc.weight"]}, "coverage"),
                        (state | {"fc.weight": torch.zeros(1)}, "shape"),
                    ):
                        with patch.object(
                            TensorStore,
                            "read",
                            side_effect=AssertionError("read before preflight"),
                        ):
                            with self.assertRaisesRegex(ValueError, error):
                                export_native(
                                    config,
                                    convert_config(config),
                                    changed,
                                    target,
                                    root / "bad",
                                )
                    for quantized in (
                        tables
                        | {
                            "model.embed_tokens.weight": tables[
                                "model.embed_tokens.weight"
                            ].to(torch.int8)
                        },
                        tables | {"lm_head.weight_scale": torch.ones(1)},
                    ):
                        save_file(quantized, target / "tables.safetensors")
                        quant_index = index | {
                            name: "tables.safetensors" for name in quantized
                        }
                        (target / "model.safetensors.index.json").write_text(
                            json.dumps({"weight_map": quant_index})
                        )
                        with patch.object(
                            TensorStore,
                            "read",
                            side_effect=AssertionError("read before preflight"),
                        ):
                            with self.assertRaisesRegex(ValueError, "quantized"):
                                export_native(
                                    config,
                                    convert_config(config),
                                    state,
                                    target,
                                    root / "bad",
                                )
                    save_file(tables, target / "tables.safetensors")
                    (target / "model.safetensors.index.json").write_text(
                        json.dumps({"weight_map": index})
                    )
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
