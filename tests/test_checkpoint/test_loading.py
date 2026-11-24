"""Public loading and strict initialization, with small real checkpoints."""

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from specforge.checkpoint import save_draft
from specforge.checkpoint.resolve import _ADAPTERS, register_adapter
from specforge.checkpoint.tensors import TensorStore
from specforge.checkpoint.weight_plan import WeightPlan
from specforge.loading import (
    DraftWeights,
    construct_draft,
    construct_initialized_draft,
    initialize_draft,
    load_draft,
    read_draft,
)
from specforge.modeling import build_draft_model
from tests.test_config.draft_fixtures import draft_config
from tests.test_modeling.test_canonical_dflash import inputs_for


def write_target(path, extra=None):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"hidden_size": 32, "vocab_size": 64, "num_hidden_layers": 128})
    )
    tensors = {
        "model.embed_tokens.weight": torch.randn(64, 32),
        "lm_head.weight": torch.randn(64, 32),
    } | (extra or {})
    save_file(tensors, path / "model.safetensors")
    return tensors


class LoadingTest(unittest.TestCase):

    def test_historical_checkpoint_formats_are_rejected_before_weight_reads(self):
        with TemporaryDirectory() as temp:
            path = Path(temp)
            for architecture in (
                "DFlashDraftModel",
                "LlamaForCausalLMEagle3",
                "Qwen3_5MTPDraftModel",
            ):
                with self.subTest(architecture=architecture):
                    (path / "config.json").write_text(
                        json.dumps(
                            {"model_type": "qwen3", "architectures": [architecture]}
                        )
                    )
                    with patch("specforge.checkpoint.resolve.TensorStore") as store:
                        with self.assertRaisesRegex(
                            ValueError, "canonical SpecForge checkpoint"
                        ):
                            read_draft(path)
                        store.assert_not_called()

    def test_canonical_forward_restore_and_structure_guard(self):
        with TemporaryDirectory() as tmp:
            path, target = Path(tmp) / "draft", Path(tmp) / "target"
            write_target(target)
            original = initialize_draft(draft_config(), target=target)
            args = inputs_for(original.config.draft)
            save_draft(original, path)
            # Canonical and historical DFlash share a class name, not a format.
            self.assertEqual(original.config.architectures, ["DFlashDraftModel"])
            with self.assertRaisesRegex(ValueError, "does not match"):
                read_draft(path, adapter="legacy")
            with patch("huggingface_hub.snapshot_download", return_value=str(path)):
                restored = load_draft("test/draft", target=target)
            torch.testing.assert_close(
                restored(*args).logits, original(*args).logits, rtol=0, atol=0
            )
            self.assertEqual(read_draft(path).config, original.config.draft)
            with self.assertRaisesRegex(ValueError, "structure"):
                load_draft(
                    path,
                    target=target,
                    draft=replace(original.config.draft, initializer_range=0.1),
                )
            (target / "config.json").write_text(
                json.dumps(
                    {
                        "hidden_size": 32,
                        "vocab_size": 64,
                        "num_hidden_layers": 128,
                        "logits_mup_width_multiplier": 2,
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "target_logit_transform"):
                load_draft(path, target=target)

    def test_warm_start_and_target_copy_are_explicit_and_disjoint(self):
        config = draft_config(target_bindings=(), target_logit_transform=None)
        with TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "source", Path(tmp) / "target"
            original = build_draft_model(config)
            save_draft(original, source)
            extra = {
                "teacher." + k.replace("self_attn.", "attention."): v.clone()
                for k, v in original.backbone.layers[0].state_dict().items()
            }
            write_target(target, extra)
            copy = {
                "backbone.layers.1": {
                    "from": "teacher",
                    "rename": {"attention": "self_attn"},
                }
            }
            fresh = initialize_draft(config, target=target, from_target=copy)
            torch.testing.assert_close(
                fresh.backbone.layers[1].state_dict(),
                original.backbone.layers[0].state_dict(),
                rtol=0,
                atol=0,
            )
            changed = replace(
                config,
                spec_config=replace(
                    config.spec_config,
                    block=replace(config.spec_config.block, block_size=4),
                ),
            )
            warm = initialize_draft(changed, weights_from=source)
            self.assertEqual(warm.config.draft, changed)
            torch.testing.assert_close(
                warm.state_dict(), original.state_dict(), rtol=0, atol=0
            )
            with patch.object(
                TensorStore, "read", side_effect=AssertionError("preflight read")
            ):
                for kwargs in (
                    {
                        "weights_from": source,
                        "target": target,
                        "from_target": copy,
                    },
                    {
                        "target": target,
                        "from_target": copy,
                        "dtype": torch.bfloat16,
                    },
                    {"weights_from": source, "mapping": {"fusion": "fusion"}},
                ):
                    with self.assertRaises(ValueError):
                        initialize_draft(config, **kwargs)
            partial = {
                name: name
                for name in (
                    "backbone.layers.0",
                    "backbone.final_norm",
                    "fusion",
                    "embed_tokens",
                    "lm_head",
                )
            }
            combined = initialize_draft(
                config,
                target=target,
                weights_from=source,
                mapping=partial,
                from_target=copy,
            )
            torch.testing.assert_close(
                combined.backbone.layers[1].self_attn.q_proj.weight,
                original.backbone.layers[0].self_attn.q_proj.weight,
                rtol=0,
                atol=0,
            )
            allowed = ["backbone", "embed_tokens", "lm_head"]
            initialize_draft(
                config,
                weights_from=source,
                mapping={"fusion": "fusion"},
                allow_new=allowed,
            )

    def test_format_detection_and_ambiguous_target_keys(self):
        with TemporaryDirectory() as tmp:
            path, target = Path(tmp) / "draft", Path(tmp) / "target"
            save_draft(build_draft_model(draft_config()), path)
            write_target(target, {"embed_tokens.weight": torch.randn(64, 32)})
            with self.assertRaisesRegex(ValueError, "expected one target key"):
                load_draft(path, target=target)
            register_adapter(
                "test_duplicate", lambda p: True, _ADAPTERS["canonical"][1]
            )
            try:
                with self.assertRaisesRegex(ValueError, "adapter"):
                    read_draft(path)
                self.assertEqual(
                    read_draft(path, adapter="canonical").config, draft_config()
                )
            finally:
                del _ADAPTERS["test_duplicate"]
            payload = json.loads((path / "config.json").read_text())
            payload["specforge_artifact"] = {"format": "sglang", "version": 1}
            (path / "config.json").write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "serving-only"):
                read_draft(path)

    def test_a_draft_constructed_on_meta_is_refused(self):
        with self.assertRaisesRegex(ValueError, "real device"):
            construct_initialized_draft(draft_config(), device="meta")
        with self.assertRaisesRegex(ValueError, "real device"):
            construct_draft("unread", device=torch.device("meta"))

    def test_a_reduced_vocabulary_saved_unmapped_loads_no_mapping(self):
        unmapped = torch.zeros(8, dtype=torch.bool), torch.zeros(4, dtype=torch.long)
        mapped = torch.arange(8) < 4, torch.zeros(4, dtype=torch.long)
        identity = torch.ones(4, dtype=torch.bool), torch.zeros(4, dtype=torch.long)
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.safetensors"
            for (t2d, d2t), expected in (
                (unmapped, False),
                (mapped, True),
                (identity, True),
            ):
                save_file({"vocabulary.t2d": t2d, "vocabulary.d2t": d2t}, path)
                plan = WeightPlan(tuple(sorted(TensorStore(path).refs().items())))
                weights = DraftWeights(plan, finishes_initialization=False)
                self.assertEqual(weights.loads_vocabulary_mapping, expected)
