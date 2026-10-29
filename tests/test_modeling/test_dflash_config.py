"""DFlash-family configuration finalization precedes model allocation."""

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.config import Config
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.config import (
    AutoDraftConfig,
    DFlash2Config,
    DFlashConfig,
    DominoConfig,
    DSparkConfig,
)
from specforge.training.model_loading import resolve_draft_config


def payload(architecture="DFlashDraftModel", **method):
    return {
        "architectures": [architecture],
        "model_type": "qwen3",
        "vocab_size": 64,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "num_target_layers": 8,
        "block_size": 4,
        "dflash_config": method,
    }


VARIANTS = (
    ("DFlashDraftModel", DFlashConfig, {}),
    (
        "DFlash2DraftModel",
        DFlash2Config,
        {
            "conv_kernel_size": 2,
            "conv_group_size": 4,
            "selector_rank": 4,
            "selector_top_k": 8,
        },
    ),
    ("DominoDraftModel", DominoConfig, {"emb_dim": 8, "gru_hidden_dim": 8}),
    (
        "DSparkDraftModel",
        DSparkConfig,
        {"markov_rank": 4, "confidence_head_alpha": 1.0},
    ),
)


class DFlashConfigTest(unittest.TestCase):
    def test_attention_dimensions_are_validated_before_model_allocation(self):
        mla = {
            "kv_lora_rank": 8,
            "q_lora_rank": 4,
            "qk_nope_head_dim": 4,
            "qk_rope_head_dim": 4,
            "v_head_dim": 4,
        }
        cases = [
            ("gqa", "num_attention_heads", 0),
            ("gqa", "num_key_value_heads", 3),
            ("gqa", "head_dim", 3),
            ("gqa", "hidden_size", 0),
            ("gqa", "intermediate_size", 0),
        ]
        for name in ("num_attention_heads", "num_key_value_heads", "head_dim"):
            cases.extend(("gqa", name, value) for value in (True, 2.5, "4"))
        for name in mla:
            cases.extend(("mla", name, value) for value in (True, 2.5, "4"))
        for mode, name, value in cases:
            with self.subTest(mode=mode, name=name, value=value):
                config = AutoDraftConfig.from_dict(payload(attention_mode=mode))
                if mode == "mla":
                    for key, dimension in mla.items():
                        setattr(config, key, dimension)
                # New HF releases validate assignments; bypass that layer to
                # exercise our contract for legacy HF configs as well.
                config.__dict__[name] = value
                with mock.patch(
                    "torch.nn.Linear", side_effect=AssertionError("model allocation")
                ):
                    with self.assertRaisesRegex(ValueError, name):
                        AutoDraftConfig.finalize(config)
                    with self.assertRaisesRegex(ValueError, name):
                        AutoDraftModel.from_config(config)

    def test_invalid_layout_and_window_have_config_errors(self):
        for layout in (None, "full_attention", [None], [[]]):
            with self.subTest(layout=layout):
                config = AutoDraftConfig.from_dict(payload())
                config.__dict__["layer_types"] = layout
                with self.assertRaisesRegex(ValueError, "layer_types"):
                    AutoDraftConfig.finalize(config)
        for window in (True, 1.5, "8"):
            with self.subTest(window=window):
                config = AutoDraftConfig.from_dict(payload())
                config.layer_types = ["sliding_attention"]
                config.__dict__["sliding_window"] = window
                with self.assertRaisesRegex(ValueError, "sliding_window"):
                    AutoDraftConfig.finalize(config)

    def test_markov_rank_does_not_coerce_invalid_falsey_values(self):
        for rank in (False, "", 0.0):
            with self.subTest(rank=rank):
                config = AutoDraftConfig.from_dict(
                    payload("DSparkDraftModel", markov_rank=rank)
                )
                with self.assertRaisesRegex(ValueError, "markov_rank"):
                    AutoDraftConfig.finalize(config)

    def test_resolved_layout_and_direction_are_written_to_hf_config(self):
        config = AutoDraftConfig.from_dict(payload())
        options = replace(
            DFlashConfig.from_model_config(config),
            layer_types=("sliding_attention",),
            sliding_window=8,
            is_causal=True,
        )
        resolved = options.to_model_config(config)
        self.assertEqual(DFlashConfig.from_model_config(resolved), options)
        restored = AutoDraftConfig.from_dict(resolved.to_dict())
        self.assertEqual(DFlashConfig.from_model_config(restored), options)
        self.assertEqual(config.layer_types, ["full_attention"])
        self.assertIsNone(getattr(config, "is_causal", None))

    def test_variant_settings_and_weights_survive_save_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            for architecture, _, method in VARIANTS:
                with self.subTest(architecture=architecture):
                    config = AutoDraftConfig.from_dict(payload(architecture, **method))
                    model = AutoDraftModel.from_config(config)
                    path = Path(directory) / architecture
                    model.save_pretrained(path)
                    restored = AutoDraftModel.from_pretrained(
                        path, local_files_only=True
                    )
                    self.assertEqual(restored.draft_options, model.draft_options)
                    torch.testing.assert_close(
                        restored.state_dict(), model.state_dict(), rtol=0, atol=0
                    )

    def test_finalization_is_idempotent_and_preserves_unknown_fields(self):
        for architecture, options_cls, method in VARIANTS:
            with self.subTest(architecture=architecture):
                config = AutoDraftConfig.from_dict(
                    payload(architecture, **method, future_field={"x": [1]})
                )
                original = copy.deepcopy(config.to_dict())
                resolved = AutoDraftConfig.finalize(config)
                self.assertEqual(config.to_dict(), original)
                self.assertEqual(
                    AutoDraftConfig.finalize(resolved).to_dict(), resolved.to_dict()
                )
                self.assertEqual(resolved.dflash_config["target_layer_ids"], [4])
                self.assertEqual(resolved.dflash_config["future_field"], {"x": [1]})
                model = AutoDraftModel.from_config(config)
                self.assertIsInstance(model.draft_options, options_cls)
                self.assertEqual(model.config.dflash_config, resolved.dflash_config)
                self.assertEqual(config.to_dict(), original)

    def test_explicit_capture_layers_do_not_require_target_depth(self):
        data = payload(target_layer_ids=[1, 3])
        del data["num_target_layers"]
        config = AutoDraftConfig.finalize(AutoDraftConfig.from_dict(data))
        model = AutoDraftModel.from_config(config)
        self.assertEqual(model.target_layer_ids, [1, 3])

    def test_block_size_alias_uses_top_level_precedence(self):
        data = payload(block_size=2)
        config = AutoDraftConfig.finalize(AutoDraftConfig.from_dict(data))
        self.assertEqual(config.block_size, 4)
        self.assertEqual(config.dflash_config["block_size"], 4)
        del data["block_size"]
        config = AutoDraftConfig.finalize(AutoDraftConfig.from_dict(data))
        self.assertEqual(config.block_size, 2)
        self.assertEqual(config.dflash_config["block_size"], 2)

    def test_invalid_variant_settings_fail_without_allocating_parameters(self):
        cases = (
            ("DFlash2DraftModel", {}, "conv_kernel_size"),
            (
                "DFlash2DraftModel",
                {**VARIANTS[1][2], "conv_kernel_size": 5},
                "block_size",
            ),
            (
                "DFlash2DraftModel",
                {**VARIANTS[1][2], "conv_group_size": 3},
                "hidden_size",
            ),
            (
                "DFlash2DraftModel",
                {**VARIANTS[1][2], "selector_top_k": 65},
                "vocab_size",
            ),
            (
                "DFlash2DraftModel",
                {**VARIANTS[1][2], "final_logit_softcapping": 0},
                "softcapping",
            ),
            ("DominoDraftModel", {}, "emb_dim"),
            (
                "DominoDraftModel",
                {**VARIANTS[2][2], "projector_type": "dspark"},
                "projector_type",
            ),
            ("DSparkDraftModel", {"confidence_head_with_markov": True}, "markov_rank"),
            (
                "DSparkDraftModel",
                {"markov_rank": 2, "markov_head_type": "invalid"},
                "markov_head_type",
            ),
        )
        for architecture, method, error in cases:
            with self.subTest(architecture=architecture, method=method):
                config = AutoDraftConfig.from_dict(payload(architecture, **method))
                with mock.patch(
                    "torch.nn.Linear", side_effect=AssertionError("model allocation")
                ):
                    with self.assertRaisesRegex(ValueError, error):
                        AutoDraftConfig.finalize(config)
                    with self.assertRaisesRegex(ValueError, error):
                        AutoDraftModel.from_config(config)

    def test_dspark_defaults_and_head_constraints_are_resolved_early(self):
        config = AutoDraftConfig.finalize(
            AutoDraftConfig.from_dict(
                payload("DSparkDraftModel", confidence_head_alpha=0.5)
            )
        )
        self.assertEqual(config.dflash_config["projector_type"], "dspark")
        self.assertTrue(config.dflash_config["enable_confidence_head"])
        config.num_key_value_heads = config.num_attention_heads
        with self.assertRaisesRegex(ValueError, "defaults to GQA"):
            AutoDraftConfig.finalize(config)
        config.dflash_config["attention_mode"] = "mha"
        self.assertEqual(
            AutoDraftConfig.finalize(config).dflash_config["attention_mode"], "mha"
        )

    def test_run_overrides_are_applied_before_final_validation(self):
        data = payload("DFlash2DraftModel", **{**VARIANTS[1][2], "conv_kernel_size": 6})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(data))
            cfg = Config.model_validate(
                {
                    "model": {
                        "target_model_path": "unused",
                        "draft_model_config": str(path),
                        "draft_block_size": 8,
                    },
                    "data": {"hidden_states_path": "/unused"},
                    "training": {"strategy": "dflash"},
                }
            )
            provider = (
                builtin_algorithm_registry()
                .resolve("dflash")
                .providers.model.draft_config
            )
            resolved = resolve_draft_config(cfg, provider=provider)
            self.assertEqual(resolved.block_size, 8)
            self.assertEqual(resolved.dflash_config["block_size"], 8)
            self.assertEqual(resolved.dflash_config["conv_kernel_size"], 6)
            cfg.model.draft_block_size = None
            with self.assertRaisesRegex(ValueError, "must not exceed block_size"):
                resolve_draft_config(cfg, provider=provider)

    def test_all_checked_in_family_configs_finalize_without_models(self):
        config_dir = Path(__file__).resolve().parents[2] / "configs"
        names = {name for name, _, _ in VARIANTS}
        for path in config_dir.glob("*.json"):
            data = json.loads(path.read_text())
            if data.get("architectures", [None])[0] in names:
                with self.subTest(path=path.name):
                    config = AutoDraftConfig.finalize(AutoDraftConfig.from_dict(data))
                    self.assertGreaterEqual(config.block_size, 2)


if __name__ == "__main__":
    unittest.main()
