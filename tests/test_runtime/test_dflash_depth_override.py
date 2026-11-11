"""Depth overrides preserve explicit capture choices unless the depth changes."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from transformers import Qwen3Config

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.model_providers import apply_dflash_overrides
from specforge.config import Config
from specforge.training.model_loading import resolve_draft_config


def draft_config(**kwargs):
    return Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        dflash_config={"target_layer_ids": [1, 3]},
        **kwargs,
    )


def override(config, depth, block_size=None):
    config.num_hidden_layers = depth
    apply_dflash_overrides(
        SimpleNamespace(
            model=SimpleNamespace(
                draft_num_hidden_layers=depth, draft_block_size=block_size
            )
        ),
        config,
    )


class DFlashDepthOverrideTest(unittest.TestCase):
    def test_same_depth_preserves_capture_layers_without_target_depth(self):
        for extra in ({}, {"num_target_layers": 8}):
            with self.subTest(extra=extra):
                config = draft_config(**extra)
                override(config, 1, block_size=8)
                self.assertEqual(config.dflash_config["target_layer_ids"], [1, 3])
                self.assertEqual(config.dflash_config["block_size"], 8)

    def test_resize_rebuilds_uniform_layout_and_capture_layers(self):
        config = draft_config(num_target_layers=8)
        override(config, 2)
        self.assertEqual(config.layer_types, ["full_attention"] * 2)
        self.assertEqual(config.dflash_config["target_layer_ids"], [1, 5])

    def test_resize_requires_valid_target_depth(self):
        for depth in (None, True, 0, "8"):
            with self.subTest(depth=depth):
                config = draft_config(num_target_layers=depth)
                with self.assertRaisesRegex(ValueError, "num_target_layers"):
                    override(config, 2)

    def test_resize_rejects_mixed_or_malformed_layout(self):
        for layout, message in [
            ([], "non-empty"),
            ([None], "non-empty"),
            (["full_attention", "sliding_attention"], "mixed"),
        ]:
            with self.subTest(layout=layout):
                config = draft_config(num_target_layers=8)
                config.__dict__["layer_types"] = layout
                with self.assertRaisesRegex(ValueError, message):
                    override(config, 3)

    def test_same_depth_still_validates_attention(self):
        config = draft_config()
        config.layer_types = ["linear_attention"]
        data = config.to_dict()
        data.update(architectures=["DFlashDraftModel"], block_size=4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(data))
            cfg = Config.model_validate(
                {
                    "model": {
                        "target_model_path": "unused",
                        "draft_model_config": str(path),
                        "draft_num_hidden_layers": 1,
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
            with self.assertRaisesRegex(ValueError, "layer_types"):
                resolve_draft_config(cfg, provider=provider)
