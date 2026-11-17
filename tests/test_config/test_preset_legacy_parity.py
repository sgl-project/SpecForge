"""Each bundled preset equals the canonical decode of the legacy JSON it replaces.

The target depth of every EAGLE target is written here, not derived from the
preset, so a preset whose default taps do not fit its target fails.
"""

import json
import unittest
from pathlib import Path

import yaml

from specforge.config.draft.codec import decode_legacy_config, parse_draft

ROOT = Path(__file__).resolve().parents[2]
PRESETS = ROOT / "specforge/config/draft/presets"
LEGACY = ROOT / "configs"

# num_hidden_layers of the target each EAGLE preset was written for.
EAGLE_TARGET_LAYERS = {
    "deepseek-v2-lite-eagle3": 27,
    "deepseek-v3-671b-eagle3": 61,
    "gemma3-1b-eagle3": 26,
    "gpt-oss-120B-eagle3": 36,
    "gpt-oss-20B-eagle3": 24,
    "ling-flash-2.0-eagle3": 32,
    "llama3-70B-ealge3": 80,
    "llama3-8B-eagle3": 32,
    "llama4-scout-17B-16E-eagle3": 48,
    "longcat-flash-eagle3": 28,
    "phi4-eagle3": 40,
    "qwen2-5-vl-7b-eagle3": 28,
    "qwen2.5-0.5b-eagle3": 24,
    "qwen2.5-7b-eagle3": 28,
    "qwen2.5-vl-32b-eagle3": 64,
    "qwen3-235B-A22B-eagle3": 94,
    "qwen3-30B-A3B-eagle3.1": 48,
    "qwen3-30B-A3B-eagle3": 48,
    "qwen3-32b-eagle3": 64,
    "qwen3-4b-eagle3": 36,
    "qwen3-8b-eagle3": 36,
    "qwen3-8b-peagle": 36,
    "qwen3-coder-30B-A3B-instruct-eagle3": 48,
    "qwen3-coder-480B-A35B-instruct-eagle3": 62,
    "qwen3-coder-next-eagle3": 48,
    "qwen3-next-80b-a3b-eagle3": 48,
    "qwen3.5-35b-a3b-eagle3": 40,
    "qwq-32B-eagle3": 64,
}
# Target logit settings the codec reads from the target configuration.
TARGET_LOGITS = {"inkling-dspark": {"logits_mup_width_multiplier": 24.0}}
# Settings the legacy route took from the recipe or the tokenizer at run time.
LEGACY_SETTINGS = {
    "qwen3-8b-peagle": {"mask_token_id": 151669, "norm_before_residual": False},
}


class PresetLegacyParityTest(unittest.TestCase):
    def test_every_preset_matches_its_legacy_json(self):
        stems = sorted(path.stem for path in PRESETS.glob("*.yaml"))
        self.assertEqual(stems, sorted(path.stem for path in LEGACY.glob("*.json")))
        for stem in stems:
            with self.subTest(preset=stem):
                legacy = json.loads((LEGACY / f"{stem}.json").read_text())
                target = {
                    "hidden_size": legacy["hidden_size"],
                    "vocab_size": legacy["vocab_size"],
                    "num_hidden_layers": legacy.get(
                        "num_target_layers", EAGLE_TARGET_LAYERS.get(stem)
                    ),
                    **TARGET_LOGITS.get(stem, {}),
                }
                self.assertIsNotNone(target["num_hidden_layers"], "unknown target")
                preset = yaml.safe_load((PRESETS / f"{stem}.yaml").read_text())
                self.assertEqual(
                    parse_draft(preset, target_config=target),
                    decode_legacy_config(
                        legacy, target_config=target, **LEGACY_SETTINGS.get(stem, {})
                    ),
                )


if __name__ == "__main__":
    unittest.main()
