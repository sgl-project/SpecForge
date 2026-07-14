from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Optional

from specforge.training.strategies.registry import resolve_strategy

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG_DIR = REPO_ROOT / "examples" / "configs"
DRAFT_CONFIG_DIR = REPO_ROOT / "configs"


def _yaml_scalar(path: Path, key: str) -> Optional[str]:
    prefix = f"{key}:"
    matches = []
    for line in path.read_text().splitlines():
        content = line.split("#", 1)[0].strip()
        if not content.startswith(prefix):
            continue
        value = content[len(prefix) :].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        matches.append(value)
    if len(matches) > 1:
        raise AssertionError(f"{path} defines {key} more than once")
    return matches[0] if matches else None


def _local_draft_configs():
    for recipe in sorted(EXAMPLE_CONFIG_DIR.glob("*.yaml")):
        source = _yaml_scalar(recipe, "draft_model_config")
        if source is None or not source.endswith(".json"):
            continue
        strategy = _yaml_scalar(recipe, "strategy")
        yield recipe, strategy, REPO_ROOT / source


class ExampleDraftConfigWiringTest(unittest.TestCase):
    def test_local_draft_architecture_matches_recipe_strategy(self):
        recipes = list(_local_draft_configs())
        self.assertTrue(recipes)

        for recipe, strategy, draft_config in recipes:
            with self.subTest(recipe=recipe.name):
                spec = resolve_strategy(strategy)
                self.assertTrue(draft_config.is_file(), draft_config)
                payload = json.loads(draft_config.read_text())
                self.assertEqual(
                    payload.get("architectures"),
                    [spec.assembly.draft_config.architecture],
                )
                expected_auto_model = spec.assembly.draft_config.expected_auto_map_model
                if expected_auto_model is not None:
                    self.assertEqual(
                        payload.get("auto_map", {}).get("AutoModel"),
                        expected_auto_model,
                    )

    def test_every_checked_in_draft_config_has_a_unified_yaml_recipe(self):
        referenced = {
            draft_config.resolve()
            for _, _, draft_config in _local_draft_configs()
            if draft_config.is_file()
        }
        checked_in = {path.resolve() for path in DRAFT_CONFIG_DIR.glob("*.json")}

        self.assertEqual(checked_in, referenced)


if __name__ == "__main__":
    unittest.main(verbosity=2)
