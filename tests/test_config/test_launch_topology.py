from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from specforge.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG_DIR = REPO_ROOT / "examples" / "configs"


def _recipes() -> dict[str, Path]:
    paths = [
        path
        for path in sorted(EXAMPLE_CONFIG_DIR.rglob("*.yaml"))
        if not path.name.startswith(".")
    ]
    recipes = {path.name: path for path in paths}
    if len(recipes) != len(paths):
        raise AssertionError("example recipe filenames must be globally unique")
    return recipes


class ExampleLaunchTopologyTest(unittest.TestCase):
    def test_every_recipe_matches_its_directory_topology(self):
        self.assertTrue(
            (EXAMPLE_CONFIG_DIR / "online" / "colocated" / "README.md").is_file()
        )
        for filename, path in _recipes().items():
            with self.subTest(config=filename):
                payload = yaml.safe_load(path.read_text())
                data = payload["data"]
                mode = (
                    "online"
                    if data.get("train_data_path") or data.get("prompts_path")
                    else "offline"
                )
                deployment = payload["deployment"]
                topology = (
                    "colocated"
                    if deployment["mode"] == "local_colocated"
                    else "disaggregated"
                )
                expected_parent = Path(mode) / topology
                if mode == "online" and topology == "disaggregated":
                    ownership = (
                        "managed-local"
                        if "managed_local" in deployment["disaggregated"]
                        else "external"
                    )
                    expected_parent /= ownership
                self.assertEqual(
                    path.relative_to(EXAMPLE_CONFIG_DIR).parent,
                    expected_parent,
                )

    def test_every_recipe_validates_for_its_declared_world_size(self):
        recipes = _recipes()
        self.assertTrue(recipes)

        for filename, path in recipes.items():
            with self.subTest(config=filename):
                config = Config.from_file(str(path))
                topology = config.deployment.trainer
                config.validate_world_size(topology.nnodes * topology.nproc_per_node)


if __name__ == "__main__":
    unittest.main(verbosity=2)
