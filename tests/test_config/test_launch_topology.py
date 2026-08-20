import unittest
from pathlib import Path

from specforge.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG_DIR = REPO_ROOT / "examples" / "configs"


def _recipes() -> dict[str, Path]:
    return {
        path.name: path
        for path in sorted(EXAMPLE_CONFIG_DIR.glob("*.yaml"))
        if not path.name.startswith(".")
    }


class ExampleLaunchTopologyTest(unittest.TestCase):
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
