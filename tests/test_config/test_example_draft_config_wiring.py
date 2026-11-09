"""Bundled run recipes use complete canonical drafts and valid objective options."""

import unittest
from pathlib import Path

from specforge.application.composition import resolve_run
from specforge.config import Config
from specforge.config.draft.codec import parse_draft

ROOT = Path(__file__).resolve().parents[2]


class TrainingPresetTest(unittest.TestCase):
    def test_recipes_resolve_canonical_models_and_objectives(self):
        for path in sorted((ROOT / "examples/configs").rglob("*.yaml")):
            with self.subTest(recipe=path.relative_to(ROOT)):
                cfg = Config.from_file(str(path))
                resolved = resolve_run(cfg)
                draft = parse_draft(resolved.config.draft, target_config=None)
                self.assertEqual(
                    resolved.algorithm.spec.draft.default_architecture,
                    draft.architecture,
                )
                self.assertIsNotNone(resolved.config.training.objective)


if __name__ == "__main__":
    unittest.main()
