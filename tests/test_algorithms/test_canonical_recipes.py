"""All example objectives resolve without loading target weights."""

import unittest
from pathlib import Path

import yaml

from specforge.algorithms.objectives import resolve_objective
from specforge.config import Config
from specforge.config.draft.codec import parse_draft

ROOT = Path(__file__).resolve().parents[2]


class CanonicalRecipesTest(unittest.TestCase):
    def test_objective_and_backend_choices(self):
        for path in (ROOT / "examples/configs").rglob("*.yaml"):
            raw = yaml.safe_load(path.read_text())
            if not isinstance(raw, dict) or "draft" not in raw:
                continue
            with self.subTest(recipe=str(path)):
                cfg = Config.model_validate(raw)
                payload = (
                    yaml.safe_load((ROOT / cfg.draft).read_text())
                    if isinstance(cfg.draft, str)
                    else cfg.draft
                )
                draft = parse_draft(payload, target_config=None)
                result = resolve_objective(
                    draft.architecture,
                    name=cfg.training.objective,
                    options=cfg.training.objective_options,
                    loss=cfg.training.loss,
                    weighting=cfg.training.weighting,
                )
                self.assertIn(
                    cfg.training.attention_backend,
                    result.registration.training.attention_backends,
                )
