"""Run in a fresh interpreter so a prior parity test cannot mask eager imports."""

import sys

from specforge.modeling import build_draft_model
from tests.test_config.draft_fixtures import draft_config

model = build_draft_model(draft_config(), device="meta")
assert model.config.draft.architecture == "DFlashDraftModel"
for name in ("legacy", "dflash", "llama3_eagle", "mtp", "registry", "config"):
    assert f"specforge.modeling.draft.{name}" not in sys.modules, name

# Model registration belongs to the canonical modeling package.
from specforge.modeling.draft.models.registry import resolve_draft_model

assert resolve_draft_model("DFlashDraftModel") is type(model)
