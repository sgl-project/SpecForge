"""Run in a fresh interpreter so a prior parity test cannot mask eager imports."""

import sys

from specforge.modeling import build_draft_model
from tests.test_config.draft_fixtures import draft_config

model = build_draft_model(draft_config(), device="meta")
assert model.config.draft.architecture == "DFlashDraftModel"
for name in ("legacy", "dflash", "llama3_eagle", "mtp", "registry", "config"):
    assert f"specforge.modeling.draft.{name}" not in sys.modules, name

# Explicit old API use still registers every historical architecture.
from specforge.modeling.draft import available_drafts

assert "Qwen3_5MTPDraftModel" not in available_drafts()
assert "DFlashDraftModel" in available_drafts()
