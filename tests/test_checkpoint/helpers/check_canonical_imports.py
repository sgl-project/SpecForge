"""Fresh-interpreter check of the public draft package after a canonical build."""

from specforge.modeling import build_draft_model
from tests.test_config.draft_fixtures import draft_config

model = build_draft_model(draft_config(), device="meta")
assert model.config.draft.architecture == "DFlashDraftModel"

# The public package no longer exposes historical model factories.
import specforge.modeling.draft as draft

assert not hasattr(draft, "available_drafts")
