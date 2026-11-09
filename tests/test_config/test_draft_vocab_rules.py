"""Generic vocabulary rules hold for every architecture, not only target-context ones."""

import unittest
from unittest.mock import patch

from specforge.config.draft import registry
from specforge.config.draft.architecture import DFlashSpec
from specforge.config.draft.registry import register_spec
from tests.test_config.draft_fixtures import draft_config


class VocabularyRulesTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(registry._SPECS))
        # an architecture with no contract of its own, so only the generic rules apply
        register_spec("UnconstrainedDraftModel", DFlashSpec, lambda _: None)

    def config(self, **overrides):
        return draft_config(architecture="UnconstrainedDraftModel", **overrides)

    def test_a_bound_lm_head_needs_the_full_vocabulary(self):
        with self.assertRaisesRegex(ValueError, "bound lm_head requires draft_vocab_size"):
            self.config(draft_vocab_size=32)

    def test_draft_vocabulary_never_exceeds_the_vocabulary(self):
        with self.assertRaisesRegex(ValueError, "must not exceed vocab_size"):
            self.config(
                draft_vocab_size=65, target_bindings=(), target_logit_transform=None
            )

    def test_tied_embeddings_need_equal_vocabularies(self):
        with self.assertRaisesRegex(ValueError, "equal vocabulary sizes"):
            self.config(
                draft_vocab_size=32,
                target_bindings=(),
                target_logit_transform=None,
                tie_word_embeddings=True,
            )


if __name__ == "__main__":
    unittest.main()
