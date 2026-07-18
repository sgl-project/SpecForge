"""Import-light tests for shared training model assembly helpers."""

import types
import unittest

from specforge.training.model_utils import resolve_mask_token_id


class _Tokenizer:
    def __init__(
        self,
        *,
        size=32,
        mask_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        unk_token_id=None,
    ):
        self.size = size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id

    def __len__(self):
        return self.size


def _draft(mask_token_id=None):
    config = types.SimpleNamespace(dflash_config={"mask_token_id": mask_token_id})
    return types.SimpleNamespace(config=config)


class TestResolveMaskTokenId(unittest.TestCase):
    def test_priority_is_explicit_draft_tokenizer_unused_then_fallback(self):
        tokenizer = _Tokenizer(size=30, mask_token_id=7, pad_token_id=3)
        self.assertEqual(
            resolve_mask_token_id(
                explicit=12,
                tokenizer=tokenizer,
                draft_model=_draft(16),
                embedding_vocab_size=32,
            ),
            12,
        )
        self.assertEqual(
            resolve_mask_token_id(
                explicit=None,
                tokenizer=tokenizer,
                draft_model=_draft(16),
                embedding_vocab_size=32,
            ),
            16,
        )
        self.assertEqual(
            resolve_mask_token_id(
                explicit=None,
                tokenizer=tokenizer,
                embedding_vocab_size=32,
            ),
            7,
        )
        tokenizer.mask_token_id = None
        self.assertEqual(
            resolve_mask_token_id(
                explicit=None,
                tokenizer=tokenizer,
                embedding_vocab_size=32,
            ),
            30,
        )
        tokenizer.size = 32
        self.assertEqual(
            resolve_mask_token_id(
                explicit=None,
                tokenizer=tokenizer,
                embedding_vocab_size=32,
            ),
            3,
        )

    def test_candidates_must_fit_embedding_table(self):
        tokenizer = _Tokenizer(size=32, pad_token_id=33)
        with self.assertRaisesRegex(ValueError, "outside draft embedding"):
            resolve_mask_token_id(
                explicit=33,
                tokenizer=tokenizer,
                embedding_vocab_size=32,
            )
        with self.assertRaisesRegex(ValueError, "unable to resolve"):
            resolve_mask_token_id(
                explicit=None,
                tokenizer=tokenizer,
                embedding_vocab_size=32,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
