# coding=utf-8
"""t2d/d2t construction from token frequencies.

The mapping is built once per run but consumed by every training step and by
serving, so this pins it against the formulation it replaced rather than
against hand-written expectations: the two must agree bit for bit, including
dtypes, or a rebuilt mapping would silently reorder a checkpoint's head rows.
"""

import random
import unittest
from collections import Counter

import torch

from specforge.data.preprocessing import process_token_dict_to_mappings


def _reference_mappings(top_n, target_vocab_size):
    """The pre-vectorization construction, transcribed and frozen.

    Kept as literal Python so it cannot drift with the implementation under
    test; its ``i in used_tokens`` list scan is exactly the cost that made the
    real thing worth replacing.
    """
    used_tokens = [key for key, _frequency in top_n]
    used_tokens.sort()
    d2t = [used_tokens[i] - i for i in range(len(used_tokens))]
    t2d = [i in used_tokens for i in range(target_vocab_size)]
    return torch.tensor(d2t), torch.tensor(t2d)


class VocabMappingConstructionTest(unittest.TestCase):
    def test_matches_the_reference_construction(self):
        random.seed(0)
        for vocab_size, draft_vocab_size in ((256, 64), (5000, 1500), (20000, 7000)):
            with self.subTest(vocab_size=vocab_size, draft=draft_vocab_size):
                distinct = min(vocab_size, int(draft_vocab_size * 1.7))
                counts = Counter(
                    {
                        token: random.randint(1, 1000)
                        for token in random.sample(range(vocab_size), distinct)
                    }
                )
                expected = _reference_mappings(
                    counts.most_common(draft_vocab_size), vocab_size
                )
                actual = process_token_dict_to_mappings(
                    Counter(counts), draft_vocab_size, vocab_size
                )

                for name, got, want in zip(("d2t", "t2d"), actual, expected):
                    self.assertEqual(want.dtype, got.dtype, name)
                    self.assertTrue(torch.equal(want, got), name)

    def test_satisfies_the_invariant_the_model_checks_on_install(self):
        """nonzero(t2d) == d2t + arange, which d2t being an offset table needs."""
        from specforge.core.compact_teacher import validate_vocab_mapping_consistency

        counts = Counter({token: token + 1 for token in range(0, 200, 3)})
        d2t, t2d = process_token_dict_to_mappings(counts, 32, 256)

        validate_vocab_mapping_consistency(t2d, d2t)
        self.assertEqual(32, int(t2d.sum()))
        selected = torch.nonzero(t2d, as_tuple=False).flatten()
        self.assertTrue(torch.equal(selected, d2t + torch.arange(32)))

    def test_pads_when_the_corpus_has_too_few_distinct_tokens(self):
        """A short corpus must still yield exactly draft_vocab_size entries."""
        counts = Counter({1: 5, 7: 3})
        d2t, t2d = process_token_dict_to_mappings(counts, 8, 64)

        self.assertEqual((8,), tuple(d2t.shape))
        self.assertEqual((64,), tuple(t2d.shape))
        self.assertEqual(8, int(t2d.sum()))
        # The observed tokens survive the padding.
        self.assertTrue(bool(t2d[1]) and bool(t2d[7]))


if __name__ == "__main__":
    unittest.main()
