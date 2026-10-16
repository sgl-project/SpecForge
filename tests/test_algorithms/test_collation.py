from __future__ import annotations

import unittest

import torch

from specforge.algorithms.common.collation import pad_and_concatenate_features


def _sample(length, *, with_teacher):
    feature = {
        "input_ids": torch.arange(length).unsqueeze(0),
        "loss_mask": torch.ones(1, length, dtype=torch.long),
        "hidden_states": torch.ones(1, length, 4),
    }
    if with_teacher:
        feature["target_last_hidden_states"] = torch.full((1, length, 3), 2.0)
    return feature


class PadAndConcatenateFeaturesTest(unittest.TestCase):
    REQUIRED = ("input_ids", "loss_mask", "hidden_states")
    OPTIONAL = ("target_last_hidden_states",)

    def _collate(self, features):
        return pad_and_concatenate_features(
            features,
            sequence_axes={key: 1 for key in (*self.REQUIRED, *self.OPTIONAL)},
            required_keys=self.REQUIRED,
            optional_keys=self.OPTIONAL,
        )

    def test_optional_key_is_padded_when_every_sample_has_it(self):
        batch = self._collate(
            [_sample(2, with_teacher=True), _sample(3, with_teacher=True)]
        )

        teacher = batch["target_last_hidden_states"]
        self.assertEqual((2, 3, 3), tuple(teacher.shape))
        self.assertTrue(torch.all(teacher[0, :2] == 2.0))
        self.assertTrue(torch.all(teacher[0, 2:] == 0.0))

    def test_optional_key_is_omitted_when_no_sample_has_it(self):
        batch = self._collate(
            [_sample(2, with_teacher=False), _sample(3, with_teacher=False)]
        )

        self.assertEqual(set(self.REQUIRED), set(batch))

    def test_mixed_optional_presence_is_rejected(self):
        with self.assertRaisesRegex(KeyError, "target_last_hidden_states"):
            self._collate(
                [_sample(2, with_teacher=True), _sample(3, with_teacher=False)]
            )

    def test_missing_required_key_is_still_rejected(self):
        broken = _sample(2, with_teacher=True)
        del broken["hidden_states"]

        with self.assertRaisesRegex(KeyError, "hidden_states"):
            self._collate([broken])


if __name__ == "__main__":
    unittest.main()
