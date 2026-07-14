"""Position-ID collation stays shape-generic without a VLM-only collator."""

import unittest

import torch

from specforge.data.utils import DataCollatorWithPadding


def _feature(length, position_ids):
    return {
        "input_ids": torch.arange(length).unsqueeze(0),
        "attention_mask": torch.ones(1, length, dtype=torch.long),
        "loss_mask": torch.ones(1, length, dtype=torch.long),
        "position_ids": position_ids,
    }


class PositionIdsCollatorTest(unittest.TestCase):
    def test_collates_2d_position_ids(self):
        features = [
            _feature(2, torch.tensor([[3, 4]])),
            _feature(3, torch.tensor([[7, 8, 9]])),
        ]

        batch = DataCollatorWithPadding()(features)

        self.assertTrue(
            torch.equal(
                batch["position_ids"],
                torch.tensor([[3, 4, 0], [7, 8, 9]]),
            )
        )

    def test_collates_generic_3d_position_ids(self):
        features = [
            _feature(
                2,
                torch.tensor(
                    [
                        [[1, 2]],
                        [[11, 12]],
                    ]
                ),
            ),
            _feature(
                3,
                torch.tensor(
                    [
                        [[3, 4, 5]],
                        [[13, 14, 15]],
                    ]
                ),
            ),
        ]

        batch = DataCollatorWithPadding()(features)

        self.assertEqual(batch["position_ids"].shape, (2, 2, 3))
        self.assertTrue(
            torch.equal(
                batch["position_ids"],
                torch.tensor(
                    [
                        [[1, 2, 0], [3, 4, 5]],
                        [[11, 12, 0], [13, 14, 15]],
                    ]
                ),
            )
        )


if __name__ == "__main__":
    unittest.main()
