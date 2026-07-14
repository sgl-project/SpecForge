import types
import unittest
from unittest import mock

import torch

from scripts import train_eagle3


class TestRunBackwardAndUpdate(unittest.TestCase):
    def test_returns_optimizer_owned_distributed_norm_without_reducing_again(self):
        expected_norm = torch.tensor(3.0)

        class RecordingOptimizer:
            def __init__(self):
                self.calls = 0

            def step(self):
                self.calls += 1
                return expected_norm

        optimizer = RecordingOptimizer()
        loss = torch.tensor(2.0, requires_grad=True)
        args = types.SimpleNamespace(draft_accumulation_steps=1)

        with (
            mock.patch.object(train_eagle3.dist, "is_initialized", return_value=True),
            mock.patch.object(train_eagle3.dist, "all_reduce") as all_reduce,
        ):
            actual_norm = train_eagle3.run_backward_and_update(
                args, [loss], optimizer, global_step=1
            )

        self.assertIs(actual_norm, expected_norm)
        self.assertEqual(optimizer.calls, 1)
        all_reduce.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
