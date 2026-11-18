"""Liger preserves DSpine's accumulated updates and fixed transfer buffers."""

import copy
import unittest

import torch

from specforge.modeling.draft.layers.kernels import configure_kernels
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import StepContext
from tests.test_modeling.dspine_fixtures import training_inputs, training_model
from tests.test_modeling.test_dspine_accumulation import training_runner
from tests.test_runtime._fixtures import train_window


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA and Liger")
class DSpineLigerTest(unittest.TestCase):
    def test_replayed_updates(self):
        for dtype in (torch.float32, torch.bfloat16):
            original = training_model(
                chunk_size=1, head_dim=16, device="cuda", dtype=dtype
            )
            inputs = training_inputs()
            results = []
            for enabled in (False, True):
                model = copy.deepcopy(original)
                configure_kernels(model.draft_model, use_liger=enabled)
                before = model.draft_model.transfer_codes.clone()
                runner, optimizer = training_runner(model)
                torch.manual_seed(321)
                metrics = train_window(
                    runner,
                    [
                        TrainBatch(
                            [str(index)],
                            "dspine",
                            {k: v[index : index + 1] for k, v in inputs.items()},
                        )
                        for index in range(2)
                    ],
                    StepContext(global_step=100, total_steps=600),
                )
                self.assertEqual(len(optimizer.gradients), 1)
                self.assertIsNone(model.target_head.weight.grad)
                torch.testing.assert_close(
                    model.draft_model.transfer_codes, before, atol=0, rtol=0
                )
                results.append((metrics.loss, optimizer.gradients[-1]))
            tol = 0.02 if dtype == torch.bfloat16 else 3e-5
            torch.testing.assert_close(results[0], results[1], atol=tol, rtol=tol)


if __name__ == "__main__":
    unittest.main()
