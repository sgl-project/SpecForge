"""P-EAGLE packed COD training updates its mask and survives canonical reload."""

import tempfile
import unittest

import torch

from specforge.algorithms.objectives import build_objective_model
from specforge.loading import load_draft
from specforge.modeling import build_draft_model
from specforge.optimizer import BF16Optimizer
from tests.test_algorithms.test_objectives import training_batch
from tests.test_modeling.test_canonical_eagle3 import install_mapping
from tests.test_modeling.test_canonical_peagle import model_config


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class PEagleTrainingTest(unittest.TestCase):
    def test_packed_training_and_reload(self):
        config = model_config(target_width=32)
        draft = build_draft_model(config, device="cuda", dtype=torch.bfloat16)
        install_mapping(draft)
        teacher = torch.nn.Linear(
            32, 64, bias=False, device="cuda", dtype=torch.bfloat16
        )
        options = dict(
            mask_token_id=31,
            num_depths=3,
            down_sample_ratio=1.0,
            down_sample_ratio_min=1.0,
        )
        model = build_objective_model(draft, target_head=teacher, options=options)
        batch = training_batch(config, "cuda")
        batch["lengths"] = torch.tensor([4, 5], device="cuda")
        before = draft.mask_hidden.detach().clone()
        optimizer = BF16Optimizer(draft, lr=0.01, warmup_ratio=0.0, total_steps=2)
        model(batch).loss.backward()
        for name, parameter in draft.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(draft.mask_hidden.grad.float().abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, draft.mask_hidden))
        self.assertIsNone(teacher.weight.grad)
        with tempfile.TemporaryDirectory() as path:
            draft.save_pretrained(path)
            restored = load_draft(path, device="cuda", dtype=torch.bfloat16)
        other = build_objective_model(restored, target_head=teacher, options=options)
        torch.manual_seed(43)
        expected = model(batch).loss
        torch.manual_seed(43)
        actual = other(batch).loss
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
