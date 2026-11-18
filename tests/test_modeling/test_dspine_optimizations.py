"""DSpine uses shared loss execution without changing its sampling or gradients."""

import unittest
from dataclasses import replace

import torch

from tests.test_modeling.dspine_fixtures import training_inputs, training_model


class DSpineExecutionTest(unittest.TestCase):
    def test_chunking_and_checkpointing_preserve_all_gradients(self):
        model = training_model()
        batch = training_inputs()
        expected = None
        for chunk, checkpoint in ((0, False), (1, False), (1, True)):
            model.execution = replace(
                model.execution, chunk_size=chunk, checkpoint=checkpoint
            )
            model.zero_grad(set_to_none=True)
            torch.manual_seed(19)
            result = model(
                batch, step=300, total_steps=600, alignment_ce=torch.tensor(0.5)
            )
            result.loss.backward()
            current = (
                result.loss.detach(),
                {
                    name: p.grad.clone()
                    for name, p in model.named_parameters()
                    if p.grad is not None
                },
            )
            if expected is None:
                expected = current
            else:
                torch.testing.assert_close(current, expected)
            self.assertIsNone(model.target_head.weight.grad)

    def test_refinement_masks_select_ce_and_l1_independently(self):
        from specforge.algorithms.objectives.losses import LossContext, TokenLoss

        logits = torch.randn(3, 4, requires_grad=True)
        teacher = torch.randn(3, 4)
        labels = torch.tensor([0, 1, 2])
        context = LossContext(
            torch.ones(3),
            ce_mask=torch.tensor([1, 0, 0]),
            l1_mask=torch.tensor([0, 1, 0]),
        )
        loss = TokenLoss("ce_l1", {"ce_weight": 0.1, "l1_weight": 0.9})
        actual = loss(logits, labels, teacher=teacher, context=context)
        ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        l1 = (logits.softmax(-1) - teacher.softmax(-1)).abs().sum(-1)
        expected = 0.1 * ce * context.ce_mask + 0.9 * l1 * context.l1_mask
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0],
            torch.autograd.grad(expected.sum(), logits)[0],
        )
