"""Objective composition over the shared logits-loss outputs."""

import unittest
from itertools import product

import torch
import torch.nn.functional as F

from specforge.core.fused_loss import ce_l1_loss
from specforge.core.token_objective import (
    TokenObjectiveConfig,
    compose_token_objective,
    token_objective_chunk_terms,
)
from tests.test_utils import test_dflash_losses as fixtures


class TokenObjectiveTest(unittest.TestCase):
    def test_loss_and_gradient_with_independent_weights(self):
        torch.manual_seed(42)
        for mode, dtype in product(
            (None, "alpha", "tv", "lambda"), (torch.bool, torch.int64, torch.float32)
        ):
            with self.subTest(mode=mode, mask_dtype=dtype):
                logits = torch.randn(2, 3, 5, requires_grad=True)
                teacher = torch.randn_like(logits, requires_grad=True)
                labels = torch.randint(5, (2, 3))
                mask = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=dtype)
                hard_weights = mask * torch.tensor([1.0, 0.5, 0.25])
                teacher_weights = mask * torch.tensor([0.3, 0.7, 1.0])
                config = TokenObjectiveConfig(
                    hard_label_mode=mode,
                    hard_ce_weight=0.4,
                    teacher_l1_weight=0.6,
                    kl_scale=0.7,
                    kl_decay=1.3,
                )
                losses = ce_l1_loss(logits, labels, teacher_logits=teacher)
                terms = token_objective_chunk_terms(
                    config,
                    cross_entropy=losses.cross_entropy,
                    hard_weights=hard_weights,
                    teacher_l1=losses.teacher_l1,
                    teacher_weights=teacher_weights,
                )
                probability_num = (losses.cross_entropy.neg().exp() * mask).sum()
                actual = compose_token_objective(
                    config,
                    ce_num=terms.ce_num,
                    hard_tv_num=terms.hard_tv_num,
                    teacher_l1_num=terms.teacher_l1_num,
                    probability_num=probability_num,
                    probability_den=mask.sum(),
                )

                ce = F.cross_entropy(
                    logits.flatten(0, 1), labels.flatten(), reduction="none"
                ).view_as(labels)
                probability = ce.neg().exp()
                hard = ce
                if mode == "tv":
                    hard = 1 - probability
                elif mode == "lambda":
                    coefficient = 0.7 * torch.exp(
                        -1.3 * ((probability * mask).sum() / mask.sum()).detach()
                    )
                    hard = coefficient * ce + (1 - coefficient) * (1 - probability)
                l1 = (logits.softmax(-1) - teacher.detach().softmax(-1)).abs().sum(-1)
                expected = (
                    0.4 * (hard * hard_weights).sum()
                    + 0.6 * (l1 * teacher_weights).sum()
                )
                torch.testing.assert_close(actual, expected)
                gradient, teacher_gradient = torch.autograd.grad(
                    actual, (logits, teacher), retain_graph=True, allow_unused=True
                )
                torch.testing.assert_close(
                    gradient, torch.autograd.grad(expected, logits)[0]
                )
                self.assertIsNone(teacher_gradient)

    def test_teacher_term_requires_explicit_data_and_weights(self):
        config = TokenObjectiveConfig(teacher_l1_weight=0.9)
        values = torch.ones(2, 3)
        for teacher_l1, teacher_weights in ((None, values), (values, None)):
            with self.assertRaisesRegex(ValueError, "teacher_l1 and teacher_weights"):
                token_objective_chunk_terms(
                    config,
                    cross_entropy=values,
                    hard_weights=values,
                    teacher_l1=teacher_l1,
                    teacher_weights=teacher_weights,
                )
        with self.assertRaisesRegex(ValueError, "teacher_l1_num"):
            compose_token_objective(config, ce_num=values.sum())

    def test_model_mapping_keeps_teacher_requirements_independent(self):
        logits, _, _, _, anchors, keep = fixtures._sample_tensors()
        dflash = fixtures._make_model(logits, anchors, keep, lk_loss_type="lambda")
        self.assertEqual(dflash.token_objective.hard_label_mode, "lambda")
        self.assertFalse(dflash.requires_teacher_distribution)
        dflash.lk_loss_type = "tv"
        self.assertEqual(dflash.token_objective.hard_label_mode, "tv")
        dspark = fixtures._make_dspark_model(logits, anchors, keep)
        self.assertTrue(dspark.requires_teacher_distribution)
        self.assertEqual(dspark.token_objective.hard_ce_weight, 0.1)
        dspark.dspark_l1_loss_alpha = 0
        self.assertFalse(dspark.requires_teacher_distribution)
        # Confidence still requires teacher states when the token objective does not.
        dspark.draft_model.confidence_head = torch.nn.Identity()
        dspark.dspark_confidence_head_alpha = 1
        _, ids, mask, hidden, _, _ = fixtures._sample_tensors()
        with self.assertRaisesRegex(ValueError, "target_last_hidden_states"):
            dspark(ids, hidden, mask)
