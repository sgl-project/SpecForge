"""Independent formulas for block loss weights, LK mixing and DSpark distillation."""

import unittest
from types import SimpleNamespace as NS

import torch
from torch import nn
from torch.nn import functional as F

from specforge.algorithms.losses.blocks import anchor_scales
from specforge.algorithms.losses.dflash import (
    compose_token_objective,
    dflash_objective_chunk_terms,
)
from specforge.algorithms.losses.dspark import dspark_objective_chunk_terms
from specforge.algorithms.objectives.block_options import DFlashLossOptions
from specforge.algorithms.objectives.dspark import DSparkObjective


def model_for(slots):
    return NS(
        lm_head=nn.Identity(),
        candidate_selector=None,
        markov_head=None,
        confidence_head=None,
        config=NS(
            draft=NS(
                spec_config=NS(
                    protocol=NS(num_model_slots=slots), confidence=NS(with_markov=False)
                )
            )
        ),
    )


def dpace_reference(probability, mask, alpha, mode):
    smoothed = torch.where(mask > 0, (1 - alpha) * probability + alpha, 1)
    prefix = smoothed.cumprod(-1)
    suffix = (prefix * mask).flip(-1).cumsum(-1).flip(-1)
    if mode == "dpace-cumulative-confidence-only":
        return prefix
    return (
        suffix
        if mode == "dpace"
        else suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
    )


class BlockLossFormulaTest(unittest.TestCase):
    def test_dflash_dpace_and_lk_match_loss_and_gradient_formulas(self):
        torch.manual_seed(12)
        labels = torch.randint(7, (2, 2, 5))
        mask = torch.tensor(
            [[[0, 1, 1, 1, 0], [0, 1, 1, 0, 0]], [[0, 1, 1, 1, 1], [0, 0, 0, 0, 0]]]
        ).float()
        scale = anchor_scales(mask)
        for mode in (
            "dflash",
            "dpace",
            "dpace-cumulative-confidence-only",
            "dpace-continuation-value-only",
        ):
            for lk in (None, "alpha", "lambda", "tv"):
                with self.subTest(mode=mode, lk=lk):
                    logits = torch.randn(2, 2, 5, 7, requires_grad=True)
                    reference = logits.detach().clone().requires_grad_()
                    options = DFlashLossOptions(
                        loss_type=mode,
                        lk_loss_type=lk,
                        loss_decay_gamma=3,
                        dpace_alpha=0.4,
                        fused_logits_loss=False,
                    )
                    terms = dflash_objective_chunk_terms(
                        model_for(5), options, logits, labels, mask, labels, scale
                    )
                    actual = (
                        compose_token_objective(
                            options,
                            terms.ce_loss_num,
                            terms.tv_loss_num,
                            terms.target_probability_num,
                            mask.sum(),
                        )
                        / terms.loss_den
                    )
                    ce = F.cross_entropy(
                        reference.flatten(0, 2), labels.flatten(), reduction="none"
                    ).reshape_as(mask)
                    prob = (-ce).exp()
                    if mode == "dflash":
                        weights = mask * torch.exp(
                            -(torch.arange(5) - 1).clamp_min(0) / 3
                        )
                        denominator = weights.sum()
                    else:
                        weights = (
                            mask
                            * dpace_reference(prob.detach(), mask, 0.4, mode)
                            * scale
                        )
                        denominator = torch.tensor(2.0)
                    token = ce
                    if lk == "tv":
                        token = 1 - prob
                    elif lk == "lambda":
                        mixing = torch.exp(-(prob.detach() * mask).sum() / mask.sum())
                        token = mixing * ce + (1 - mixing) * (1 - prob)
                    expected = (weights * token).sum() / denominator
                    torch.testing.assert_close(actual, expected)
                    actual.backward()
                    expected.backward()
                    torch.testing.assert_close(
                        logits.grad, reference.grad, atol=2e-7, rtol=2e-5
                    )
                    self.assertFalse(terms.loss_den.requires_grad)

    def test_dspark_ce_l1_and_detached_confidence_target(self):
        torch.manual_seed(13)
        model = model_for(3)
        model.confidence_head = lambda hidden: hidden[..., 0]
        logits = torch.randn(1, 2, 3, 7, requires_grad=True)
        reference = logits.detach().clone().requires_grad_()
        teacher = torch.randn_like(logits)
        labels = torch.randint(7, (1, 2, 3))
        mask = torch.tensor([[[1, 1, 1], [1, 0, 0]]], dtype=torch.bool)
        options = DSparkObjective(fused_logits_loss=False)
        terms = dspark_objective_chunk_terms(
            model,
            options,
            nn.Identity(),
            logits,
            labels,
            labels,
            mask.float(),
            mask,
            teacher,
        )
        actual = 0.1 * terms[0] + 0.9 * terms[1] + terms[2]
        ce = F.cross_entropy(
            reference.flatten(0, 2), labels.flatten(), reduction="none"
        ).reshape_as(mask)
        l1 = (reference.softmax(-1) - teacher.softmax(-1)).abs().sum(-1)
        confidence = F.binary_cross_entropy_with_logits(
            reference[..., 0], (1 - 0.5 * l1).detach(), reduction="none"
        )
        expected = ((0.1 * ce + 0.9 * l1 + confidence) * mask).sum()
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(logits.grad, reference.grad)
