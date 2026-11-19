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

    def test_sequence_normalized_objective_matches_dense_reference(self):
        import torch.nn.functional as F

        model = training_model(replacement_probability=0.0)
        batch = training_inputs()
        for override, ramp in [(None, 0.5), (2, 1.0), (0, 1.0)]:
            model.objective.alignment_warmup_steps = override
            model.objective.alignment_warmup_ratio = 0.04 if override is None else 0.0
            kwargs = dict(
                step=2,
                total_steps=100,
                alignment_ce=torch.tensor(1.0),
            )
            torch.manual_seed(11)
            (p,) = model.objective(model, batch, **kwargs).predictions
            logits = p.project()["logits"].float()
            labels, teacher = p.labels, p.teacher.logits().float()
            w = model.weighting(p.mask.float(), p.positions)

            def ce(scores, targets):
                return F.cross_entropy(
                    scores.flatten(0, 1), targets.flatten(), reduction="none"
                ).view_as(w)

            def l1(scores, target):
                return (scores.softmax(-1) - target.softmax(-1)).abs().sum(-1)

            def mean(values, mask):
                weights = w * mask
                return (values * weights).sum() / weights.sum().clamp_min(1e-30)

            ids = logits.topk(model.objective.top_k).indices
            covered = ids == labels[..., None]
            eligible = (
                teacher.softmax(-1).gather(-1, ids).sum(-1)
                > model.objective.candidate_mass_threshold
            )
            inputs, draft = p.context_inputs, model.draft_model
            refined = draft.refine(inputs["pre_injection"], inputs["predecessors"])
            scores = torch.einsum(
                "...h,...kh->...k", refined, draft.lm_head.candidate_weights(ids)
            ).float()
            refinement = 0.1 * mean(
                ce(scores, covered.long().argmax(-1)), covered.any(-1)
            ) + 0.9 * mean(l1(scores, teacher.gather(-1, ids)), eligible)
            codes = F.embedding(labels, draft.transfer_codes).unsqueeze(-2)
            alignment = (
                1 - F.cosine_similarity(inputs["layer_features"], codes, dim=-1)
            ).mean(-1)
            expected = (
                (
                    0.1 * ce(logits, labels)
                    + 0.9 * l1(logits, teacher)
                    + 0.5 * ramp * alignment
                )
                * w
            ).sum() + refinement * p.mask.sum()
            torch.manual_seed(11)
            result = model(batch, **kwargs)
            self.assertEqual(result.denominator.item(), batch["input_ids"].shape[0])
            torch.testing.assert_close(result.numerator, expected)

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
