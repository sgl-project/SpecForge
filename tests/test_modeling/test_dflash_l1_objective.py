import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from specforge.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
    _triton_available,
)
from specforge.algorithms.dflash.providers import resume_contract
from specforge.core import fused_loss, token_objective
from specforge.core.token_objective import TokenObjectiveConfig
from specforge.modeling.draft.dflash2 import CandidateSelector


class _Draft(nn.Module):
    """Selector-bearing draft stub with the identity unary transform."""

    def __init__(self):
        super().__init__()
        self.candidate_selector = object()

    @staticmethod
    def transform_unary_logits(logits):
        return logits


def _model(**kwargs):
    return OnlineDFlashModel(
        draft_model=_Draft(),
        target_lm_head=nn.Identity(),
        target_embed_tokens=nn.Embedding(4, 4),
        mask_token_id=3,
        block_size=3,
        attention_backend="eager",
        selector_loss_alpha=0.0,
        **kwargs,
    )


def _l1(draft_logits, target_logits):
    return (
        (
            torch.softmax(draft_logits.float(), -1)
            - torch.softmax(target_logits.float(), -1)
        )
        .abs()
        .sum(-1)
    )


OUTPUT_HIDDEN = torch.tensor(
    [[[0.0, 0.0, 0.0, 0.0], [0.0, 4.0, 1.0, 0.0], [0.0, 3.0, 4.0, 2.0]]]
)
TARGET_HIDDEN = torch.tensor(
    [[[0.0, 5.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0], [1.0, 0.0, 2.0, 0.0]]]
)
INPUT_IDS = torch.tensor([[0, 1, 3]])


def _forward(model, *, teacher=True, detailed=False):
    hidden = OUTPUT_HIDDEN.clone().requires_grad_()
    model._forward_draft_blocks = lambda **_kwargs: (
        torch.tensor([[0]]),
        torch.tensor([[True]]),
        hidden,
    )
    loss, _accuracy, metrics = model(
        input_ids=INPUT_IDS,
        hidden_states=torch.zeros(1, 3, 4),
        loss_mask=torch.ones(1, 3),
        target_last_hidden_states=TARGET_HIDDEN if teacher else None,
        collect_detailed_metrics=detailed,
    )
    return hidden, loss, metrics


class _SelectorDraft(nn.Module):
    """DFlash2-style draft stub with a real top-2 candidate selector."""

    def __init__(self):
        super().__init__()
        self.candidate_selector = CandidateSelector(
            hidden_size=4, vocab_size=4, state_rank=2, top_k=2, initializer_range=0.02
        )

    @staticmethod
    def transform_unary_logits(logits):
        return logits


def _expected_terms():
    # Block anchored at 0: positions 1 and 2 predict tokens 1 and 3 from the
    # draft states at 1 and 2; the target state at label - 1 predicts each label.
    draft = OUTPUT_HIDDEN[0, 1:]
    teacher = TARGET_HIDDEN[0, :2]
    labels = INPUT_IDS[0, 1:]
    ce = torch.nn.functional.cross_entropy(draft, labels, reduction="sum")
    l1 = _l1(draft, teacher).sum()
    return ce, l1, 2.0


class DFlashL1ObjectiveTest(unittest.TestCase):
    """The weights bind to the shared token objective; the wrapper only
    supplies the teacher logits (``lm_head`` on the aligned target state)."""

    def test_weights_bind_to_the_shared_token_objective(self):
        self.assertEqual(_model().token_objective, TokenObjectiveConfig())
        self.assertFalse(_model().requires_teacher_distribution)

        model = _model(ce_loss_alpha=0.1, l1_loss_alpha=0.9)
        self.assertEqual(
            model.token_objective,
            TokenObjectiveConfig(hard_ce_weight=0.1, teacher_l1_weight=0.9),
        )
        self.assertTrue(model.requires_teacher_distribution)

    def test_chunk_l1_matches_eager_distance(self):
        model = _model(ce_loss_alpha=0.1, l1_loss_alpha=0.9)
        hidden = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 2.0, 1.0]]]])
        teacher = torch.tensor([[[[0.0, 0.0, 0.0], [3.0, 0.0, 1.0]]]])
        target_ids = torch.tensor([[[0, 0]]])
        weights = torch.tensor([[[0.0, 1.0]]])
        model.lm_head = nn.Identity()

        with mock.patch.object(
            token_objective,
            "teacher_tv_distance",
            wraps=token_objective.teacher_tv_distance,
        ) as shared_distance:
            terms = model._dflash_objective_chunk_terms(
                hidden, target_ids, weights, target_ids, None, teacher
            )
        # The distance is the shared objective's, fed with lm_head(teacher).
        shared_distance.assert_called_once()
        torch.testing.assert_close(shared_distance.call_args.args[1], teacher)

        torch.testing.assert_close(
            terms.teacher_l1_num, _l1(hidden[0, 0, 1], teacher[0, 0, 1])
        )
        torch.testing.assert_close(
            terms.ce_loss_num,
            torch.nn.functional.cross_entropy(
                hidden[0, 0, 1].unsqueeze(0), torch.tensor([0])
            ),
        )

    def test_forward_composes_weighted_ce_and_l1(self):
        hidden, loss, metrics = _forward(_model(ce_loss_alpha=0.1, l1_loss_alpha=0.9))
        ce, l1, den = _expected_terms()

        torch.testing.assert_close(loss, (0.1 * ce + 0.9 * l1) / den)
        ratios = metrics["ratio_metrics"]
        for name, value in (
            ("ce_loss", ce / den),
            ("l1_loss", l1 / den),
            ("token_objective", (0.1 * ce + 0.9 * l1) / den),
        ):
            numerator, denominator = ratios[name]
            torch.testing.assert_close(numerator / denominator, value)

        loss.backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())
        self.assertGreater(hidden.grad[0, 1:].abs().sum().item(), 0.0)
        torch.testing.assert_close(hidden.grad[0, 0], torch.zeros(4))

    def test_l1_gradient_matches_eager_autograd(self):
        hidden, loss, _ = _forward(_model(ce_loss_alpha=0.0, l1_loss_alpha=1.0))
        loss.backward()
        reference = OUTPUT_HIDDEN[0, 1:].clone().requires_grad_()
        (_l1(reference, TARGET_HIDDEN[0, :2]).sum() / 2.0).backward()
        torch.testing.assert_close(hidden.grad[0, 1:], reference.grad)

    def test_defaults_keep_the_plain_ce_objective(self):
        _hidden, loss, metrics = _forward(_model())
        ce, _l1_sum, den = _expected_terms()

        torch.testing.assert_close(loss, ce / den)
        self.assertNotIn("l1_loss", metrics["ratio_metrics"])
        self.assertNotIn("token_objective", metrics["ratio_metrics"])

    def test_l1_requires_teacher_states(self):
        with self.assertRaisesRegex(ValueError, "target_last_hidden_states"):
            _forward(_model(ce_loss_alpha=0.1, l1_loss_alpha=0.9), teacher=False)

    def test_l1_trains_with_teacher_diagnostics_off(self):
        # ``teacher_metrics`` governs only the diagnostics; the objective
        # still reads the teacher state on every step.
        model = OnlineDFlashModel(
            draft_model=_SelectorDraft(),
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(4, 4),
            mask_token_id=3,
            block_size=3,
            attention_backend="eager",
            selector_loss_alpha=0.0,
            ce_loss_alpha=0.1,
            l1_loss_alpha=0.9,
            teacher_metrics=False,
        )
        hidden, loss, metrics = _forward(model, detailed=True)
        ce, l1, den = _expected_terms()

        torch.testing.assert_close(loss, (0.1 * ce + 0.9 * l1) / den)
        ratios = metrics["ratio_metrics"]
        self.assertIn("l1_loss", ratios)
        self.assertFalse(
            [name for name in ratios if name.startswith("dflash/teacher/")]
        )
        loss.backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_rejects_non_ce_base_objectives(self):
        with self.assertRaisesRegex(ValueError, "plain DFlash CE"):
            _model(l1_loss_alpha=0.9, lk_loss_type="alpha")
        with self.assertRaisesRegex(ValueError, "plain DFlash CE"):
            _model(l1_loss_alpha=0.9, loss_type="dpace")
        with self.assertRaisesRegex(ValueError, "must be > 0"):
            _model(ce_loss_alpha=0.0, l1_loss_alpha=0.0)

    def test_fused_unary_head_is_disabled_with_l1(self):
        model = _model(ce_loss_alpha=0.1, l1_loss_alpha=0.9)
        self.assertFalse(model._use_fused_unary_head(torch.zeros(1, 1, 3, 4)))

    def test_selector_draft_with_detailed_metrics(self):
        model = OnlineDFlashModel(
            draft_model=_SelectorDraft(),
            target_lm_head=nn.Identity(),
            target_embed_tokens=nn.Embedding(4, 4),
            mask_token_id=3,
            block_size=3,
            attention_backend="eager",
            selector_loss_alpha=1.0,
            ce_loss_alpha=0.1,
            l1_loss_alpha=0.9,
        )
        hidden, loss, metrics = _forward(model, detailed=True)
        ce, l1, den = _expected_terms()
        ratios = metrics["ratio_metrics"]
        numerator, denominator = ratios["token_objective"]
        torch.testing.assert_close(numerator / denominator, (0.1 * ce + 0.9 * l1) / den)
        self.assertIn("position_1/teacher/unary_distribution_overlap", ratios)
        # The teacher overlap diagnostic is 1 - L1 / 2 per position.
        overlap_num, overlap_den = ratios["dflash/teacher/unary_distribution_overlap"]
        torch.testing.assert_close(overlap_num / overlap_den, 1.0 - l1 / den / 2.0)
        loss.backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_resume_contract_records_weights_only_off_default(self):
        draft = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=1), target_layer_ids=[1]
        )
        plain = resume_contract(None, draft, _model())
        self.assertNotIn("dflash_l1_loss_alpha", plain)
        weighted = resume_contract(
            None, draft, _model(ce_loss_alpha=0.1, l1_loss_alpha=0.9)
        )
        self.assertEqual(weighted["dflash_ce_loss_alpha"], 0.1)
        self.assertEqual(weighted["dflash_l1_loss_alpha"], 0.9)


@unittest.skipUnless(
    torch.cuda.is_available() and _triton_available(), "requires CUDA and Triton"
)
class DFlashL1ObjectiveCudaTest(unittest.TestCase):
    """CUDA: BF16 logits take the shared fused TV kernel and match eager."""

    def test_bf16_chunk_terms_take_the_fused_kernel_and_match_eager(self):
        torch.manual_seed(0)
        hidden_size, vocab_size = 8, 4099  # a masked tail for the kernel
        model = _model(ce_loss_alpha=0.1, l1_loss_alpha=0.9)
        model.lm_head = nn.Linear(hidden_size, vocab_size, bias=False).to(
            "cuda", torch.bfloat16
        )
        model.lm_head.weight.requires_grad_(False)
        hidden = (torch.randn(1, 2, 3, hidden_size, device="cuda") * 3).bfloat16()
        teacher = (torch.randn(1, 2, 3, hidden_size, device="cuda") * 3).bfloat16()
        target_ids = torch.randint(0, vocab_size, (1, 2, 3), device="cuda")
        weights = torch.tensor([[[0.0, 1.0, 1.0], [0.0, 1.0, 0.5]]], device="cuda")

        def run(fused):
            draft = hidden.clone().requires_grad_()
            with (
                mock.patch.dict(
                    os.environ, {token_objective.FUSED_TEACHER_TV_ENV: fused}
                ),
                mock.patch.object(
                    fused_loss, "fused_tv_loss", wraps=fused_loss.fused_tv_loss
                ) as kernel,
            ):
                terms = model._dflash_objective_chunk_terms(
                    draft, target_ids, weights, target_ids, None, teacher
                )
            self.assertEqual(kernel.called, fused == "1")
            terms.teacher_l1_num.backward()
            return terms.teacher_l1_num.detach().float(), draft.grad.float()

        fused_l1, fused_grad = run("1")
        eager_l1, eager_grad = run("0")
        self.assertGreater(eager_l1.item(), 0.0)
        torch.testing.assert_close(fused_l1, eager_l1, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(fused_grad, eager_grad, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
