"""D-PAL uses distribution overlap for credit and recorded tokens for CE."""

import unittest

import torch

from specforge.algorithms.objectives import resolve_objective
from specforge.algorithms.objectives.execution import execute_batch
from specforge.algorithms.objectives.execution_options import LossExecution
from specforge.algorithms.objectives.losses import PositionWeighting, TokenLoss
from specforge.algorithms.objectives.predictions import (
    PredictionBatch,
    Predictions,
    Teacher,
)
from specforge.algorithms.objectives.projection import Projection
from tests.test_algorithms.test_objective_contracts import build_runtime
from tests.test_algorithms.test_objectives import training_batch


class DPALTest(unittest.TestCase):
    def test_actor_gradient_and_sequence_denominator(self):
        torch.manual_seed(7)
        logits = torch.randn(4, 3, 5, requires_grad=True)
        teacher = torch.randn_like(logits)
        labels = torch.randint(5, (4, 3))
        mask = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 0]])
        sequence = torch.tensor([0, 0, 1, 2])
        prediction = Predictions(
            hidden=logits,
            projection=Projection(lambda x: {"logits": x}),
            teacher=Teacher(teacher),
            labels=labels,
            mask=mask,
            positions=torch.arange(3).expand_as(labels),
            chain_ids=torch.arange(4)[:, None].expand_as(labels),
            sequence_ids=sequence[:, None].expand_as(labels),
        )
        p, q = teacher.softmax(-1), logits.softmax(-1)
        overlap = torch.minimum(p, q).sum(-1)
        weights = torch.zeros_like(overlap)
        for i, row in enumerate(mask):
            prefix = 1
            for j in range(len(row)):
                if row[j]:
                    prefix *= 0.3 + 0.7 * overlap[i, j].detach().item()
                    weights[i, : j + 1] += prefix * row[: j + 1]
        weights[:2] /= 2  # sequence 0 has two valid anchors
        ce = -q.log().gather(-1, labels[..., None]).squeeze(-1)
        for loss, actor in (
            (TokenLoss("kl"), (p * (p.log() - q.log())).sum(-1)),
            (TokenLoss("neg_log_overlap"), -overlap.log()),
            (TokenLoss("renyi_half"), -2 * (p * q).sqrt().sum(-1).log()),
            (
                TokenLoss("ce_l1", {"ce_weight": 0.1, "l1_weight": 0.45}),
                0.1 * ce + 0.45 * (p - q).abs().sum(-1),
            ),
        ):
            reference = (actor * weights).sum() / 2
            (grad,) = torch.autograd.grad(reference, logits, retain_graph=True)
            for size in (0, 1):
                with self.subTest(kind=loss.kind, chunk=size):
                    actual = execute_batch(
                        PredictionBatch([prediction]),
                        loss,
                        PositionWeighting("dpal"),
                        training=False,
                        execution=LossExecution(chunk_size=size),
                    )
                    torch.testing.assert_close(actual.denominator, torch.tensor(2.0))
                    torch.testing.assert_close(actual.loss, reference)
                    torch.testing.assert_close(
                        torch.autograd.grad(actual.loss, logits)[0], grad
                    )

    def test_extreme_distributions(self):
        logits = torch.tensor([[1000.0, -1000.0]], requires_grad=True)
        teacher = torch.tensor([[-float("inf"), 0.0]])
        labels = torch.tensor([1])
        for kind in ("kl", "neg_log_overlap", "renyi_half"):
            loss = TokenLoss(kind)(logits, labels, teacher=teacher)
            torch.testing.assert_close(loss, torch.tensor([2000.0]))
            torch.testing.assert_close(
                torch.autograd.grad(loss.sum(), logits)[0], torch.tensor([[1.0, -1.0]])
            )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_kl_with_zero_teacher_mass(self):
        from specforge.losses.fused import fused_kl_div_loss, kl_div_loss

        logits = torch.randn(1, 2, 65537, device="cuda", requires_grad=True)
        teacher = torch.full_like(logits, -float("inf"))
        teacher[..., -1] = 0
        reference = kl_div_loss(logits, teacher)
        actual = fused_kl_div_loss(logits, teacher)
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), logits)[0],
            torch.autograd.grad(reference.sum(), logits)[0],
        )

    def test_cpu_training(self):
        self.check_forward("cpu", torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_bf16_training(self):
        self.check_forward("cuda", torch.bfloat16)

    def check_forward(self, device, dtype):
        cfg, runtime = build_runtime(
            "DFlashDraftModel",
            options={"teacher_metrics": True, "num_anchors": 3},
            loss={
                "terms": {"token": {"kind": "renyi_half"}},
                "execution": {"chunk_size": 1},
            },
            weighting={"kind": "dpal"},
        )
        runtime.to(device=device, dtype=dtype)
        batch = training_batch(cfg, device)
        result = runtime(batch, step=0, total_steps=1, collect_detailed_metrics=True)
        result.loss.backward()
        self.assertIn("renyi_half_loss", result.ratios)
        self.assertTrue(torch.isfinite(result.loss))
        self.assertTrue(
            any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in runtime.draft_model.parameters()
            )
        )
        self.assertIsNone(batch["target_last_hidden_states"].grad)

    def test_incompatible_selections(self):
        with self.assertRaisesRegex(ValueError, "overlap_chain"):
            resolve_objective("DFlash2DraftModel", weighting={"kind": "dpal"})
        with self.assertRaisesRegex(ValueError, "LK"):
            resolve_objective(
                "DFlashDraftModel",
                loss={"terms": {"token": {"kind": "lk"}}},
                weighting={"kind": "dpal"},
            )
