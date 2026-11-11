"""Acceptance losses preserve legacy reduction and complete-chain weighting."""

import unittest
from dataclasses import replace

import torch

from specforge.algorithms.objectives.execution import execute_batch
from specforge.algorithms.objectives.losses import (
    LossContext,
    PositionWeighting,
    TokenLoss,
)
from specforge.algorithms.objectives.predictions import PredictionBatch, Predictions
from specforge.core.lk_loss import compute_lk_loss
from specforge.core.position_weights import DPACE_LOSS_TYPES, dpace_weights
from tests.test_algorithms.test_objective_legacy_parity import (
    assert_loss_and_gradients,
    block_pair,
)
from tests.test_algorithms.test_objectives import training_batch
from tests.test_modeling.legacy_rope_reference import legacy_fp32_rope


class AcceptanceLossTest(unittest.TestCase):
    @legacy_fp32_rope()
    def test_dflash_lk_and_dpace_match_legacy(self):
        for weighting in ("uniform", *sorted(DPACE_LOSS_TYPES)):
            for mode in ("alpha", "tv", "lambda"):
                with self.subTest(weighting=weighting, mode=mode):
                    torch.manual_seed(7)
                    runtime, legacy, names = block_pair("dflash")
                    runtime.token_loss = TokenLoss(
                        kind=f"lk_{mode}", options={"kl_scale": 0.7, "kl_decay": 1.3}
                    )
                    runtime.weighting = PositionWeighting(kind=weighting)
                    legacy.lk_loss_type, legacy.kl_scale, legacy.kl_decay = (
                        mode,
                        0.7,
                        1.3,
                    )
                    legacy.loss_type = "dflash" if weighting == "uniform" else weighting
                    batch = {
                        k: v.repeat(2, *([1] * (v.ndim - 1)))
                        for k, v in training_batch(
                            runtime.draft_model.config.draft
                        ).items()
                    }
                    batch["loss_mask"][1, 3:] = 0  # unequal valid anchor counts
                    args = {
                        k: batch[k] for k in ("input_ids", "hidden_states", "loss_mask")
                    }
                    torch.manual_seed(11)
                    actual = runtime(batch, step=1, total_steps=10).loss
                    torch.manual_seed(11)
                    expected = legacy(**args, collect_detailed_metrics=False)[0]
                    assert_loss_and_gradients(runtime, legacy, actual, expected, names)

    def test_teacher_lk_preserves_vocabulary_mass_and_normalization(self):
        torch.manual_seed(9)
        logits = torch.randn(2, 3, 4, requires_grad=True)
        teacher = torch.randn(2, 3, 7)
        restricted = teacher[..., :4]
        p = teacher.softmax(-1)[..., :4]
        mass = p.sum(-1)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        denominator = torch.tensor(6.0)
        for mode in ("alpha", "tv", "lambda"):
            with self.subTest(mode=mode):
                loss = TokenLoss(
                    kind=f"lk_{mode}",
                    options={"target": "teacher", "kl_scale": 0.7, "kl_decay": 1.3},
                )
                actual = (
                    loss(
                        logits,
                        restricted.argmax(-1),
                        teacher=restricted,
                        context=LossContext(mask, mass, denominator),
                    )
                    * mask
                ).sum() / denominator
                acceptance = torch.minimum(p, logits.softmax(-1)).sum(-1)
                ce = -(restricted.softmax(-1) * logits.log_softmax(-1)).sum(-1)
                expected = compute_lk_loss(
                    kl_loss=(ce * mask).sum() / denominator,
                    acceptance_rate=(acceptance * mask).sum() / mask.sum(),
                    log_acceptance_rate=(acceptance.log() * mask).sum() / mask.sum(),
                    lk_loss_type=mode,
                    kl_scale=0.7,
                    kl_decay=1.3,
                )
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(
                    torch.autograd.grad(actual, logits, retain_graph=True)[0],
                    torch.autograd.grad(expected, logits, retain_graph=True)[0],
                )
                self.assertIn("target_last_hidden_states", loss.inputs)

    def test_dpace_groups_chains_across_prediction_sets(self):
        torch.manual_seed(13)
        logits = torch.randn(3, 3, 5, requires_grad=True)
        labels = torch.arange(9).reshape(3, 3) % 5
        mask = torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]])
        prediction = Predictions(
            hidden=logits,
            projector=lambda x: {"logits": x},
            mask=mask,
            positions=torch.arange(3).expand(3, 3),
            labels=labels,
            chain_ids=torch.arange(3)[:, None].expand(3, 3),
            sequence_ids=torch.tensor([0, 0, 1])[:, None].expand(3, 3),
        )
        # Interleave chains across depth sets, as EAGLE does.
        depths = [
            replace(
                prediction,
                **{
                    name: getattr(prediction, name)[:, d : d + 1]
                    for name in (
                        "hidden",
                        "labels",
                        "mask",
                        "positions",
                        "chain_ids",
                        "sequence_ids",
                    )
                },
            )
            for d in range(3)
        ]
        for kind in DPACE_LOSS_TYPES:
            weighting = PositionWeighting(kind=kind, alpha=0.3)
            actual = execute_batch(
                PredictionBatch(depths), TokenLoss(kind="ce"), weighting, training=False
            )
            ce = -logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
            weights = dpace_weights((-ce).exp(), mask, alpha=0.3, mode=kind) * mask
            expected = ((ce * weights).sum(-1) * torch.tensor([0.5, 0.5, 1])).sum() / 2
            torch.testing.assert_close(actual.loss, expected)
            torch.testing.assert_close(actual.denominator, torch.tensor(2.0))
            torch.testing.assert_close(
                torch.autograd.grad(actual.loss, logits, retain_graph=True)[0],
                torch.autograd.grad(expected, logits, retain_graph=True)[0],
            )
        with self.assertRaisesRegex(ValueError, "unique positions"):
            execute_batch(
                PredictionBatch([depths[0], depths[0]]),
                TokenLoss(kind="ce"),
                weighting,
                training=False,
            )

    def test_dpace_runs_all_protocols(self):
        from torch import nn

        from specforge.algorithms.objectives import build_objective_model
        from specforge.modeling import build_draft_model
        from tests.test_algorithms.test_objectives import configurations

        for config in configurations():
            if config.architecture == "MTPDraftModel":
                continue  # one proposal per position: D-PACE is rejected at build
            with self.subTest(architecture=config.architecture):
                torch.manual_seed(11)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = build_draft_model(config, device=device)
                for name in config.target_bindings:
                    model.get_submodule(name).load_target_weight(
                        torch.randn(
                            config.vocab_size, config.hidden_size, device=device
                        )
                    )
                if hasattr(model, "vocabulary"):
                    model.vocabulary.load_mapping(
                        t2d=torch.arange(config.vocab_size, device=device)
                        < config.draft_vocab_size,
                        d2t=torch.zeros(
                            config.draft_vocab_size, dtype=torch.long, device=device
                        ),
                    )
                if hasattr(model, "initialize_transfer_space"):
                    model.initialize_transfer_space()
                runtime = build_objective_model(
                    model,
                    options={"num_anchors": 2} if config.spec_config.protocol else {},
                    target_head=nn.Linear(32, config.vocab_size, device=device),
                    weighting={"kind": "dpace"},
                )
                result = runtime(training_batch(config, device), step=1, total_steps=10)
                self.assertTrue(torch.isfinite(result.loss))
                result.loss.backward()
                self.assertTrue(
                    all(
                        p.grad is not None and torch.isfinite(p.grad).all()
                        for p in model.parameters()
                        if p.requires_grad
                    )
                )

    def test_option_bounds_and_hybrid_eta(self):
        for kwargs in (
            {"kind": "dpace", "alpha": -0.1},
            {"kind": "uniform", "alpha": 0.5},
        ):
            with self.assertRaises(ValueError):
                PositionWeighting(**kwargs)
        for options in (
            {"target": "other"},
            {"kl_scale": -1},
            {"kl_decay": float("nan")},
            {"unknown": 1},
        ):
            with self.assertRaises(ValueError):
                TokenLoss(kind="lk_lambda", options=options)
        logits, teacher = torch.randn(2, 1, 4), torch.randn(2, 1, 4)
        result = TokenLoss(kind="lk", options={"eta": 2.0})(
            logits, teacher.argmax(-1), teacher=teacher
        )
        self.assertTrue(torch.isfinite(result).all())
