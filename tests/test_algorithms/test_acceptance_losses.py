"""Acceptance losses preserve legacy reduction and complete-chain weighting."""

import functools
import unittest
from dataclasses import replace

import torch

from specforge.algorithms.objectives import losses
from specforge.algorithms.objectives.execution import (
    chain_weights,
    execute_batch,
    project_predictions,
)
from specforge.algorithms.objectives.losses import (
    LossContext,
    PositionWeighting,
    TokenLoss,
    register_loss,
)
from specforge.algorithms.objectives.predictions import (
    PredictionBatch,
    Predictions,
    Teacher,
)
from specforge.algorithms.objectives.projection import Projection
from specforge.losses.lk import compute_lk_loss
from specforge.losses.position_weights import DPACE_LOSS_TYPES, dpace_weights
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
                    runtime, legacy, names = block_pair(
                        "dflash",
                        loss={
                            "terms": {
                                "token": {
                                    "kind": f"lk_{mode}",
                                    "options": (
                                        {"kl_scale": 0.7, "kl_decay": 1.3}
                                        if mode == "lambda"
                                        else {}
                                    ),
                                }
                            }
                        },
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
                    options={"target": "teacher"}
                    | ({"kl_scale": 0.7, "kl_decay": 1.3} if mode == "lambda" else {}),
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
            projection=Projection(lambda x: {"logits": x}),
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

    def test_loss_options_are_checked_when_the_loss_is_built(self):
        TokenLoss(kind="lk", options={"eta": 2.0})
        TokenLoss(kind="ce_l1", options={"ce_weight": 0.2, "l1_weight": 0.8})
        TokenLoss(kind="lk_lambda", options={"kl_scale": 0.5, "kl_decay": 2.0})
        for kind, options in (
            ("ce_l1", {"ce_wieght": 0.2}),
            ("kl", {"eta": 2.0}),
            ("default", {"ce_weight": 0.2}),
            ("lk_tv", {"kl_scale": 0.5}),
            ("lk_alpha", {"kl_decay": 2.0}),
        ):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                TokenLoss(kind=kind, options=options)

    def test_registered_losses_keep_their_own_options(self):
        def temperature_ce(logits, labels, teacher=None, **options):
            return logits.new_zeros(labels.shape) + options["temperature"]

        def scaled(compute):
            @functools.wraps(compute)
            def loss(logits, labels, teacher=None, *, scale=1.0):
                return scale * compute(logits, labels, teacher, temperature=1.0)

            return loss

        register_loss("test_temperature_ce")(temperature_ce)
        register_loss("test_scaled_ce")(scaled(temperature_ce))
        try:
            labels = torch.zeros(1, 2, dtype=torch.long)
            for kind, options, expected in (
                ("test_temperature_ce", {"temperature": 2.0}, 2.0),
                ("test_scaled_ce", {"scale": 3.0}, 3.0),
            ):
                with self.subTest(kind=kind):
                    loss = TokenLoss(kind=kind, options=options)
                    values = loss(torch.zeros(1, 2, 3), labels)
                    self.assertTrue(bool((values == expected).all()))
            with self.assertRaises(ValueError):
                TokenLoss(kind="test_scaled_ce", options={"scael": 3.0})
        finally:
            losses._LOSSES.pop("test_temperature_ce")
            losses._LOSSES.pop("test_scaled_ce")

    def test_label_acceptance_is_normalized_by_supervised_positions(self):
        logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.0, 0.0]]])
        labels = torch.tensor([[0, 1, 0, 0]])
        mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        acceptance = logits.softmax(-1).gather(-1, labels[..., None]).squeeze(-1)
        context = LossContext(mask, None, torch.tensor(4.0))
        tv = TokenLoss(kind="lk_tv")(logits, labels, context=context)
        torch.testing.assert_close(tv, (1 - acceptance) * 2)

    def test_geometric_zero_factor_keeps_the_anchor_slot_finite(self):
        weights = PositionWeighting(kind="geometric", factor=0.0)(
            torch.tensor([0.0, 1.0, 1.0]), torch.tensor([-1, 0, 1])
        )
        self.assertTrue(torch.isfinite(weights).all())

    def test_dpace_ends_chains_at_targets_outside_the_draft_vocabulary(self):
        """A teacher top-1 the draft cannot propose stops the acceptance chain."""
        teacher = torch.full((1, 3, 4), -5.0)
        teacher[0, 0, 0] = teacher[0, 1, 3] = teacher[0, 2, 1] = 5.0
        prediction = Predictions(
            hidden=torch.zeros(1, 3, 3),
            projection=Projection(lambda hidden: {"logits": hidden}),
            mask=torch.ones(1, 3),
            positions=torch.arange(3)[None],
            chain_ids=torch.zeros(1, 3, dtype=torch.long),
            sequence_ids=torch.zeros(1, 3, dtype=torch.long),
            teacher=Teacher(teacher),
            vocab_mask=torch.tensor([True, True, True, False]),
            requires=frozenset({"teacher_logits"}),
            default_loss=TokenLoss(kind="soft_ce"),
        )
        for kind in DPACE_LOSS_TYPES:
            with self.subTest(kind=kind):
                weighting = PositionWeighting(kind=kind, alpha=0.0)
                projected = project_predictions(prediction, TokenLoss(), weighting)
                [(weights, _)] = chain_weights([prediction], [projected], weighting)
                self.assertGreater(weights[0].item(), 0)
                self.assertEqual(weights[1:].tolist(), [0.0, 0.0])

    def test_dpace_orders_each_chain_by_position(self):
        torch.manual_seed(13)
        prediction = Predictions(
            hidden=torch.randn(3, 3, 5),
            projection=Projection(lambda x: {"logits": x}),
            mask=torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]]),
            positions=torch.arange(3).expand(3, 3),
            labels=torch.arange(9).reshape(3, 3) % 5,
            chain_ids=torch.arange(3)[:, None].expand(3, 3),
            sequence_ids=torch.tensor([0, 0, 1])[:, None].expand(3, 3),
        )
        fields = ("hidden", "labels", "mask", "positions", "chain_ids", "sequence_ids")
        depths = [
            replace(
                prediction, **{n: getattr(prediction, n)[:, d : d + 1] for n in fields}
            )
            for d in range(3)
        ]
        weighting = PositionWeighting(kind="dpace", alpha=0.3)
        results = [
            execute_batch(
                PredictionBatch(sets), TokenLoss(kind="ce"), weighting, training=False
            )
            for sets in (depths, depths[::-1])
        ]
        torch.testing.assert_close(results[1].loss, results[0].loss)
