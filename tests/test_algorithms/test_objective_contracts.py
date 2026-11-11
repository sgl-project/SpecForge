"""Regression checks for canonical objective alignment and reduction contracts."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from torch import nn

from specforge.algorithms.objectives import build_objective_model
from specforge.algorithms.objectives.base import ObjectiveResult, microbatch_mean
from specforge.algorithms.objectives.dflash2 import DFlash2Objective
from specforge.algorithms.objectives.dspark import DSparkObjective
from specforge.algorithms.objectives.execution import evaluate_predictions
from specforge.algorithms.objectives.inputs import prepare_blocks
from specforge.algorithms.objectives.losses import PositionWeighting, TokenLoss
from specforge.algorithms.objectives.predictions import Predictions
from specforge.modeling import build_draft_model
from tests.test_algorithms.test_objectives import (
    capture_projections,
    configurations,
    dflash2_config,
    dspark_config,
    dspine_config,
    eagle_config,
    peagle_config,
    training_batch,
)
from tests.test_modeling.test_canonical_eagle3 import install_mapping


def build_runtime(architecture, *, dtype=None, **selection):
    config = next(c for c in configurations() if c.architecture == architecture)
    torch.manual_seed(0)
    model = build_draft_model(config, **({"dtype": dtype} if dtype else {}))
    if hasattr(model, "vocabulary"):
        install_mapping(model)
    for name in config.target_bindings:
        model.get_submodule(name).load_target_weight(
            torch.randn(config.vocab_size, config.hidden_size)
        )
    if hasattr(model, "initialize_transfer_space"):
        model.initialize_transfer_space()
    width = training_batch(config)["target_last_hidden_states"].shape[-1]
    head = nn.Linear(width, config.vocab_size, bias=False)
    return config, build_objective_model(model, target_head=head, **selection)


class ObjectiveContractTest(unittest.TestCase):
    def test_fractional_denominator_and_microbatch_mean(self):
        first = ObjectiveResult(torch.tensor(0.6), torch.tensor(0.2))
        second = ObjectiveResult(torch.tensor(10.0), torch.tensor(2.0))
        self.assertAlmostEqual(first.loss.item(), 3)
        means = [microbatch_mean(x) for x in (first, second)]
        self.assertAlmostEqual(
            (
                sum(x.numerator for x in means) / sum(x.denominator for x in means)
            ).item(),
            4,
        )
        self.assertEqual(
            ObjectiveResult(torch.tensor(0.0), torch.tensor(0.0)).loss.item(), 0
        )

    def test_selector_schedule_and_disabled_parameters(self):
        objective = DFlash2Objective(selector_warmup_ratio=0.2, selector_ramp_ratio=0.3)
        self.assertEqual(
            [objective.selector_scale(i, 10) for i in range(6)],
            [0, 0, 1 / 3, 2 / 3, 1, 1],
        )
        with self.assertRaisesRegex(ValueError, "steps or ratio"):
            DFlash2Objective(selector_ramp_steps=2, selector_ramp_ratio=0.2)
        model = build_draft_model(dflash2_config())
        build_objective_model(model, options={"selector_loss_alpha": 0})
        self.assertFalse(
            any(p.requires_grad for p in model.candidate_selector.parameters())
        )

    def test_freezing_failure_does_not_mutate_parameters(self):
        model = build_draft_model(dflash2_config())
        before = [p.requires_grad for p in model.parameters()]
        with self.assertRaisesRegex(ValueError, "unknown frozen"):
            build_objective_model(model, frozen=("fusion", "missing"))
        self.assertEqual(before, [p.requires_grad for p in model.parameters()])
        config = dspine_config()
        config = replace(config, target_bindings=("lm_head",))
        model = build_draft_model(config)
        with self.assertRaisesRegex(ValueError, "embed_tokens"):
            build_objective_model(model, frozen=("fusion",))
        self.assertTrue(all(p.requires_grad for p in model.fusion.parameters()))
        build_objective_model(model, frozen=("embed_tokens",))

    def test_capture_shapes_and_teacher_precision(self):
        model = build_draft_model(dflash2_config(), dtype=torch.bfloat16)
        runtime = build_objective_model(model, target_head=nn.Linear(32, 32))
        batch = training_batch(model.config.draft)
        for field in (
            "hidden_states",
            "target_last_hidden_states",
            "mtp_hidden_states",
            "target_logits",
        ):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                runtime(
                    {**batch, field: torch.zeros(1, 8, 32)}, step=0, total_steps=None
                )
        with (
            patch.object(runtime, "objective") as objective,
            patch("specforge.algorithms.objectives.execution.execute_batch"),
        ):
            runtime(batch, step=0, total_steps=None)
            captured = objective.call_args.args[1]
            self.assertEqual(captured["hidden_states"].dtype, torch.bfloat16)
            self.assertEqual(captured["target_last_hidden_states"].dtype, torch.float32)
            teacher = runtime.predictions(
                torch.zeros(1, 1, 32, dtype=torch.bfloat16),
                captured,
                torch.tensor([[1]]),
                mask=torch.ones(1, 1),
                labels=torch.zeros(1, 1, dtype=torch.long),
            ).teacher
            self.assertEqual(teacher.values.dtype, torch.float32)
            self.assertEqual(teacher.logits().dtype, torch.float32)

    def test_dspark_stops_at_unsupervised_gap(self):
        config = dspark_config()
        batch = training_batch(config)
        batch["loss_mask"] = torch.tensor([[1, 1, 0, 1, 0, 0, 0, 0, 0]])
        runtime = build_objective_model(
            build_draft_model(config),
            options={"num_anchors": 1},
            target_head=nn.Linear(32, config.vocab_size),
        )
        captured = []

        def prepare(*args, **kwargs):
            blocks = prepare_blocks(*args, **kwargs)
            captured.append(blocks)
            return blocks

        with patch(
            "specforge.algorithms.objectives.dspark.prepare_blocks", side_effect=prepare
        ):
            result = runtime(batch, step=0, total_steps=None)
        self.assertEqual(captured[0].weights.flatten().tolist(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(result.denominator.item(), 1)
        self.assertIn("acc", result.ratios)

    def test_common_loss_and_weighting_keep_selector(self):
        config = dflash2_config()
        runtime = build_objective_model(
            build_draft_model(config),
            options={"num_anchors": 2},
            loss={"kind": "kl"},
            weighting={"kind": "exponential", "gamma": 2},
        )
        batch = training_batch(config)
        batch["target_logits"] = torch.randn(1, 9, config.vocab_size)
        captured = capture_projections(runtime.draft_model)
        result = runtime(batch, step=0, total_steps=None)
        result.loss.backward()
        self.assertIn("target_last_hidden_states", runtime.registration.inputs)
        self.assertTrue(
            all(
                p.grad is not None
                for p in runtime.draft_model.candidate_selector.parameters()
            )
        )
        logits = captured[0]["logits"]
        teacher = torch.randn_like(logits)
        expected = torch.nn.functional.kl_div(
            logits.float().log_softmax(-1),
            teacher.float().softmax(-1),
            reduction="none",
        ).sum(-1)
        torch.testing.assert_close(
            runtime.token_loss(logits, logits.argmax(-1), teacher=teacher), expected
        )
        torch.testing.assert_close(
            runtime.weighting(torch.ones(4), torch.arange(4)),
            torch.exp(-torch.arange(4) / 2),
        )

    def test_eagle_positions_and_full_vocabulary_accuracy(self):
        config = eagle_config()
        model = build_draft_model(config)
        install_mapping(model)
        batch = training_batch(config)
        batch["position_ids"] = torch.arange(9)[None] * 2 + 17
        batch["target_logits"] = torch.zeros(1, 9, config.vocab_size)
        batch["target_logits"][..., 1] = 20  # Outside the even-ID draft vocabulary.
        runtime = build_objective_model(model, options={"ttt_length": 2})
        entries = []
        model.register_forward_pre_hook(lambda m, args: entries.append(args[2][0]))
        result = runtime(batch, step=0, total_steps=None)
        for depth, entry in enumerate(entries):
            torch.testing.assert_close(
                entry.position_ids, batch["position_ids"] + depth
            )
            torch.testing.assert_close(
                entry.attention_layout.query_positions, torch.arange(9)[None] + depth
            )
        self.assertEqual(result.ratios["acc"][0].item(), 0)
        self.assertEqual(result.ratios["acc"][1].item(), 6)
        self.assertEqual(result.denominator.item(), 1)
        with self.assertRaisesRegex(ValueError, "unpacked"):
            runtime(
                {**batch, "lengths": torch.tensor([4, 5])}, step=0, total_steps=None
            )

    def test_option_bounds(self):
        for config, options in (
            (peagle_config(), {"down_sample_ratio": 0}),
            (peagle_config(), {"down_sample_ratio": 0.1, "down_sample_ratio_min": 0.2}),
            (dspine_config(), {"candidate_mass_threshold": 1}),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                build_objective_model(build_draft_model(config), options=options)
        with self.assertRaises(ValueError):
            PositionWeighting(kind="exponential")

    def test_dspine_eval_disables_replacement(self):
        config = dspine_config()
        model = build_draft_model(config)
        for name in config.target_bindings:
            model.get_submodule(name).load_target_weight(
                torch.randn(config.vocab_size, config.hidden_size)
            )
        model.initialize_transfer_space()
        runtime = build_objective_model(
            model,
            options={"num_anchors": 1},
            target_head=nn.Linear(32, config.vocab_size),
        )
        model.eval()  # Existing trainers change draft mode, not wrapper mode.
        with patch(
            "torch.rand_like", side_effect=AssertionError("replacement sampled in eval")
        ):
            result = runtime(training_batch(config), step=0, total_steps=None)
        self.assertTrue(torch.isfinite(result.loss))

    def test_registry_agreement(self):
        from specforge.algorithms.objectives import objective_registrations
        from specforge.config.draft.registry import _SPECS
        from specforge.modeling.draft.models.registry import _MODELS

        architectures = {
            architecture
            for registration in objective_registrations().values()
            for architecture in registration.architectures
        }
        self.assertEqual(architectures, set(_SPECS))
        self.assertEqual(architectures, set(_MODELS))

    def test_import_without_optional_kernels(self):
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-m", "tests.test_algorithms.objective_import_check"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_peagle_excludes_cross_document_labels(self):
        config = peagle_config()
        model = build_draft_model(config)
        install_mapping(model)
        batch = {key: value[:, :6] for key, value in training_batch(config).items()}
        batch.update(
            loss_mask=torch.ones(1, 6),
            attention_mask=torch.ones(1, 6),
            lengths=torch.tensor([3, 3]),
        )
        teacher = torch.zeros(1, 6, config.vocab_size)
        teacher[..., 2] = 10
        batch["target_logits"] = teacher
        captured = capture_projections(model)
        runtime = build_objective_model(model, options={"num_depths": 1})
        result = runtime(batch, step=0, total_steps=None)
        probabilities = teacher[..., model.vocabulary.t2d].softmax(-1)
        ce = -(
            probabilities * captured[0]["logits"].reshape(1, 6, -1).log_softmax(-1)
        ).sum(-1)
        torch.testing.assert_close(result.loss, ce[:, [0, 1, 3, 4]].mean())
        self.assertEqual(result.denominator.item(), 1)

    def test_registered_loss_extends_existing_algorithm(self):
        from specforge.algorithms.objectives import register_loss
        from specforge.algorithms.objectives.losses import _LOSSES

        with patch.dict(_LOSSES):

            @register_loss("custom", requires=("logits", "labels", "teacher_logits"))
            def custom(logits, labels, teacher):
                return (logits.float() - teacher).square().mean(-1)

            config = dflash2_config()
            runtime = build_objective_model(
                build_draft_model(config),
                options={"num_anchors": 1},
                loss={"kind": "custom"},
            )
            batch = training_batch(config)
            batch["target_logits"] = torch.randn(1, 9, config.vocab_size)
            runtime(batch, step=0, total_steps=None).loss.backward()
            self.assertIn("target_last_hidden_states", runtime.registration.inputs)
            self.assertTrue(
                all(
                    p.grad is not None
                    for p in runtime.draft_model.candidate_selector.parameters()
                )
            )
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_loss("custom")(custom)

    def test_explicit_weighting_replaces_eagle_default(self):
        config = eagle_config()
        model = build_draft_model(config)
        install_mapping(model)
        batch = training_batch(config)
        batch["target_logits"] = torch.randn(1, 9, config.vocab_size)
        runtime = build_objective_model(model, options={"ttt_length": 3})
        projected = capture_projections(model)
        default_loss = runtime(batch, step=0, total_steps=None).loss
        expected = []
        valid = batch["loss_mask"].bool() & batch["attention_mask"].bool()
        for depth, output in enumerate(projected):
            positions = torch.arange(9)[None] + depth + 1
            safe = positions.clamp(max=8)
            mask = (
                valid.gather(1, safe)
                & (positions < 9)
                & model.vocabulary.t2d[
                    batch["target_logits"].argmax(-1).gather(1, safe)
                ]
            )
            teacher = batch["target_logits"][:, safe[0], :][
                ..., model.vocabulary.t2d
            ].softmax(-1)
            logits = output["logits"].reshape(1, 9, -1)
            ce = -(teacher * logits.log_softmax(-1)).sum(-1)
            expected.append((ce * mask).sum() / 9)
        torch.testing.assert_close(
            default_loss, sum(0.8**i * term for i, term in enumerate(expected))
        )
        for weighting in (
            {"kind": "uniform"},
            {"kind": "geometric", "factor": 0.5},
            {"kind": "exponential", "gamma": 2},
        ):
            with self.subTest(weighting=weighting):
                configured = build_objective_model(
                    model, options={"ttt_length": 3}, weighting=weighting
                )
                result = configured(batch, step=0, total_steps=None)
                factors = configured.weighting(torch.ones(3), torch.arange(3))
                torch.testing.assert_close(
                    result.loss, (torch.stack(expected) * factors).sum()
                )

    def test_mtp_rejects_position_weightings(self):
        """MTP proposes one token per position; decay or D-PACE has nothing to act on."""
        for weighting in (
            {"kind": "exponential", "gamma": 1.0},
            {"kind": "geometric", "factor": 0.5},
            {"kind": "dpace"},
        ):
            with self.subTest(weighting=weighting["kind"]):
                with self.assertRaisesRegex(ValueError, "capabilities"):
                    build_runtime("MTPDraftModel", weighting=weighting)
        config, default = build_runtime("MTPDraftModel")
        _, uniform = build_runtime("MTPDraftModel", weighting={"kind": "uniform"})
        batch = training_batch(config)
        torch.testing.assert_close(
            uniform(batch, step=0, total_steps=1).loss,
            default(batch, step=0, total_steps=1).loss,
        )

    def test_eagle_legacy_denominator_is_fp32_under_a_bf16_draft(self):
        config, runtime = build_runtime("Eagle3DraftModel", dtype=torch.bfloat16)
        batch = training_batch(config)
        del batch["mtp_hidden_states"]
        length = 257  # not representable in bf16
        batch["input_ids"] = torch.arange(length)[None] % config.vocab_size
        for key in ("loss_mask", "attention_mask"):
            batch[key] = torch.ones(1, length, dtype=torch.long)
        for key in ("hidden_states", "target_last_hidden_states"):
            batch[key] = torch.randn(1, length, batch[key].shape[-1])
        batch["hidden_states"] = batch["hidden_states"].to(torch.bfloat16)
        predictions = next(runtime.objective.rollout(runtime, batch))
        self.assertEqual(predictions.denominator.dtype, torch.float32)
        self.assertEqual(predictions.denominator.item(), length)

    def test_accuracy_counts_supervised_positions_unweighted(self):
        """Position weights shape the loss, not acc, which picks the best checkpoint."""
        logits = torch.full((1, 3, 5), -5.0)
        logits[0, 0, 4] = logits[0, 1, 1] = logits[0, 2, 2] = 5.0
        predictions = Predictions(
            hidden=torch.zeros(1, 3, 2),
            projector=lambda hidden: {"logits": logits},
            mask=torch.ones(1, 3),
            positions=torch.tensor([[0, 1, 2]]),
            labels=torch.tensor([[3, 1, 2]]),
        )
        for weighting in (
            PositionWeighting(kind="uniform"),
            PositionWeighting(kind="exponential", gamma=1.0),
            PositionWeighting(kind="geometric", factor=0.5),
        ):
            with self.subTest(weighting=weighting.kind):
                correct, total = evaluate_predictions(
                    predictions, TokenLoss(), weighting
                ).ratios["acc"]
                self.assertEqual((correct.item(), total.item()), (2.0, 3.0))

    def test_explicit_selection_rejects_options_it_would_ignore(self):
        build_runtime("DSpineDraftModel", weighting={"kind": "uniform"})
        build_runtime("DSparkDraftModel", loss={"kind": "kl"})
        build_runtime(
            "DSpineDraftModel", options={"ce_weight": 0.2}, loss={"kind": "kl"}
        )
        build_runtime(
            "DFlashDraftModel",
            options={"loss_decay_gamma": None},
            weighting={"kind": "uniform"},
        )
        uniform, kl = {"weighting": {"kind": "uniform"}}, {"loss": {"kind": "kl"}}
        for architecture, options, selection in (
            ("DFlashDraftModel", {"loss_decay_gamma": 7.0}, uniform),
            ("DSpineDraftModel", {"loss_decay_gamma": 7.0}, uniform),
            ("Eagle3DraftModel", {"depth_decay": 0.1}, uniform),
            ("DSparkDraftModel", {"ce_loss_alpha": 0.1}, kl),
        ):
            with self.subTest(architecture=architecture, options=options):
                with self.assertRaisesRegex(ValueError, next(iter(options))):
                    build_runtime(architecture, options=options, **selection)

    def test_dspark_zero_confidence_weight_freezes_the_head(self):
        _, runtime = build_runtime(
            "DSparkDraftModel", options={"confidence_loss_alpha": 0.0}
        )
        head = runtime.draft_model.confidence_head
        self.assertFalse(any(p.requires_grad for p in head.parameters()))
        self.assertEqual(runtime.frozen_parameters, ("confidence_head",))
        config = dspark_config(confidence=False)
        model = build_draft_model(config)
        for name in config.target_bindings:
            model.get_submodule(name).load_target_weight(
                torch.randn(config.vocab_size, config.hidden_size)
            )
        runtime = build_objective_model(model, options={"confidence_loss_alpha": 0.0})
        self.assertEqual(runtime.frozen_parameters, ())
        misspelled = property(lambda objective: ("confidence_haed",))
        with patch.object(DSparkObjective, "frozen_parameters", misspelled):
            with self.assertRaisesRegex(ValueError, "confidence_haed"):
                build_runtime("DSparkDraftModel")

    def test_dspark_projection_requires_integer_predecessors(self):
        config = dspark_config()
        model = build_draft_model(config)
        slots = config.spec_config.protocol.num_model_slots
        hidden = torch.zeros(1, slots, config.hidden_size)
        for predecessors in (None, torch.zeros(1, slots)):
            with self.assertRaisesRegex(ValueError, "predecessor_ids"):
                model.project(hidden, predecessor_ids=predecessors)

    def test_peagle_accepts_two_dimensional_document_lengths(self):
        config = peagle_config()
        losses = []
        for lengths in (torch.tensor([3, 3]), torch.tensor([[3, 3]])):
            torch.manual_seed(0)
            model = build_draft_model(config)
            install_mapping(model)
            batch = {k: v[:, :6] for k, v in training_batch(config).items()}
            batch.update(
                loss_mask=torch.ones(1, 6),
                attention_mask=torch.ones(1, 6),
                lengths=lengths,
                target_logits=torch.randn(1, 6, config.vocab_size),
            )
            runtime = build_objective_model(model, options={"num_depths": 1})
            losses.append(runtime(batch, step=0, total_steps=None).loss)
        torch.testing.assert_close(losses[0], losses[1])
