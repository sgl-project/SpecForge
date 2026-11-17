"""Public loss modes preserve the mathematical contract and output requests."""

import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import torch

from specforge.algorithms.objectives.implementations.registry import (
    _IMPLEMENTATIONS,
    ExecutionRequest,
    TermInputs,
    compute_loss,
    implementation_registrations,
    register_implementation,
    select_implementation,
)
from specforge.algorithms.objectives.losses import LossContext, TokenLoss


class LossRoutingTest(unittest.TestCase):

    def test_loss_input_kinds_and_reserved_options_fail_at_construction(self):
        from specforge.algorithms.objectives.losses import _LOSSES, register_loss
        from specforge.algorithms.objectives.registry import resolve_objective

        for kind in ("value", "bce", "cosine"):
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(ValueError, "loss inputs"),
            ):
                resolve_objective("MTPDraftModel", loss={"kind": kind})
        with self.assertRaisesRegex(ValueError, "input kind"):
            register_loss("bad", input_kind="featurez")
        with self.assertRaisesRegex(ValueError, "unknown ce_l1 options"):
            TokenLoss("ce_l1", {"context": None})
        with patch.dict(_LOSSES):

            @register_loss("ext_ctx", contextual=True)
            def loss(logits, labels, teacher, context, temperature):
                return logits[..., 0] / temperature

            request = self.request()
            result = TokenLoss("ext_ctx", {"temperature": 2.0})(
                request.tensor(0), request.labels, context=request.terms[0].context
            )
            torch.testing.assert_close(result, request.tensor(0)[..., 0] / 2)

    def test_undeclared_implementations_do_not_receive_new_request_shapes(self):
        request = self.request()
        with patch.dict(_IMPLEMENTATIONS):
            supports = Mock(side_effect=AssertionError("legacy supports was called"))
            register_implementation("legacy", supports=supports)(
                lambda req, group: None
            )
            with self.assertRaisesRegex(ValueError, "requires CUDA/ROCm"):
                compute_loss(request, fused=True)
            value = replace(
                request,
                terms=(replace(request.terms[0], loss=TokenLoss("value")),),
                tensors={0: torch.ones_like(request.labels, dtype=torch.float32)},
            )
            compute_loss(value, fused=True)
            supports.assert_not_called()

    def test_supervision_only_prepares_requested_statistics(self):
        from specforge.algorithms.objectives.execution import prepare_supervision
        from specforge.algorithms.objectives.execution_options import LossExecution
        from specforge.algorithms.objectives.losses import PositionWeighting
        from specforge.algorithms.objectives.plan import LossPlan, LossTerm
        from specforge.algorithms.objectives.predictions import Predictions, Teacher
        from specforge.algorithms.objectives.projection import Projection
        from tests.test_algorithms.test_loss_plan import run

        labels = torch.zeros(4, 1, dtype=torch.long)
        for kind, vocabulary, calls in (
            ("ce_l1", True, 4),
            ("lk_tv", False, 4),
            ("lk_tv", True, 8),
        ):
            with self.subTest(kind=kind, vocabulary=vocabulary):
                head = Mock(side_effect=lambda hidden: hidden)
                teacher = torch.zeros(4, 1, 6)
                prediction = Predictions(
                    hidden=torch.randn(4, 1, 4 if vocabulary else 6),
                    projection=Projection(lambda x: {"logits": x}),
                    labels=labels,
                    mask=torch.ones_like(labels),
                    positions=labels,
                    teacher=Teacher(teacher, head),
                    requires=frozenset({"teacher_logits"}),
                    vocab_mask=(
                        torch.tensor([True] * 4 + [False] * 2) if vocabulary else None
                    ),
                    denominator=torch.tensor(4.0),
                    plan=LossPlan((LossTerm(loss=TokenLoss(kind)),)),
                )
                supervision = prepare_supervision(
                    prediction, TokenLoss(kind), PositionWeighting(), LossExecution()
                )
                self.assertIsNone(supervision.teacher_mass)
                head.reset_mock()
                with torch.no_grad():
                    run(prediction, chunk=1)
                self.assertEqual(head.call_count, calls)

    def test_computation_rejects_invalid_labels_and_term_execution(self):
        from specforge.algorithms.objectives.plan import LossPlan, LossTerm
        from specforge.algorithms.objectives.predictions import Predictions
        from specforge.algorithms.objectives.projection import Projection
        from tests.test_algorithms.test_loss_plan import run

        request = self.request()
        prediction = Predictions(
            hidden=request.tensor(0),
            projection=Projection(lambda x: {"logits": x}),
            labels=request.labels,
            mask=request.terms[0].mask,
            positions=request.labels,
        )

        def duplicate(calc):
            calc.run()
            calc.run()

        def wrong_labels(calc):
            calc.evaluate(
                "ce", TokenLoss("ce"), calc.outputs()["logits"], calc.labels.flatten()
            )

        for compute, message in (
            (duplicate, "exactly once"),
            (wrong_labels, "labels must align"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                run(replace(prediction, compute=compute))
        scalar = replace(
            prediction,
            plan=LossPlan((LossTerm(loss=TokenLoss("value"), reduction="scalar"),)),
            compute=lambda calc: calc.add(0, calc.predictions.hidden),
        )
        with self.assertRaisesRegex(ValueError, "Scalar loss must return"):
            run(scalar)

    def request(self):
        logits = torch.randn(2, 1, 5, requires_grad=True)
        labels = torch.tensor([[1], [3]])
        weights = torch.ones_like(labels, dtype=torch.float32)
        term = TermInputs(
            TokenLoss("ce"), weights, weights, weights.sum(), LossContext(weights)
        )
        return ExecutionRequest(
            None,
            (term,),
            labels,
            None,
            None,
            {},
            unreduced=True,
            tensors={0: logits},
        )

    def test_modes_and_specialized_contract(self):
        request = self.request()
        reference = select_implementation(request)
        expected = compute_loss(request)
        with patch.dict(_IMPLEMENTATIONS):

            @register_implementation(
                "test_special",
                category="specialized",
                losses=("ce",),
                supports=lambda req, group: (
                    None if group == (0,) else "single term only"
                ),
            )
            def special(req, group):
                return reference.compute(req, group)

            self.assertEqual(
                select_implementation(request, fused=True).name, "test_special"
            )
            actual = compute_loss(request, fused=True)
            torch.testing.assert_close(actual.values, expected.values)
            torch.testing.assert_close(
                torch.autograd.grad(actual.values[0].sum(), request.tensor(0))[0],
                torch.autograd.grad(expected.values[0].sum(), request.tensor(0))[0],
            )
            wrong = replace(
                request, terms=(replace(request.terms[0], loss=TokenLoss("kl")),)
            )
            with self.assertRaisesRegex(ValueError, "requires CUDA/ROCm"):
                select_implementation(wrong, fused=True)
            with self.assertRaises(TypeError):
                compute_loss(request, impl="test_special")

    def test_ce_outputs_share_the_reference_and_token_implementations(self):
        request = self.request()
        request = replace(
            request,
            teacher=torch.randn_like(request.tensor(0)),
            requested_outputs={"return_teacher_l1": True, "top_k": 2},
        )
        result = compute_loss(request)
        torch.testing.assert_close(
            result.values[0], TokenLoss("ce")(request.tensor(0), request.labels)
        )
        self.assertEqual(
            set(result.statistics[0]), {"teacher_l1", "topk_ids", "topk_values"}
        )
        with self.assertRaisesRegex(ValueError, "Unknown CE outputs"):
            compute_loss(replace(request, requested_outputs={"unknown": True}))
        entries = implementation_registrations()
        self.assertEqual(entries["linear_ce"].category, "generic")
        self.assertEqual(entries["domino_suffix_ce"].category, "specialized")
        self.assertEqual(entries["domino_suffix_ce"].losses, ("ce",))
        self.assertNotIn("fused_ce_statistics", entries)

    def test_scalar_regularizer_runs_once_without_trimming(self):
        from specforge.algorithms.objectives.plan import LossPlan, LossTerm
        from specforge.algorithms.objectives.predictions import Predictions
        from specforge.algorithms.objectives.projection import Projection
        from tests.test_algorithms.test_loss_plan import run

        hidden = torch.randn(3, 1, 5, requires_grad=True)
        labels = torch.zeros(3, 1, dtype=torch.long)
        calls = []

        def procedure(calc):
            calls.append(calc.predictions.hidden.shape[0])
            calc.run((0,))
            calc.add(1, calc.predictions.hidden.mean(0).square().sum())

        p = Predictions(
            hidden=hidden,
            projection=Projection(lambda x: {"logits": x}),
            labels=labels,
            mask=torch.tensor([[1.0], [0.0], [1.0]]),
            positions=torch.zeros_like(labels),
            compute=procedure,
            plan=LossPlan(
                (
                    LossTerm(),
                    LossTerm("router", 0.3, TokenLoss("value"), reduction="scalar"),
                )
            ),
        )
        actual = run(p, chunk=1)
        expected = (
            TokenLoss("ce")(hidden, labels) * p.mask
        ).sum() / p.mask.sum() + 0.3 * hidden.mean(0).square().sum()
        self.assertEqual(calls, [3])
        torch.testing.assert_close(actual.loss, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.loss, hidden)[0],
            torch.autograd.grad(expected, hidden)[0],
        )
        with self.assertRaisesRegex(ValueError, "normalization"):
            LossTerm(reduction="scalar", denominator=torch.tensor(3.0))

    def test_context_and_statistics_scope_are_independent(self):
        from specforge.algorithms.objectives.plan import LossPlan, LossTerm
        from specforge.algorithms.objectives.predictions import Predictions, Teacher
        from specforge.algorithms.objectives.projection import Projection
        from tests.test_algorithms.test_loss_plan import run

        for target in ("labels", "teacher"):
            for mode in ("alpha", "tv", "lambda"):
                with self.subTest(target=target, mode=mode):
                    hidden = torch.randn(3, 1, 4, requires_grad=True)
                    teacher = torch.randn(3, 1, 7)
                    teacher[..., 4:] -= 10
                    calls = []

                    def project(x):
                        calls.append(x.shape[0])
                        return {"logits": x}

                    p = Predictions(
                        hidden=hidden,
                        projection=Projection(project),
                        labels=torch.zeros(3, 1, dtype=torch.long),
                        mask=torch.tensor([[1.0], [0.0], [1.0]]),
                        positions=torch.zeros(3, 1, dtype=torch.long),
                        teacher=Teacher(teacher),
                        denominator=torch.tensor(6.0),
                        vocab_mask=(
                            torch.tensor([True] * 4 + [False] * 3)
                            if target == "teacher"
                            else None
                        ),
                        plan=LossPlan(
                            (
                                LossTerm(
                                    loss=TokenLoss("lk_" + mode, {"target": target})
                                ),
                            )
                        ),
                    )
                    expected = run(p, chunk=0)
                    expected_grad = torch.autograd.grad(expected.loss, hidden)[0]
                    calls.clear()
                    actual = run(p, chunk=1)
                    self.assertEqual(max(calls), 3 if mode == "lambda" else 1)
                    torch.testing.assert_close(actual.loss, expected.loss)
                    torch.testing.assert_close(
                        torch.autograd.grad(actual.loss, hidden)[0],
                        expected_grad,
                    )

    def test_data_inputs_are_not_loss_options(self):
        for kind, name in (
            ("kl", "targets"),
            ("reverse_kl", "targets"),
            ("js", "targets"),
            ("tv", "targets"),
            ("nla", "targets"),
            ("lk", "targets"),
            ("value", "values"),
        ):
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(ValueError, f"unknown {kind} options"),
            ):
                TokenLoss(kind, {name: 1.0})

    def test_saved_implementation_names_are_checked_without_aliases(self):
        from specforge.algorithms.objectives.implementations.registry import (
            ExecutionSelection,
        )

        selection = ExecutionSelection()
        selection.restore({"0:0": "token_ce"})
        selection.record("0:0", "token_ce")
        for name in ("fused_ce_statistics", "linear_ce"):
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(ValueError, "changed on resume"),
            ):
                other = ExecutionSelection()
                other.restore({"0:0": name})
                other.record("0:0", "token_ce")

    @patch.dict(_IMPLEMENTATIONS)
    def test_empty_joint_split_keeps_selection_sequence(self):
        from specforge.algorithms.objectives.implementations.registry import (
            ExecutionSelection,
        )

        @register_implementation(
            "test_token",
            losses=("ce",),
            supports=lambda request, group: (
                None if len(group) == 1 else "single term only"
            ),
        )
        def token(request, group):
            self.assertGreater(request.input_tensor.numel(), 0)
            return select_implementation(request, group).compute(request, group)

        for length in (0, 2):
            request = self.request()
            logits = request.tensor(0)[:length]
            term = replace(
                request.terms[0],
                mask=request.terms[0].mask[:length],
                weights=request.terms[0].weights[:length],
            )
            request = replace(
                request,
                terms=(term, term),
                labels=request.labels[:length],
                tensors={0: logits, 1: logits},
            )
            selection = ExecutionSelection()
            compute_loss(request, (0, 1), fused=True, selection=selection)
            self.assertEqual(
                selection.names,
                {"0:0,1": "split", "0:0": "test_token", "0:1": "test_token"},
            )
