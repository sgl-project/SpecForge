"""Public loss modes preserve the mathematical contract and output requests."""

import sys
import types
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from specforge.algorithms.objectives.implementations.registry import (
    _IMPLEMENTATIONS,
    ExecutionRequest,
    ExecutionSelection,
    TermInputs,
    compute_loss,
    implementation_registrations,
    register_implementation,
    select_implementation,
)
from specforge.algorithms.objectives.losses import LossContext, TokenLoss


class LossRoutingTest(unittest.TestCase):

    def setUp(self):
        # Built-in implementations register on first use. Do it before any test
        # patches the registry, or the patch exit drops them for later tests.
        implementation_registrations()

    def test_loss_input_kinds_and_reserved_options_fail_at_construction(self):
        from specforge.algorithms.objectives.losses import _LOSSES, register_loss
        from specforge.algorithms.objectives.registry import resolve_objective

        for kind in ("value", "bce", "cosine"):
            with (
                self.subTest(kind=kind),
                self.assertRaisesRegex(ValueError, "incompatible.*inputs"),
            ):
                resolve_objective(
                    "MTPDraftModel", loss={"terms": {"token": {"kind": kind}}}
                )
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
        multi = replace(
            prediction,
            plan=LossPlan((LossTerm(), LossTerm()), ((0, 1),)),
            compute=lambda calc: calc.run((0, 1), top_k=2),
        )
        with self.assertRaisesRegex(ValueError, "single declared term"):
            run(multi)

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

    def test_declared_context_statistics_are_delivered(self):
        import math

        from specforge.algorithms.objectives.execution import execute_batch
        from specforge.algorithms.objectives.execution_options import LossExecution
        from specforge.algorithms.objectives.losses import (
            _LOSSES,
            PositionWeighting,
            register_loss,
        )
        from specforge.algorithms.objectives.plan import LossPlan, LossTerm
        from specforge.algorithms.objectives.predictions import (
            PredictionBatch,
            Predictions,
            Teacher,
        )
        from specforge.algorithms.objectives.projection import Projection

        def run(prediction, chunk=0):
            return execute_batch(
                PredictionBatch((prediction,), reduction="sum_means"),
                TokenLoss("ce"),
                PositionWeighting(),
                training=True,
                execution=LossExecution(chunk_size=chunk, checkpoint=False),
            ).loss

        def cross_entropy(logits, labels):
            return (
                -logits.float()
                .log_softmax(-1)
                .gather(-1, labels[..., None])
                .squeeze(-1)
            )

        # Draft probabilities (0.6, 0.4); a uniform three-token teacher of which
        # the draft vocabulary keeps two, so the retained teacher mass is 2/3.
        def masked(plan, compute=None):
            labels = torch.zeros(2, 1, dtype=torch.long)
            return Predictions(
                hidden=torch.tensor([[[0.6, 0.4]], [[0.6, 0.4]]]).log(),
                projection=Projection(lambda x: {"logits": x}),
                labels=labels,
                mask=torch.ones(2, 1),
                positions=labels,
                teacher=Teacher(torch.zeros(2, 1, 3)),
                requires=frozenset({"teacher_logits"}),
                vocab_mask=torch.tensor([True, True, False]),
                plan=plan,
                compute=compute,
            )

        with patch.dict(_LOSSES):
            seen = []

            def read_mass(logits, labels, teacher, *, context):
                seen.append(context.teacher_mass)
                return cross_entropy(logits, labels) * context.teacher_mass

            register_loss("mass_default", contextual=True, scope="positions")(read_mass)
            register_loss(
                "mass_declared",
                contextual=True,
                scope="positions",
                context_statistics=("teacher_mass",),
            )(read_mass)
            # The protocol, not the loss, supplies the teacher.
            for kind in ("mass_default", "mass_declared"):
                with self.subTest(kind=kind):
                    seen.clear()
                    loss = run(masked(LossPlan((LossTerm(loss=TokenLoss(kind)),))))
                    self.assertEqual(len(seen), 1)
                    torch.testing.assert_close(seen[0], torch.full((2, 1), 2 / 3))
                    self.assertAlmostEqual(
                        loss.item(), -math.log(0.6) * 2 / 3, places=6
                    )

            # A callback evaluates a formula that no declared term selects.
            # min(p, q) sums to 1/3 + 1/3 with the mass and to 0.5 + 0.4 without.
            teacher_tv = TokenLoss("lk_tv", {"target": "teacher"})

            def compute(calc):
                calc.run((0,))
                out = calc.evaluate(
                    "lk",
                    teacher_tv,
                    calc.outputs()["logits"],
                    calc.labels,
                    teacher=calc.teacher,
                )
                calc.add(1, out.values)

            declared = run(masked(LossPlan((LossTerm(loss=teacher_tv),))))
            self.assertAlmostEqual(declared.item(), 1 / 3, places=6)
            for main in ("ce", "ce_l1"):
                with self.subTest(intermediate_after=main):
                    plan = LossPlan(
                        (
                            LossTerm(loss=TokenLoss(main), coefficient=0.0),
                            LossTerm("computed", loss=TokenLoss("value")),
                        )
                    )
                    torch.testing.assert_close(run(masked(plan, compute)), declared)

            # A declared whole-set count does not depend on a fixed denominator,
            # and it counts only positions the vocabulary mapping keeps.
            counts = []

            def read_count(logits, labels, teacher, *, context):
                counts.append(context.supervised_count)
                count = context.supervised_count
                return cross_entropy(logits, labels) / (
                    context.mask.sum() if count is None else count
                )

            register_loss(
                "count_declared",
                contextual=True,
                scope="positions",
                context_statistics=("supervised_count",),
            )(read_count)
            plan = LossPlan((LossTerm(loss=TokenLoss("count_declared")),))
            for fixed in (None, 2.0):
                with self.subTest(fixed_denominator=fixed):
                    labels = torch.zeros(2, 1, dtype=torch.long)
                    prediction = Predictions(
                        hidden=torch.zeros(2, 1, 2),
                        projection=Projection(lambda x: {"logits": x}),
                        labels=labels,
                        mask=torch.ones(2, 1),
                        positions=labels,
                        denominator=None if fixed is None else torch.tensor(fixed),
                        plan=plan,
                    )
                    whole = run(prediction)
                    counts.clear()
                    torch.testing.assert_close(run(prediction, chunk=1), whole)
                    self.assertEqual([float(count) for count in counts], [2.0, 2.0])
                    self.assertAlmostEqual(whole.item(), math.log(2) / 2, places=6)

            # The third unit's teacher prefers a token outside the draft
            # vocabulary, so only two supervised positions remain.
            teacher = torch.zeros(3, 1, 3)
            teacher[2, 0, 2] = 5.0
            labels = torch.zeros(3, 1, dtype=torch.long)
            prediction = Predictions(
                hidden=torch.zeros(3, 1, 2),
                projection=Projection(lambda x: {"logits": x}),
                labels=labels,
                mask=torch.ones(3, 1),
                positions=labels,
                teacher=Teacher(teacher),
                requires=frozenset({"teacher_logits"}),
                vocab_mask=torch.tensor([True, True, False]),
                plan=plan,
            )
            counts.clear()
            run(prediction, chunk=1)
            self.assertEqual([float(count) for count in counts], [2.0, 2.0, 2.0])

    def test_conditional_statistics_declarations_follow_the_options(self):
        from specforge.algorithms.objectives.losses import _LOSSES, register_loss

        for mode, target, fixed, expected in (
            ("alpha", "labels", False, set()),
            ("alpha", "labels", True, {"supervised_count"}),
            ("tv", "teacher", False, {"teacher_mass"}),
            ("tv", "teacher", True, {"teacher_mass", "supervised_count"}),
            ("lambda", "labels", False, {"supervised_count"}),
            ("lambda", "teacher", False, {"teacher_mass", "supervised_count"}),
        ):
            with self.subTest(mode=mode, target=target, fixed=fixed):
                loss = TokenLoss("lk_" + mode, {"target": target})
                self.assertEqual(
                    loss.context_statistics(fixed_denominator=fixed), expected
                )
        self.assertEqual(
            TokenLoss("ce_l1").context_statistics(fixed_denominator=True), frozenset()
        )
        with patch.dict(_LOSSES):
            register_loss(
                "bad_statistics",
                contextual=True,
                context_statistics=lambda options, *, fixed_denominator: {"mean"},
            )(lambda logits, labels, teacher, *, context: logits[..., 0])
            with self.assertRaisesRegex(ValueError, "context statistics"):
                TokenLoss("bad_statistics").context_statistics(fixed_denominator=True)
            with self.assertRaisesRegex(ValueError, "contextual loss"):
                register_loss(
                    "uncontextual",
                    context_statistics=lambda options, *, fixed_denominator: (),
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

    def test_default_selection_never_raises_and_skips_opt_in_kernels(self):
        from specforge.algorithms.objectives.execution_options import LossExecution

        self.assertIsNone(LossExecution().fused)
        with self.assertRaisesRegex(ValueError, "fused must be boolean or None"):
            LossExecution(fused="auto")
        self.assertFalse(implementation_registrations()["linear_ce"].auto)
        self.assertTrue(implementation_registrations()["token_ce"].auto)
        request = self.request()
        # CPU inputs: the default uses the reference, an explicit request raises.
        self.assertEqual(select_implementation(request, fused=None).name, "reference")
        with self.assertRaisesRegex(ValueError, "requires CUDA/ROCm"):
            select_implementation(request, fused=True)
        with patch.dict(_IMPLEMENTATIONS):

            def missing_dependency(request, group):
                raise ImportError("kernel package is not installed")

            register_implementation(
                "needs_package",
                supports=missing_dependency,
                priority=30,
                losses=("ce",),
            )(lambda request, group: None)
            register_implementation(
                "opt_in",
                supports=lambda request, group: None,
                priority=20,
                losses=("ce",),
                auto=False,
            )(lambda request, group: None)
            register_implementation(
                "validated",
                supports=lambda request, group: None,
                priority=10,
                losses=("ce",),
            )(lambda request, group: None)
            skipped = []
            entry = select_implementation(request, fused=None, rejections=skipped)
            self.assertEqual(entry.name, "validated")
            self.assertIn("needs_package: kernel package is not installed", skipped)
            with self.assertRaisesRegex(ImportError, "not installed"):
                select_implementation(request, fused=True)
            del _IMPLEMENTATIONS["needs_package"]
            self.assertEqual(select_implementation(request, fused=True).name, "opt_in")
            self.assertEqual(
                select_implementation(request, fused=False).name, "reference"
            )

    def test_selection_issues_no_collective(self):
        """Ranks agree on their selections at a checkpoint, through the step runner."""
        selection = ExecutionSelection()
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.all_gather_object") as gather,
        ):
            selection.resolve(self.request(), (0,), 0, fused=None)
            gather.assert_not_called()
        self.assertEqual(selection.names, {"0:0": "reference"})

    def test_saved_implementation_names_are_checked_without_aliases(self):
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

    def _device_request(
        self, loss, *, device_type="cuda", dtype=torch.bfloat16, shape=(2, 1, 5)
    ):
        tensor = SimpleNamespace(
            device=SimpleNamespace(type=device_type),
            is_cuda=device_type == "cuda",
            dtype=dtype,
            shape=shape,
            numel=lambda: 10,
        )
        labels = torch.zeros(shape[:-1], dtype=torch.long)
        weights = torch.ones_like(labels, dtype=torch.float32)
        term = TermInputs(loss, weights, weights, weights.sum(), LossContext(weights))
        return ExecutionRequest(
            None,
            (term,),
            labels,
            tensor if loss.requires_teacher else None,
            None,
            {},
            unreduced=True,
            tensors={0: tensor},
        )

    def test_fused_true_rejects_unsupported_categorical_loss_on_cuda_and_npu(self):
        from specforge.algorithms.objectives.losses import _LOSSES, register_loss

        for device_type in ("cuda", "npu"):
            with self.subTest(device_type=device_type):
                teacher_lk = self._device_request(
                    TokenLoss("lk_alpha", {"target": "teacher"}),
                    device_type=device_type,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "No fused implementation supports categorical loss.*teacher LK uses the reference formula",
                ):
                    select_implementation(teacher_lk, fused=True)
                self.assertEqual(
                    select_implementation(teacher_lk, fused=None).name, "reference"
                )

                with patch.dict(_LOSSES):
                    register_loss("custom_cat")(
                        lambda logits, labels, teacher=None: logits[..., 0]
                    )
                    custom = self._device_request(
                        TokenLoss("custom_cat"), device_type=device_type
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        r"No fused implementation supports categorical loss \(custom_cat\)$",
                    ):
                        select_implementation(custom, fused=True)

                for kind, expected_impl in (
                    ("value", "value"),
                    ("bce", "reference"),
                    ("cosine", "reference"),
                ):
                    non_cat = self._device_request(
                        TokenLoss(kind), device_type=device_type
                    )
                    self.assertEqual(
                        select_implementation(non_cat, fused=True).name, expected_impl
                    )

    @staticmethod
    def _fake_triton_modules(*, backend="cuda", with_constexpr=True):
        fake_triton = types.ModuleType("triton")
        fake_triton.jit = lambda fn: fn
        fake_tl = types.ModuleType("triton.language")
        if with_constexpr:
            fake_tl.constexpr = lambda value: value
        fake_runtime = types.ModuleType("triton.runtime")
        if backend is not None:
            fake_runtime.driver = SimpleNamespace(
                active=SimpleNamespace(
                    get_current_target=lambda: SimpleNamespace(backend=backend)
                )
            )
        fake_triton.runtime = fake_runtime
        return {
            "triton": fake_triton,
            "triton.language": fake_tl,
            "triton.runtime": fake_runtime,
        }

    def test_auto_fused_checks_triton_for_all_kernels_and_allows_npu(self):
        from specforge.losses import fused as fused_loss

        npu_kinds = ("soft_ce", "kl", "reverse_kl", "js", "tv", "nla", "lk")
        cuda_only_kinds = ("ce", "ce_l1", "lk_alpha", "lk_tv", "lk_lambda")
        all_token_kinds = npu_kinds + cuda_only_kinds

        with patch.dict(
            sys.modules,
            {"triton": None, "triton.language": None, "triton.runtime": None},
        ):
            for kind in all_token_kinds:
                with self.subTest(missing_triton=kind):
                    req = self._device_request(TokenLoss(kind), device_type="cuda")
                    reasons = []
                    self.assertEqual(
                        select_implementation(req, fused=None, rejections=reasons).name,
                        "reference",
                    )
                    self.assertTrue(
                        any("requires Triton" in r for r in reasons), reasons
                    )
                    with self.assertRaisesRegex(ImportError, "requires Triton"):
                        select_implementation(req, fused=True)
            with self.assertRaisesRegex(ImportError, "requires Triton"):
                ExecutionSelection().resolve(
                    self._device_request(TokenLoss("ce"), device_type="cuda"),
                    (0,),
                    0,
                    fused=True,
                )

        with patch.dict(sys.modules, self._fake_triton_modules(backend="npu")):
            for kind in npu_kinds:
                with self.subTest(npu_supported=kind):
                    req = self._device_request(TokenLoss(kind), device_type="npu")
                    self.assertEqual(
                        select_implementation(req, fused=None).name, f"token_{kind}"
                    )
                    self.assertEqual(
                        select_implementation(req, fused=True).name, f"token_{kind}"
                    )

            kl_req = self._device_request(TokenLoss("kl"), device_type="npu")
            npu_logits = kl_req.tensor(0)
            expected_kl = torch.full((2, 1), 0.25)
            with (
                patch.object(fused_loss, "_require_triton"),
                patch.object(fused_loss, "_OP_KL", SimpleNamespace(value=0)),
                patch.object(
                    fused_loss._FusedLoss, "apply", return_value=expected_kl
                ) as mock_apply,
            ):
                entry = select_implementation(kl_req, fused=True)
                self.assertEqual(entry.name, "token_kl")
                out = entry.compute(kl_req, (0,))
                mock_apply.assert_called_once_with(npu_logits, npu_logits, 0)
                torch.testing.assert_close(out.values[0], expected_kl)

            for kind in cuda_only_kinds:
                with self.subTest(npu_cuda_only=kind):
                    req = self._device_request(TokenLoss(kind), device_type="npu")
                    reasons = []
                    self.assertEqual(
                        select_implementation(req, fused=None, rejections=reasons).name,
                        "reference",
                    )
                    self.assertTrue(
                        any("requires CUDA/ROCm inputs" in r for r in reasons), reasons
                    )
                    with self.assertRaisesRegex(
                        ValueError, "No fused implementation supports categorical loss"
                    ):
                        select_implementation(req, fused=True)

    @patch.dict(_IMPLEMENTATIONS)
    def test_custom_token_kernel_does_not_inherit_triton_dependency(self):
        from specforge.algorithms.objectives.implementations.token_kernels import (
            register_token_kernel,
        )

        register_token_kernel(
            "bce", lambda logits, labels, teacher=None: torch.zeros_like(labels).float()
        )
        req = self._device_request(TokenLoss("bce"), device_type="cuda")
        with patch.dict(
            sys.modules,
            {"triton": None, "triton.language": None, "triton.runtime": None},
        ):
            self.assertEqual(select_implementation(req, fused=None).name, "token_bce")
            self.assertEqual(select_implementation(req, fused=True).name, "token_bce")

    def test_npu_automatic_mode_checkpoint_resume_honors_reference(self):
        npu_req = self._device_request(TokenLoss("soft_ce"), device_type="npu")
        with patch.dict(sys.modules, self._fake_triton_modules(backend="npu")):
            resumed = ExecutionSelection()
            resumed.restore({"0:0": "reference"})
            entry = resumed.resolve(npu_req, (0,), 0, fused=None)
            self.assertEqual(entry.name, "reference")
            self.assertEqual(resumed.names, {"0:0": "reference"})

            strict = ExecutionSelection()
            strict.restore({"0:0": "reference"})
            with self.assertRaisesRegex(ValueError, "changed on resume"):
                strict.resolve(npu_req, (0,), 0, fused=True)

        with patch.dict(
            sys.modules,
            {"triton": None, "triton.language": None, "triton.runtime": None},
        ):
            downgraded = ExecutionSelection()
            downgraded.restore({"0:0": "token_soft_ce"})
            with self.assertRaisesRegex(ValueError, "changed on resume"):
                downgraded.resolve(npu_req, (0,), 0, fused=None)

    def test_domino_execute_batch_missing_triton_auto_and_strict(self):
        from specforge.algorithms.objectives.execution import execute_batch
        from specforge.algorithms.objectives.execution_options import LossExecution
        from specforge.algorithms.objectives.losses import PositionWeighting
        from specforge.algorithms.objectives.plan import LossPlan, LossTerm
        from specforge.algorithms.objectives.predictions import (
            PredictionBatch,
            Predictions,
        )
        from specforge.algorithms.objectives.projection import DominoSuffixProjection

        class _CudaHidden:
            def __init__(self, tensor):
                self._tensor = tensor
                self.device = SimpleNamespace(type="cuda")
                self.is_cuda = True
                self.dtype = torch.bfloat16
                self.shape = tensor.shape
                self.ndim = tensor.ndim

            def numel(self):
                return self._tensor.numel()

            def __getitem__(self, item):
                return _CudaHidden(self._tensor[item])

        raw = torch.randn(2, 4, 6)
        labels = torch.zeros(2, 4, dtype=torch.long)
        mask = torch.ones(2, 4)
        projection = DominoSuffixProjection(
            lambda h: {"logits": h._tensor, "base_logits": h._tensor},
            lambda h: (h._tensor, h._tensor),
            block_size=4,
            suffix_start=1,
        )
        pred = Predictions(
            hidden=_CudaHidden(raw),
            projection=projection,
            labels=labels,
            mask=mask,
            positions=torch.arange(4).expand(2, 4),
            plan=LossPlan((LossTerm("logits"), LossTerm("base_logits")), ((0, 1),)),
        )
        batch = PredictionBatch((pred,))
        with patch.dict(
            sys.modules,
            {"triton": None, "triton.language": None, "triton.runtime": None},
        ):
            auto_selection = ExecutionSelection()
            result = execute_batch(
                batch,
                TokenLoss("ce"),
                PositionWeighting(),
                training=False,
                execution=LossExecution(chunk_size=0, checkpoint=False, fused=None),
                selection=auto_selection,
            )
            self.assertTrue(torch.isfinite(result.loss))
            self.assertEqual(
                auto_selection.names,
                {"0:0,1": "split", "0:0": "reference", "0:1": "reference"},
            )
            with self.assertRaisesRegex(ImportError, "requires Triton"):
                execute_batch(
                    batch,
                    TokenLoss("ce"),
                    PositionWeighting(),
                    training=False,
                    execution=LossExecution(chunk_size=0, checkpoint=False, fused=True),
                    selection=ExecutionSelection(),
                )
