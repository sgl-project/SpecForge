"""Joint ranges are optional; term reductions and gradients are not."""

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from specforge.algorithms.objectives.execution import execute_batch
from specforge.algorithms.objectives.execution_options import LossExecution
from specforge.algorithms.objectives.implementations import domino_ce  # noqa: F401
from specforge.algorithms.objectives.implementations.linear_ce import _linear_kernel
from specforge.algorithms.objectives.implementations.registry import (
    _IMPLEMENTATIONS,
    ExecutionSelection,
    GroupResult,
    register_implementation,
)
from specforge.algorithms.objectives.losses import PositionWeighting, TokenLoss
from specforge.algorithms.objectives.plan import LossPlan, LossTerm
from specforge.algorithms.objectives.predictions import PredictionBatch, Predictions
from specforge.algorithms.objectives.projection import LinearProjection, Projection
from tests.test_algorithms.test_objective_contracts import build_runtime
from tests.test_algorithms.test_objectives import training_batch


def run(prediction, *, fused=False, chunk=0, reduction="sum_means"):
    return execute_batch(
        PredictionBatch((prediction,), reduction=reduction),
        TokenLoss(kind="ce"),
        PositionWeighting(),
        training=True,
        execution=LossExecution(fused=fused, chunk_size=chunk, trim=True),
    )


class DistributedPlan(nn.Module):
    def __init__(self, scalar=False):
        super().__init__()
        self.scalar = scalar
        self.head = nn.Linear(4, 9).double()

    def forward(self, hidden, labels, mask, other):
        def compute(calc):
            calc.run((0,))
            calc.run((1,))
            calc.add(2, calc.outputs()["logits"].square().mean(-1))
            if self.scalar:
                calc.add(3, calc.outputs()["logits"].square().mean())

        prediction = Predictions(
            hidden=hidden,
            projection=Projection(lambda x: {"logits": self.head(x)}),
            labels=labels,
            positions=torch.zeros_like(labels),
            mask=mask,
            compute=compute,
            plan=LossPlan(
                (
                    LossTerm(),
                    LossTerm(coefficient=0.3, mask=other),
                    LossTerm("square", 0.2, TokenLoss("value"), mask=other),
                )
                + (
                    (LossTerm("router", 0.1, TokenLoss("value"), reduction="scalar"),)
                    if self.scalar
                    else ()
                ),
            ),
        )
        return run(prediction, chunk=1, reduction="global_microbatch").loss


def distributed_reduction(rank, rendezvous):
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        for scalar in (False, True):
            torch.manual_seed(19)
            model = DistributedPlan(scalar=scalar)
            reference = nn.Linear(4, 9).double()
            reference.load_state_dict(model.head.state_dict())
            wrapped = DistributedDataParallel(model)
            expected = 0
            for step in range(2):
                hidden = torch.arange(24, dtype=torch.float64).reshape(6, 1, 4) / (
                    10 + step
                )
                labels = (torch.arange(6) + step).reshape(6, 1)
                mask = torch.tensor([0, 0, 0, 1, 0.5, 1.0])[:, None]
                other = torch.tensor([1, 0, 1, 0, 0.2, 0])[:, None]
                part = slice(rank * 3, (rank + 1) * 3)
                (
                    wrapped(hidden[part], labels[part], mask[part], other[part]) / 2
                ).backward()
                values = TokenLoss(kind="ce")(reference(hidden), labels)
                expected = (
                    expected
                    + (
                        (values * mask).sum() / mask.sum()
                        + 0.3 * (values * other).sum() / other.sum()
                        + 0.2
                        * (reference(hidden).square().mean(-1) * other).sum()
                        / other.sum()
                        + (0.1 * reference(hidden).square().mean() if scalar else 0)
                    )
                    / 2
                )
            expected.backward()
            torch.testing.assert_close(
                tuple(p.grad for p in model.head.parameters()),
                tuple(p.grad for p in reference.parameters()),
            )
        # Resolve before compute: every rank sees mismatches and dependency errors.

        from specforge.algorithms.objectives.implementations.registry import (
            ExecutionSelection,
            Implementation,
        )

        request = SimpleNamespace(input_tensor=hidden)
        for mismatch in (False, True):
            entry = Implementation(f"kernel_{rank if mismatch else 0}", 0, None, None)
            with patch(
                "specforge.algorithms.objectives.implementations.registry.select_implementation",
                return_value=entry,
            ):
                selection = ExecutionSelection()
                if mismatch:
                    with unittest.TestCase().assertRaisesRegex(
                        ValueError, "differ across ranks"
                    ):
                        selection.resolve(request, (0,), 0)
                else:
                    selection.resolve(request, (0,), 0)
                    assert selection.names == {"0:0": "kernel_0"}
        with (
            patch(
                "specforge.algorithms.objectives.implementations.registry.select_implementation",
                side_effect=ImportError("missing kernel dependency") if rank else None,
                return_value=entry,
            ),
            unittest.TestCase().assertRaisesRegex(
                ValueError, "missing kernel dependency"
            ),
        ):
            ExecutionSelection().resolve(request, (0,), 0)
    finally:
        dist.destroy_process_group()


class LossPlanTest(unittest.TestCase):
    @unittest.skipUnless(dist.is_gloo_available(), "Gloo required")
    def test_distributed_term_reduction_and_accumulation(self):
        with TemporaryDirectory() as tmp:
            mp.spawn(
                distributed_reduction,
                args=((Path(tmp) / "rendezvous").as_uri(),),
                nprocs=2,
                join=True,
            )

    def test_joint_range_and_independent_denominator(self):
        torch.manual_seed(31)
        head = nn.Linear(4, 9).double()
        hidden = torch.randn(5, 1, 4, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([[1.0], [0.5], [0.0], [1.0], [0.0]])
        other = torch.tensor([[0.0], [1.0], [1.0], [0.0], [0.0]])
        labels = torch.randint(9, (5, 1))

        def project(x):
            logits = head(x)
            return {"logits": logits, "base_logits": logits * 0.5}

        terms = (
            LossTerm("logits", 0.6),
            LossTerm("base_logits", 0.4),
            LossTerm("base_logits", 0.2, mask=other, denominator=torch.tensor(7.0)),
        )
        p = Predictions(
            hidden=hidden,
            projection=Projection(project),
            mask=mask,
            positions=torch.zeros_like(labels),
            labels=labels,
            plan=LossPlan(terms, ((0, 1),)),
        )
        output = project(hidden)
        ce = TokenLoss(kind="ce")
        expected = (
            0.6 * (ce(output["logits"], labels) * mask).sum() / mask.sum()
            + 0.4 * (ce(output["base_logits"], labels) * mask).sum() / mask.sum()
            + 0.2 * (ce(output["base_logits"], labels) * other).sum() / 7
        )
        parameters = (hidden, *head.parameters())
        expected_grad = torch.autograd.grad(expected, parameters, retain_graph=True)
        with patch.dict(
            _IMPLEMENTATIONS,
            {k: v for k, v in _IMPLEMENTATIONS.items() if v.category == "reference"},
            clear=True,
        ):
            calls = []

            @register_implementation(
                "test_joint",
                supports=lambda req, group: None if group == (0, 1) else "pair only",
            )
            def joint(req, group):
                if len(group) > 1:
                    calls.append(group)
                values = req.outputs()
                return GroupResult(
                    {
                        i: (
                            req.terms[i].loss(
                                values[req.predictions.plan.terms[i].output], req.labels
                            )
                            * req.terms[i].weights
                        ).sum()
                        for i in group
                    }
                )

            register_implementation(
                "test_single",
                supports=lambda req, group: None if len(group) == 1 else "single only",
            )(joint)

            for chunk in (0, 1, 3):
                for groups in (((0, 1),), ()):
                    with self.subTest(chunk=chunk, groups=groups):
                        calls.clear()
                        actual = run(
                            replace(p, plan=replace(p.plan, joint_groups=groups)),
                            fused=True,
                            chunk=chunk,
                        )
                        gradients = torch.autograd.grad(actual.loss, parameters)
                        torch.testing.assert_close(actual.loss, expected)
                        torch.testing.assert_close(gradients, expected_grad)
                        self.assertEqual(bool(calls), bool(groups))
            # Empty ranks retain the same joint-selection sequence as populated ranks.
            selection = ExecutionSelection()
            execute_batch(
                PredictionBatch((p.select(slice(0, 0)),), reduction="sum_means"),
                TokenLoss("ce"),
                PositionWeighting(),
                training=True,
                execution=LossExecution(fused=True),
                selection=selection,
            )
            self.assertEqual(
                selection.names, {"0:0,1": "test_joint", "0:2": "test_single"}
            )
            with self.assertRaisesRegex(ValueError, "Independent term reductions"):
                run(p, reduction="tokens")

    def test_computed_terms_have_independent_reductions(self):
        torch.manual_seed(17)
        hidden = torch.randn(5, 1, 4, dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([1, 0, 0, 0.5, 0], dtype=torch.float64)[:, None]
        other = torch.tensor([0, 1, 1, 0, 0], dtype=torch.float64)[:, None]
        labels = torch.zeros(5, 1, dtype=torch.long)
        weighting = PositionWeighting(kind="exponential", gamma=2)
        positions = torch.arange(5)[:, None]

        def compute(calc):
            calc.run((0,))
            output = calc.outputs()
            calc.add(1, output["logits"].square().mean(-1))
            calc.add(2, output["logits"].abs().mean(-1))

        p = Predictions(
            hidden=hidden,
            projection=Projection(lambda x: {"logits": x}),
            mask=mask,
            labels=labels,
            positions=positions,
            compute=compute,
            plan=LossPlan(
                terms=(
                    LossTerm(coefficient=0.7, denominator=torch.tensor(3.0)),
                    LossTerm(
                        "square",
                        0.2,
                        TokenLoss("value"),
                        mask=other,
                        weighting=weighting,
                    ),
                    LossTerm(
                        "absolute",
                        0.3,
                        TokenLoss("value"),
                        mask=other,
                        denominator=torch.tensor(7.0),
                    ),
                ),
            ),
        )
        weighted = weighting(other, positions)
        expected = (
            0.7 * (TokenLoss(kind="ce")(hidden, labels) * mask).sum() / 3
            + 0.2 * (hidden.square().mean(-1) * weighted).sum() / weighted.sum()
            + 0.3 * (hidden.abs().mean(-1) * other).sum() / 7
        )
        gradient = torch.autograd.grad(expected, hidden)[0]
        for chunk in (0, 1, 3):
            actual = run(p, chunk=chunk)
            torch.testing.assert_close(actual.loss, expected)
            torch.testing.assert_close(
                torch.autograd.grad(actual.loss, hidden)[0], gradient
            )
        empty = replace(
            p,
            mask=torch.zeros_like(mask),
            plan=replace(
                p.plan,
                terms=tuple(
                    replace(term, mask=torch.zeros_like(other)) for term in p.plan.terms
                ),
            ),
        )
        self.assertEqual(run(empty, chunk=1).loss.item(), 0)
        with self.assertRaisesRegex(ValueError, "every declared loss term"):
            run(replace(p, compute=lambda calc: None))

    def test_dspark_reuses_l1_statistics(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            config, runtime = build_runtime(
                "DSparkDraftModel", options={"num_anchors": 3}
            )
            runtime.to(device)
            for name in config.target_bindings:
                module = runtime.draft_model.get_submodule(name)
                module.load_target_weight(torch.randn_like(module.weight))
            batch = {
                key: value.to(device) for key, value in training_batch(config).items()
            }
            p = next(iter(runtime.objective(runtime, batch).predictions))
            for fused in (False, True) if device == "cuda" else (False,):
                results = []
                for reuse in (True, False):

                    def compute(calc):
                        original_run = calc.run

                        def run_group(group=None):
                            original_run(group)
                            if group == (0,) and not reuse:
                                calc.statistics.clear()

                        with (
                            patch.object(calc, "run", side_effect=run_group),
                            patch.object(
                                calc, "evaluate", wraps=calc.evaluate
                            ) as evaluate,
                        ):
                            p.compute(calc)
                        keys = [call.args[0] for call in evaluate.call_args_list]
                        self.assertEqual("distance" in keys, not reuse)

                    result = execute_batch(
                        PredictionBatch((replace(p, compute=compute),)),
                        TokenLoss("ce_l1"),
                        PositionWeighting(),
                        training=True,
                        execution=LossExecution(fused=fused, checkpoint=False),
                    )
                    gradient = torch.autograd.grad(
                        result.loss, p.hidden, retain_graph=True
                    )[0]
                    results.append((result.loss, gradient))
                torch.testing.assert_close(results[0], results[1])

    def test_router_loss_uses_shared_reduction_and_keeps_gradients(self):
        torch.manual_seed(13)
        hidden = torch.randn(4, 1, 5, requires_grad=True)
        router = nn.Linear(5, 3)
        labels = torch.zeros(4, 1, dtype=torch.long)
        mask = torch.tensor([1.0, 0.0, 0.5, 1.0])[:, None]
        router_mask = torch.tensor([0.0, 1.0, 1.0, 1.0])[:, None]
        # A model-owned, differentiable router z-loss; no new loss backend needed.
        z_loss = router(hidden).logsumexp(-1).square()

        def compute(calc):
            calc.run((0,))
            calc.add(1, calc.predictions.context_inputs["router_z"])

        p = Predictions(
            hidden=hidden,
            projection=Projection(lambda x: {"logits": x}),
            labels=labels,
            mask=mask,
            positions=torch.zeros_like(labels),
            context_inputs={"router_z": z_loss},
            compute=compute,
            plan=LossPlan(
                (
                    LossTerm(),
                    LossTerm("router_z", 0.1, TokenLoss("value"), mask=router_mask),
                )
            ),
        )
        expected = (TokenLoss("ce")(hidden, labels) * mask).sum() / mask.sum() + 0.1 * (
            z_loss * router_mask
        ).sum() / router_mask.sum()
        parameters = (hidden, *router.parameters())
        gradient = torch.autograd.grad(expected, parameters, retain_graph=True)
        for chunk in (0, 1, 3):
            actual = run(p, chunk=chunk)
            torch.testing.assert_close(actual.loss, expected)
            torch.testing.assert_close(
                torch.autograd.grad(actual.loss, parameters, retain_graph=True),
                gradient,
            )

    def test_whole_batch_router_balance_is_not_recomputed_per_chunk(self):
        router_logits = torch.randn(4, 1, 3, requires_grad=True)
        labels = torch.zeros(4, 1, dtype=torch.long)
        mask = torch.tensor([1.0, 0.0, 1.0, 1.0])[:, None]
        calls = []

        def balance(logits, mask):
            probs = logits.softmax(-1)
            frequency = torch.nn.functional.one_hot(probs.argmax(-1), 3).to(probs)
            count = mask.sum()
            return (
                3
                * (
                    (probs * mask[..., None]).sum((0, 1))
                    / count
                    * (frequency * mask[..., None]).sum((0, 1))
                    / count
                ).sum()
            )

        def compute(calc):
            calc.run((0,))
            calls.append(calc.predictions.hidden.shape[0])
            value = balance(calc.predictions.hidden, calc.predictions.mask)
            # Explicitly represent the batch mean as a constant per-token value.
            calc.add(1, value.expand_as(calc.predictions.mask))

        p = Predictions(
            hidden=router_logits,
            projection=Projection(lambda x: {"logits": x}),
            labels=labels,
            positions=torch.zeros_like(labels),
            mask=mask,
            compute=compute,
            chunk_safe=False,
            plan=LossPlan((LossTerm(), LossTerm("balance", 0.1, TokenLoss("value")))),
        )
        actual = run(p, chunk=1)
        expected = (
            TokenLoss("ce")(router_logits, labels) * mask
        ).sum() / mask.sum() + 0.1 * balance(router_logits, mask)
        self.assertEqual(calls, [4])
        torch.testing.assert_close(actual.loss, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual.loss, router_logits)[0],
            torch.autograd.grad(expected, router_logits)[0],
        )

    def test_extra_outputs_do_not_change_the_ce_loss(self):
        hidden = torch.randn(3, 1, 5, requires_grad=True)
        teacher = torch.randn_like(hidden)
        labels = torch.zeros(3, 1, dtype=torch.long)

        def compute(calc):
            result = calc.evaluate(
                "ce_with_statistics",
                TokenLoss(kind="ce"),
                calc.outputs()["logits"],
                calc.labels,
                teacher=teacher,
                return_teacher_l1=True,
                top_k=2,
            )
            torch.testing.assert_close(
                result.values, TokenLoss(kind="ce")(hidden, labels)
            )
            torch.testing.assert_close(
                result.statistics["teacher_l1"],
                (hidden.softmax(-1) - teacher.softmax(-1)).abs().sum(-1),
            )
            torch.testing.assert_close(
                result.statistics["topk_values"], hidden.topk(2).values
            )
            calc.add(0, result.values)

        prediction = Predictions(
            hidden=hidden,
            projection=Projection(lambda x: {"logits": x}),
            labels=labels,
            mask=torch.ones_like(labels),
            positions=torch.zeros_like(labels),
            compute=compute,
            plan=LossPlan((LossTerm("ce_value", loss=TokenLoss("value")),)),
        )
        actual = run(prediction).loss
        expected = TokenLoss(kind="ce")(hidden, labels).mean()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, hidden)[0],
            torch.autograd.grad(expected, hidden)[0],
        )
        with self.assertRaisesRegex(ValueError, "unknown ce options"):
            TokenLoss(kind="ce", options={"teacher_l1": True})

    def test_new_statistics_belong_to_the_mathematical_loss(self):
        from specforge.algorithms.objectives.losses import _LOSSES, LossOutput

        def outputs(logits, labels, teacher, options, requested, context):
            self.assertEqual(requested, {"return_entropy": True})
            logp = logits.log_softmax(-1)
            return LossOutput(
                TokenLoss("ce")(logits, labels),
                {"entropy": -(logp.exp() * logp).sum(-1)},
            )

        def compute(calc):
            calc.run((0,))
            result = calc.evaluate(
                "entropy",
                TokenLoss("ce"),
                calc.outputs()["logits"],
                calc.labels,
                return_entropy=True,
            )
            calc.add(1, result.statistics["entropy"])

        with patch.dict(_LOSSES, {"ce": replace(_LOSSES["ce"], outputs=outputs)}):
            for length in (0, 3):
                hidden = torch.randn(length, 1, 5, requires_grad=True)
                labels = torch.zeros(length, 1, dtype=torch.long)
                p = Predictions(
                    hidden=hidden,
                    projection=Projection(lambda x: {"logits": x}),
                    labels=labels,
                    mask=torch.ones_like(labels),
                    positions=torch.zeros_like(labels),
                    compute=compute,
                    plan=LossPlan(
                        (LossTerm(), LossTerm("entropy", 0.2, TokenLoss("value")))
                    ),
                )
                actual = run(p, chunk=1)
                logp = hidden.log_softmax(-1)
                expected = (
                    TokenLoss("ce")(hidden, labels).sum()
                    - 0.2 * (logp.exp() * logp).sum()
                ) / max(length, 1)
                torch.testing.assert_close(actual.loss, expected)
                torch.testing.assert_close(
                    torch.autograd.grad(actual.loss, hidden)[0],
                    torch.autograd.grad(expected, hidden)[0],
                )

    def test_intermediate_and_final_losses_use_the_same_registry(self):
        hidden = torch.randn(3, 1, 5, requires_grad=True)
        labels = torch.zeros(3, 1, dtype=torch.long)
        calls = []
        reference = _IMPLEMENTATIONS["reference"]

        def observe(request, group):
            calls.append(request.unreduced)
            return reference.compute(request, group)

        def compute(calc):
            calc.run((0,))
            result = calc.evaluate(
                "ce", TokenLoss("ce"), calc.outputs()["logits"], calc.labels
            )
            # Intermediate values keep gradients until the algorithm detaches them.
            self.assertTrue(result.values.requires_grad)
            calc.add(1, result.values)
            with self.assertRaisesRegex(ValueError, "Repeated computation key"):
                calc.evaluate(
                    "ce", TokenLoss("ce"), calc.outputs()["logits"], calc.labels
                )

        p = Predictions(
            hidden=hidden,
            projection=Projection(lambda x: {"logits": x}),
            labels=labels,
            positions=torch.zeros_like(labels),
            mask=torch.ones_like(labels),
            compute=compute,
            plan=LossPlan((LossTerm(), LossTerm("second", loss=TokenLoss("value")))),
        )
        with patch.dict(
            _IMPLEMENTATIONS, {"reference": replace(reference, compute=observe)}
        ):
            result = run(p)
        self.assertEqual(calls, [False, True])
        expected = 2 * TokenLoss("ce")(hidden, labels).mean()
        torch.testing.assert_close(result.loss, expected)
        torch.testing.assert_close(
            torch.autograd.grad(result.loss, hidden)[0],
            torch.autograd.grad(expected, hidden)[0],
        )

    def test_trim_preserves_cross_unit_computation(self):
        for safe_projection in (False, True):
            hidden = torch.randn(4, 1, 5, requires_grad=True)
            prediction = Predictions(
                hidden=hidden,
                projection=Projection(
                    lambda x: {"logits": x - x.mean(0)},
                    chunk_safe=safe_projection,
                ),
                chunk_safe=not safe_projection,
                labels=torch.zeros(4, 1, dtype=torch.long),
                positions=torch.zeros(4, 1, dtype=torch.long),
                mask=torch.tensor([1.0, 0.0, 1.0, 0.0])[:, None],
            )
            results = []
            for trim in (False, True):
                result = execute_batch(
                    PredictionBatch((prediction,)),
                    TokenLoss(kind="ce"),
                    PositionWeighting(),
                    training=True,
                    execution=LossExecution(trim=trim, chunk_size=1),
                )
                (gradient,) = torch.autograd.grad(result.loss, hidden)
                results.append((result.loss, gradient))
            torch.testing.assert_close(results[0], results[1])

    def test_projection_execution_constraints(self):
        head = nn.Linear(4, 9)
        for options, expected_calls in (
            ({"chunk_safe": False}, [4, 4]),
            ({"recompute_safe": False}, [1, 1, 1, 1]),
        ):
            with self.subTest(options=options):
                project = Mock(side_effect=lambda x: {"logits": head(x)})
                p = Predictions(
                    hidden=torch.randn(4, 1, 4, requires_grad=True),
                    projection=Projection(project, **options),
                    labels=torch.zeros(4, 1, dtype=torch.long),
                    positions=torch.zeros(4, 1, dtype=torch.long),
                    mask=torch.ones(4, 1),
                )
                run(p, chunk=1).loss.backward()
                self.assertEqual(
                    [call.args[0].shape[0] for call in project.call_args_list],
                    expected_calls,
                )

                project.reset_mock()
                p = replace(
                    p,
                    chain_ids=torch.arange(4)[:, None],
                    sequence_ids=torch.zeros(4, 1, dtype=torch.long),
                )
                result = execute_batch(
                    PredictionBatch((p,)),
                    TokenLoss(kind="ce"),
                    PositionWeighting(kind="dpace"),
                    training=True,
                    execution=LossExecution(chunk_size=1),
                )
                result.loss.backward()
                self.assertEqual(
                    [call.args[0].shape[0] for call in project.call_args_list], [4]
                )

    def test_invalid_groups_and_kernel_errors(self):
        for groups in (((0, 0),), ((0, 2),), ((0, 1), (1, 0))):
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                LossPlan((LossTerm(), LossTerm()), groups)
        hidden = torch.randn(2, 1, 4, requires_grad=True)
        p = Predictions(
            hidden=hidden,
            projection=Projection(lambda x: {"logits": x}),
            mask=torch.ones(2, 1),
            labels=torch.zeros(2, 1, dtype=torch.long),
            positions=torch.zeros(2, 1, dtype=torch.long),
        )
        with patch.dict(
            _IMPLEMENTATIONS,
            {k: v for k, v in _IMPLEMENTATIONS.items() if v.category == "reference"},
            clear=True,
        ):

            @register_implementation("broken", supports=lambda req, group: None)
            def broken(req, group):
                raise RuntimeError("kernel failed")

            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                run(p, fused=True)

    def test_linear_ce_checks_installed_liger_version(self):
        from importlib.metadata import PackageNotFoundError

        kernel = SimpleNamespace(forward=lambda return_predicted_tokens=False: None)
        module = SimpleNamespace(LigerFusedLinearCrossEntropyFunction=kernel)
        with patch.dict(
            sys.modules, {"liger_kernel.ops.fused_linear_cross_entropy": module}
        ):
            for version in ("0.8.0", "0.8.1", "0.8.2", "0.8.3", "0.8.4", "0.9.0"):
                with (
                    self.subTest(version=version),
                    patch(
                        "specforge.algorithms.objectives.implementations.linear_ce.metadata.version",
                        return_value=version,
                    ),
                ):
                    if version in ("0.8.3", "0.8.4"):
                        self.assertIs(_linear_kernel.__wrapped__(), kernel)
                    else:
                        with self.assertRaisesRegex(
                            ValueError,
                            f"requires liger-kernel.*installed version is {version}",
                        ):
                            _linear_kernel.__wrapped__()
        with patch(
            "specforge.algorithms.objectives.implementations.linear_ce.metadata.version",
            side_effect=PackageNotFoundError("liger-kernel"),
        ):
            self.assertIsNone(_linear_kernel.__wrapped__())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_linear_ce_avoids_projection_and_preserves_gradients(self):
        if _linear_kernel() is None:
            self.skipTest("Liger linear CE with predicted tokens is unavailable")
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(9)
            head = nn.Linear(32, 97, device="cuda", dtype=dtype)
            hidden = torch.randn(
                7, 1, 32, device="cuda", dtype=dtype, requires_grad=True
            )
            project = Mock(side_effect=lambda x: {"logits": head(x)})
            labels = torch.randint(97, (7, 1), device="cuda")
            p = Predictions(
                hidden=hidden,
                projection=LinearProjection(project, head),
                labels=labels,
                positions=torch.zeros_like(labels),
                mask=torch.tensor([1, 0.5, 0, 1, 0.2, 1, 0], device="cuda")[:, None],
            )
            parameters = (hidden, *head.parameters())
            expected = run(p, chunk=2)
            gradients = torch.autograd.grad(expected.loss, parameters)
            project.reset_mock()
            entry = _IMPLEMENTATIONS["linear_ce"]
            optimized = Mock(wraps=entry.compute)
            with patch.dict(
                _IMPLEMENTATIONS, {entry.name: replace(entry, compute=optimized)}
            ):
                actual = run(p, fused=True, chunk=2)
                actual_grad = torch.autograd.grad(actual.loss, parameters)
            self.assertGreater(optimized.call_count, 0)
            project.assert_not_called()
            tolerance = 0.02 if dtype == torch.bfloat16 else 3e-4
            torch.testing.assert_close(
                actual.loss, expected.loss, rtol=tolerance, atol=2e-5
            )
            torch.testing.assert_close(
                actual_grad, gradients, rtol=tolerance, atol=tolerance * 0.02
            )
            torch.testing.assert_close(actual.ratios, expected.ratios)

            def compute(calc):
                calc.run((0,))
                calc.add(1, calc.predictions.hidden.square().mean(-1))

            regularized = replace(
                p,
                compute=compute,
                plan=LossPlan(
                    (LossTerm(), LossTerm("router", 0.1, TokenLoss("value")))
                ),
            )
            reference = run(regularized, chunk=2)
            reference_grad = torch.autograd.grad(reference.loss, parameters)
            project.reset_mock()
            actual = run(regularized, fused=True, chunk=2)
            project.assert_not_called()
            torch.testing.assert_close(
                actual.loss, reference.loss, rtol=tolerance, atol=2e-5
            )
            torch.testing.assert_close(
                torch.autograd.grad(actual.loss, parameters),
                reference_grad,
                rtol=tolerance,
                atol=tolerance * 0.02,
            )

            with patch(
                "specforge.algorithms.objectives.implementations.linear_ce._linear_kernel",
                return_value=None,
            ):
                with self.assertRaisesRegex(ValueError, "requires Liger"):
                    run(p, fused=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_domino_suffix_ce_with_an_independent_term(self):
        config, runtime = build_runtime("DominoDraftModel", options={"num_anchors": 3})
        runtime.cuda()
        for name in config.target_bindings:
            module = runtime.draft_model.get_submodule(name)
            module.load_target_weight(torch.randn_like(module.weight))
        batch = {key: value.cuda() for key, value in training_batch(config).items()}
        torch.manual_seed(21)
        p = next(
            iter(runtime.objective(runtime, batch, step=1, total_steps=10).predictions)
        )
        # A synthetic extension, not an additional Domino training objective.
        extra = LossTerm(
            "base_logits", 0.2, mask=p.mask * 0.5, denominator=p.mask.new_tensor(7.0)
        )
        p = replace(p, plan=replace(p.plan, terms=p.plan.terms + (extra,)))
        parameters = tuple(v for v in runtime.parameters() if v.requires_grad)
        expected = run(p, chunk=1)
        gradients = torch.autograd.grad(
            expected.loss, parameters, retain_graph=True, allow_unused=True
        )

        def compute(calc):
            calc.run((0, 1), impl="domino_suffix_ce", fused=False)
            calc.run((2,))

        prediction = replace(p, compute=compute)
        entry = _IMPLEMENTATIONS["domino_suffix_ce"]
        optimized = Mock(wraps=entry.compute)
        with patch.dict(
            _IMPLEMENTATIONS, {entry.name: replace(entry, compute=optimized)}
        ):
            actual = run(prediction, fused=True, chunk=1)
            actual_grad = torch.autograd.grad(
                actual.loss, parameters, retain_graph=True, allow_unused=True
            )
        self.assertGreater(optimized.call_count, 0)
        torch.testing.assert_close(actual.loss, expected.loss, rtol=3e-4, atol=3e-5)
        torch.testing.assert_close(actual_grad, gradients, rtol=5e-4, atol=3e-5)
        torch.testing.assert_close(actual.ratios, expected.ratios)
        for projection in (
            Projection(p.projection.project),
            replace(p.projection, block_size=p.projection.block_size + 1),
        ):
            with self.assertRaisesRegex(ValueError, "cannot execute"):
                run(replace(prediction, projection=projection), fused=True, chunk=1)
