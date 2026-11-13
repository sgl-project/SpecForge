"""Optimized execution preserves supervision, reductions and parameter gradients."""

import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import torch
from torch import nn

from specforge.algorithms.objectives.execution import execute_batch
from specforge.algorithms.objectives.execution_options import LossExecution
from specforge.algorithms.objectives.losses import _LOSSES, PositionWeighting, TokenLoss
from specforge.algorithms.objectives.predictions import PredictionBatch, Predictions
from tests.test_algorithms.test_objective_contracts import build_runtime
from tests.test_algorithms.test_objectives import training_batch


class LossExecutionTest(unittest.TestCase):
    def test_chunk_checkpoint_and_trim_preserve_objectives(self):
        for architecture in (
            "DFlashDraftModel",
            "DFlash2DraftModel",
            "DominoDraftModel",
            "DSparkDraftModel",
            "DSpineDraftModel",
            "Eagle3DraftModel",
            "PEagleDraftModel",
            "MTPDraftModel",
            "DMTPDraftModel",
        ):
            options = (
                {"num_anchors": 3}
                if architecture
                not in ("Eagle3DraftModel", "PEagleDraftModel", "MTPDraftModel")
                else {}
            )
            config, runtime = build_runtime(architecture, options=options)
            batch = training_batch(config)
            for weighting in (None, {"kind": "dpace"}):
                if weighting and architecture == "MTPDraftModel":
                    continue
                if weighting:
                    runtime.weighting = PositionWeighting(**weighting)
                expected = None
                for checkpoint, trim, chunk in (
                    (False, False, 0),
                    (False, False, 1),
                    (True, True, 1),
                ):
                    with self.subTest(
                        architecture=architecture,
                        weighting=weighting,
                        checkpoint=checkpoint,
                    ):
                        runtime.execution = LossExecution(
                            chunk_size=chunk, checkpoint=checkpoint, trim=trim
                        )
                        runtime.zero_grad(set_to_none=True)
                        torch.manual_seed(19)
                        result = runtime(
                            batch,
                            step=1,
                            total_steps=10,
                            alignment_ce=torch.tensor(0.5),
                        )
                        result.loss.backward()
                        actual = (
                            result.loss.detach(),
                            result.denominator,
                            {
                                n: p.grad.clone()
                                for n, p in runtime.named_parameters()
                                if p.grad is not None
                            },
                            result.ratios,
                        )
                        if expected is None:
                            expected = actual
                        else:
                            torch.testing.assert_close(
                                actual, expected, rtol=5e-4, atol=3e-5
                            )

    def test_compact_teacher_and_trim_do_not_project_full_vocabulary(self):
        for architecture in ("Eagle3DraftModel", "PEagleDraftModel"):
            config, runtime = build_runtime(architecture)
            batch = training_batch(config)
            batch.pop("target_logits", None)
            expected = None
            for compact in (None, 7):
                with self.subTest(architecture=architecture, chunk=compact):
                    runtime.execution = LossExecution(
                        chunk_size=2,
                        checkpoint=True,
                        teacher_chunk_size=compact,
                        trim=bool(compact),
                    )
                    runtime.zero_grad(set_to_none=True)
                    torch.manual_seed(23)
                    with patch.object(
                        runtime.target_head,
                        "forward",
                        wraps=runtime.target_head.forward,
                    ) as head:
                        result = runtime(batch, step=0, total_steps=10)
                        result.loss.backward()
                    actual = (
                        result.loss.detach(),
                        result.ratios,
                        {
                            n: p.grad.clone()
                            for n, p in runtime.named_parameters()
                            if p.grad is not None
                        },
                    )
                    if compact is None:
                        self.assertGreater(head.call_count, 0)
                        expected = actual
                    else:
                        self.assertEqual(head.call_count, 0)
                        torch.testing.assert_close(
                            actual, expected, rtol=3e-4, atol=2e-5
                        )

    def test_empty_trim_and_fractional_weights(self):
        hidden = torch.randn(5, 1, 8, requires_grad=True)
        head = nn.Linear(8, 17, bias=False)
        for mask in (
            torch.zeros(5, 1),
            torch.tensor([[0.2], [0.0], [0.5], [0.0], [0.1]]),
        ):
            predictions = Predictions(
                hidden=hidden,
                projector=lambda h: {"logits": head(h)},
                mask=mask,
                positions=torch.zeros(5, 1, dtype=torch.long),
                labels=torch.zeros(5, 1, dtype=torch.long),
                denominator=torch.tensor(5.0),
            )
            values = []
            for trim in (False, True):
                head.zero_grad(set_to_none=True)
                hidden.grad = None
                with patch.object(head, "forward", wraps=head.forward) as calls:
                    result = execute_batch(
                        PredictionBatch((predictions,), reduction="sum_means"),
                        TokenLoss(kind="ce"),
                        PositionWeighting(),
                        training=True,
                        execution=LossExecution(chunk_size=2, trim=trim),
                    )
                    result.loss.backward()
                values.append(
                    (
                        result.loss.detach(),
                        hidden.grad.clone(),
                        head.weight.grad.clone(),
                    )
                )
                self.assertTrue(
                    all(call.args[0].shape[0] <= 2 for call in calls.call_args_list)
                )
            torch.testing.assert_close(values[0], values[1])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fused_losses_match_eager_values_and_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            logits = torch.randn(
                3, 2, 67, device="cuda", dtype=dtype, requires_grad=True
            )
            teacher = torch.randn_like(logits)
            labels = torch.randint(67, (3, 2), device="cuda")
            for kind in (
                "ce",
                "soft_ce",
                "ce_l1",
                "kl",
                "reverse_kl",
                "js",
                "tv",
                "nla",
                "lk",
                "lk_lambda",
            ):
                with self.subTest(dtype=dtype, kind=kind):
                    loss = TokenLoss(kind=kind)
                    # LK uses complete-set statistics even when chunking is requested.
                    from specforge.algorithms.objectives.losses import LossContext

                    context = LossContext(torch.ones_like(labels))
                    reference = loss(logits, labels, teacher=teacher, context=context)
                    (expected_grad,) = torch.autograd.grad(reference.sum(), logits)
                    compute = loss.registration.fused_compute
                    optimized = Mock(wraps=compute)
                    with patch.dict(
                        _LOSSES,
                        {kind: replace(loss.registration, fused_compute=optimized)},
                    ):
                        actual = loss(
                            logits, labels, teacher=teacher, context=context, fused=True
                        )
                    optimized.assert_called_once()
                    (actual_grad,) = torch.autograd.grad(actual.sum(), logits)
                    self.assertIsNotNone(compute)
                    tolerance = 0.02 if dtype == torch.bfloat16 else 2e-4
                    torch.testing.assert_close(
                        actual, reference, rtol=tolerance, atol=2e-5
                    )
                    torch.testing.assert_close(
                        actual_grad,
                        expected_grad,
                        rtol=tolerance,
                        atol=tolerance * 0.01,
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_canonical_fused_convolution_values_and_gradients(self):
        from specforge.config.draft.conv import GroupedCausalConvSpec
        from specforge.modeling.draft.layers.conv import triton as kernels
        from specforge.modeling.draft.layers.conv.grouped_causal import (
            GroupedCausalConv,
        )
        from specforge.modeling.draft.layers.kernels import configure_kernels

        conv = GroupedCausalConv(
            GroupedCausalConvSpec(kernel_size=3, group_size=4),
            hidden_size=32,
            block_size=4,
            device="cuda",
            dtype=torch.float32,
        )
        with torch.no_grad():
            conv.base_kernel.normal_(std=0.1)
            conv.kernel_projection.weight.normal_(std=0.05)
        hidden = torch.randn(
            2, 8, 32, device="cuda", dtype=torch.float32, requires_grad=True
        )
        results = []
        for fused in (False, True):
            configure_kernels(conv, use_fused_conv=fused)
            with patch.object(
                kernels,
                "dflash2_grouped_conv_fused",
                wraps=kernels.dflash2_grouped_conv_fused,
            ) as kernel:
                first, coefficients = conv.prepare(hidden)
                output = conv.finish(first.tanh(), coefficients)
                gradients = torch.autograd.grad(
                    output.float().square().sum(), (hidden, *conv.parameters())
                )
            self.assertEqual(kernel.call_count, 2 if fused else 0)
            results.append((output, gradients))
        tolerance = 2e-4
        torch.testing.assert_close(
            results[1], results[0], rtol=tolerance, atol=tolerance * 0.1
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fused_auxiliary_and_domino_objectives(self):
        from specforge.core import domino_loss

        for architecture in (
            "DFlash2DraftModel",
            "DominoDraftModel",
            "DSparkDraftModel",
            "DSpineDraftModel",
        ):
            config, runtime = build_runtime(architecture, options={"num_anchors": 3})
            runtime.cuda()
            for name in config.target_bindings:
                runtime.draft_model.get_submodule(name).load_target_weight(
                    torch.randn(config.vocab_size, config.hidden_size, device="cuda")
                )
            batch = {k: v.cuda() for k, v in training_batch(config).items()}
            expected = None
            for fused in (False, True):
                with self.subTest(architecture=architecture, fused=fused):
                    runtime.execution = LossExecution(
                        chunk_size=1, checkpoint=True, fused=fused
                    )
                    runtime.zero_grad(set_to_none=True)
                    torch.manual_seed(31)
                    with patch.object(
                        domino_loss,
                        "domino_weighted_cross_entropy",
                        wraps=domino_loss.domino_weighted_cross_entropy,
                    ) as kernel:
                        result = runtime(
                            batch,
                            step=1,
                            total_steps=10,
                            alignment_ce=torch.tensor(0.5, device="cuda"),
                        )
                        result.loss.backward()
                    actual = (
                        result.loss.detach(),
                        {
                            n: p.grad.clone()
                            for n, p in runtime.named_parameters()
                            if p.grad is not None
                        },
                    )
                    if not fused:
                        expected = actual
                    else:
                        torch.testing.assert_close(
                            actual, expected, rtol=8e-4, atol=3e-5
                        )
                        if architecture == "DominoDraftModel":
                            self.assertGreater(kernel.call_count, 0)

    def test_execution_options_reject_invalid_values(self):
        for values in (
            {"chunk_size": -1},
            {"teacher_chunk_size": 0},
            {"checkpoint": 1},
            {"fused": "yes"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                LossExecution(**values)
