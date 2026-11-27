"""Diagnostics are free, chunk invariant and absent on non-logging steps."""

import unittest
from dataclasses import replace
from functools import partial
from pkgutil import resolve_name
from unittest.mock import patch

import torch

from specforge.algorithms.metrics.domino import measure as domino_measure
from specforge.algorithms.metrics.dspark import measure as dspark_measure
from specforge.algorithms.objectives.diagnostics import (
    collect_block_metrics,
    collect_prediction_metrics,
    diagnostic_slices,
)
from specforge.algorithms.objectives.inputs import prepare_blocks
from specforge.algorithms.objectives.plan import LossPlan, LossTerm
from specforge.training.runner import _reduce_ratio_metrics
from tests.test_algorithms.test_objective_contracts import build_runtime
from tests.test_algorithms.test_objectives import training_batch


def rng_states(device):
    """Every generator a forward on this device can advance."""
    states = [torch.get_rng_state()]
    if device.type == "cuda":
        states.append(torch.cuda.get_rng_state(device))
    return states


def wide_batch(config, *, rows=2, length=140):
    """Fully supervised rows, long enough for more than 128 anchors each."""
    reference = training_batch(config)
    return dict(
        input_ids=torch.randint(0, config.vocab_size, (rows, length)),
        loss_mask=torch.ones(rows, length),
        attention_mask=torch.ones(rows, length, dtype=torch.long),
        **{
            name: torch.randn(rows, length, reference[name].shape[-1])
            for name in ("hidden_states", "target_last_hidden_states")
        },
    )


class BlockDiagnosticsTest(unittest.TestCase):
    def test_diagnostics_preserve_loss_gradients_and_rng(self):
        for architecture, weighting in (
            ("DFlashDraftModel", None),
            ("DFlash2DraftModel", None),
            # D-PACE diagnostics project a second time for the chain weights.
            ("DFlash2DraftModel", {"kind": "dpace"}),
            ("DominoDraftModel", None),
            ("DSparkDraftModel", None),
        ):
            with self.subTest(architecture=architecture, weighting=weighting):
                options = {"num_anchors": 4}
                if architecture in ("DFlashDraftModel", "DFlash2DraftModel"):
                    options["teacher_metrics"] = True
                config, runtime = build_runtime(
                    architecture, options=options, weighting=weighting
                )
                batch = training_batch(config)
                device = next(runtime.parameters()).device
                results, gradients, generators = [], [], []
                for enabled in (False, True):
                    runtime.zero_grad(set_to_none=True)
                    torch.manual_seed(48)
                    result = runtime(
                        batch,
                        step=0,
                        total_steps=10,
                        collect_detailed_metrics=enabled,
                    )
                    generators.append(rng_states(device))
                    result.loss.backward()
                    results.append(result)
                    gradients.append(
                        {
                            name: p.grad.clone()
                            for name, p in runtime.named_parameters()
                            if p.grad is not None
                        }
                    )
                # Identical, not close: the report runs after the loss graph
                # is built and may neither touch it nor draw random numbers.
                exact = dict(rtol=0, atol=0)
                torch.testing.assert_close(results[0].loss, results[1].loss, **exact)
                torch.testing.assert_close(gradients[0], gradients[1], **exact)
                torch.testing.assert_close(generators[0], generators[1], **exact)
                self.assertFalse(results[0].sums)
                self.assertTrue(
                    any(
                        name.endswith("/walk_accepted_length")
                        for name in results[1].ratios
                    )
                )
                self.assertTrue(
                    all(not v.requires_grad for v in results[1].sums.values())
                )
                runtime.execution = replace(runtime.execution, chunk_size=1)
                torch.manual_seed(48)
                chunked = runtime(batch, step=0, total_steps=10)
                torch.testing.assert_close(chunked.ratios, results[1].ratios)
                torch.testing.assert_close(chunked.sums, results[1].sums)

    def test_diagnostic_chunks_follow_the_loss_chunk(self):
        """Unset is the protocol's 128 blocks per row; an explicit zero is unsplit."""
        for architecture, target, units in (
            (
                "DFlashDraftModel",
                "specforge.algorithms.objectives.diagnostics.dflash_unary_diagnostics",
                lambda call: call.args[0].shape[1],
            ),
            (
                "DSparkDraftModel",
                "specforge.algorithms.objectives.protocols.dspark.measure",
                lambda call: call.args[1].shape[0],
            ),
        ):
            config, runtime = build_runtime(architecture, options={"num_anchors": 130})
            batch = wide_batch(config)
            for chunk_size, expected in (
                (None, [256, 4]),
                (0, [260]),
                (100, [100, 100, 60]),
            ):
                with self.subTest(architecture=architecture, chunk_size=chunk_size):
                    runtime.execution = replace(
                        runtime.execution, chunk_size=chunk_size
                    )
                    function = resolve_name(target)
                    with patch(target, wraps=function) as measured, torch.no_grad():
                        runtime(batch, step=0, total_steps=10)
                    self.assertEqual(
                        [units(call) for call in measured.call_args_list], expected
                    )

    def test_teacher_metrics_extend_the_capture_contract(self):
        from specforge.algorithms.objectives.registry import resolve_objective

        for architecture in ("DFlashDraftModel", "DFlash2DraftModel"):
            for enabled in (False, True):
                resolved = resolve_objective(
                    architecture, options={"teacher_metrics": enabled}
                )
                self.assertEqual(resolved.requires_teacher, enabled)
                self.assertEqual(
                    "target_last_hidden_states" in resolved.registration.inputs, enabled
                )

    def test_disabled_diagnostics_do_not_project_and_teacher_is_opt_in(self):
        config, runtime = build_runtime("DFlashDraftModel", options={"num_anchors": 4})
        batch = training_batch(config)
        with patch(
            "specforge.algorithms.objectives.diagnostics.collect_block_metrics"
        ) as report:
            runtime(batch, step=0, total_steps=10, collect_detailed_metrics=False)
            report.assert_not_called()
        result = runtime(batch, step=0, total_steps=10)
        self.assertFalse(
            any(name.startswith("dflash/teacher/") for name in result.ratios)
        )

    def test_zero_unit_block_diagnostics(self):
        for architecture, teacher_metrics, weighting in (
            ("DFlashDraftModel", False, None),
            ("DFlashDraftModel", True, {"kind": "dpace"}),
            ("DFlash2DraftModel", True, None),
            ("DFlash2DraftModel", False, {"kind": "dpace"}),
        ):
            with self.subTest(
                architecture=architecture,
                teacher_metrics=teacher_metrics,
                weighting=weighting,
            ):
                config, runtime = build_runtime(
                    architecture,
                    options={"num_anchors": 4, "teacher_metrics": teacher_metrics},
                    weighting=weighting,
                )
                batch = training_batch(config)
                blocks = prepare_blocks(config, batch, num_anchors=4)
                output = runtime.draft_model(
                    blocks.input_ids,
                    batch["hidden_states"],
                    blocks.layer_inputs,
                    output_logits=False,
                )
                selector = getattr(runtime.draft_model, "candidate_selector", None)
                extra = (
                    {"predecessors": blocks.predecessors}
                    if selector is not None
                    else {}
                )
                plan_kwargs = (
                    {"plan": LossPlan((LossTerm(), LossTerm("selector", 1.0)))}
                    if selector is not None
                    else {}
                )
                predictions = runtime.block_predictions(
                    output, blocks, batch, context_inputs=extra, **plan_kwargs
                )
                kwargs = dict(
                    chunk_size=0,
                    top_k=selector.top_k if selector is not None else 16,
                    selector=selector,
                    selector_alpha=0.5 if selector is not None else None,
                    teacher_metrics=teacher_metrics,
                )
                nonempty_ratios, nonempty_sums = collect_block_metrics(
                    runtime, predictions, blocks, **kwargs
                )
                empty_predictions = predictions.select(slice(0, 0))
                empty_blocks = replace(
                    blocks,
                    anchors=blocks.anchors[:, :0],
                    weights=blocks.weights[:, :0],
                    predecessors=blocks.predecessors[:, :0],
                )
                for chunk_size in (0, 4):
                    self.assertEqual(
                        diagnostic_slices(runtime, empty_predictions, chunk_size), []
                    )
                ratios, sums = collect_block_metrics(
                    runtime, empty_predictions, empty_blocks, **kwargs
                )
                self.assertEqual(set(ratios), set(nonempty_ratios))
                self.assertEqual(set(sums), set(nonempty_sums))
                for name, (num, den) in ratios.items():
                    self.assertEqual(num.shape, nonempty_ratios[name][0].shape)
                    self.assertEqual(den.shape, nonempty_ratios[name][1].shape)
                    self.assertTrue(torch.isfinite(num).all())
                    self.assertTrue(torch.isfinite(den).all())
                    if name != "selector_loss_alpha":
                        self.assertEqual(den.sum().item(), 0.0, name)
                        self.assertEqual(num.sum().item(), 0.0, name)
                for name, count in sums.items():
                    self.assertEqual(count.shape, nonempty_sums[name].shape)
                    self.assertEqual(count.sum().item(), 0.0, name)

    def test_zero_unit_prediction_diagnostics(self):
        for architecture, weighting, label_offset, prefix, plan, measure_fn, slots in (
            (
                "DSparkDraftModel",
                {"kind": "dpace"},
                1,
                True,
                LossPlan((LossTerm(), LossTerm("confidence"))),
                dspark_measure,
                slice(None),
            ),
            (
                "DominoDraftModel",
                None,
                0,
                False,
                LossPlan(
                    (
                        LossTerm("base_logits", 0.5, name="base"),
                        LossTerm("logits", 0.5, name="final"),
                    ),
                    joint_groups=((0, 1),),
                ),
                partial(domino_measure, base_weight=0.5),
                slice(1, None),
            ),
        ):
            with self.subTest(architecture=architecture):
                config, runtime = build_runtime(
                    architecture, options={"num_anchors": 4}, weighting=weighting
                )
                batch = training_batch(config)
                blocks = prepare_blocks(
                    config,
                    batch,
                    num_anchors=4,
                    label_offset=label_offset,
                    prefix=prefix,
                )
                output = runtime.draft_model(
                    blocks.input_ids,
                    batch["hidden_states"],
                    blocks.layer_inputs,
                    predecessor_ids=blocks.predecessors.flatten(1),
                    output_logits=False,
                )
                predictions = runtime.block_predictions(
                    output,
                    blocks,
                    batch,
                    projection_inputs={"predecessor_ids": blocks.predecessors},
                    plan=plan,
                )
                runtime.execution = replace(runtime.execution, chunk_size=0)
                nonempty_ratios, nonempty_sums = collect_prediction_metrics(
                    runtime, predictions, blocks, measure_fn, chunk_size=0, slots=slots
                )
                empty_ratios, empty_sums = collect_prediction_metrics(
                    runtime,
                    predictions.select(slice(0, 0)),
                    replace(blocks, anchors=blocks.anchors[:, :0]),
                    measure_fn,
                    chunk_size=0,
                    slots=slots,
                )
                self.assertEqual(set(empty_ratios), set(nonempty_ratios))
                self.assertEqual(empty_sums, nonempty_sums)
                for name, (num, den) in empty_ratios.items():
                    self.assertEqual(num.shape, nonempty_ratios[name][0].shape, name)
                    self.assertEqual(den.shape, nonempty_ratios[name][1].shape, name)
                    self.assertTrue(torch.isfinite(num).all(), name)
                    self.assertTrue(torch.isfinite(den).all(), name)
                    self.assertEqual(num.sum().item(), 0.0, name)
                    self.assertEqual(den.sum().item(), 0.0, name)
                reduce_kwargs = dict(
                    device=torch.device("cpu"), process_group=None, reduce=False
                )
                self.assertEqual(
                    set(_reduce_ratio_metrics(empty_ratios, **reduce_kwargs)),
                    set(_reduce_ratio_metrics(nonempty_ratios, **reduce_kwargs)),
                )


if __name__ == "__main__":
    unittest.main()
