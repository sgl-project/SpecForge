"""Diagnostics are free, chunk invariant and absent on non-logging steps."""

import unittest
from dataclasses import replace
from pkgutil import resolve_name
from unittest.mock import patch

import torch

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


if __name__ == "__main__":
    unittest.main()
