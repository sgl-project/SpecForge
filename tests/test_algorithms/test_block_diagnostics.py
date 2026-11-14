"""Diagnostics are detached, chunk invariant and absent on non-logging steps."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from tests.test_algorithms.test_objective_contracts import build_runtime
from tests.test_algorithms.test_objectives import training_batch


class BlockDiagnosticsTest(unittest.TestCase):
    def test_diagnostics_preserve_loss_and_gradients(self):
        for architecture in (
            "DFlashDraftModel",
            "DFlash2DraftModel",
            "DominoDraftModel",
            "DSparkDraftModel",
        ):
            with self.subTest(architecture=architecture):
                options = {"num_anchors": 4}
                if architecture in ("DFlashDraftModel", "DFlash2DraftModel"):
                    options["teacher_metrics"] = True
                config, runtime = build_runtime(architecture, options=options)
                batch = training_batch(config)
                results, gradients = [], []
                for enabled in (False, True):
                    runtime.zero_grad(set_to_none=True)
                    torch.manual_seed(48)
                    result = runtime(
                        batch,
                        step=0,
                        total_steps=10,
                        collect_detailed_metrics=enabled,
                    )
                    result.loss.backward()
                    results.append(result)
                    gradients.append(
                        {
                            name: p.grad.clone()
                            for name, p in runtime.named_parameters()
                            if p.grad is not None
                        }
                    )
                torch.testing.assert_close(results[0].loss, results[1].loss)
                torch.testing.assert_close(gradients[0], gradients[1])
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
