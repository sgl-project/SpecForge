"""Named terms keep mathematical choices, execution and schedules separate."""

import unittest

import torch

from specforge.algorithms.objectives.registry import resolve_objective
from tests.test_algorithms.test_objective_contracts import build_runtime
from tests.test_algorithms.test_objectives import configurations, training_batch


class NamedLossTest(unittest.TestCase):
    def test_configuration_boundaries(self):
        settings = resolve_objective(
            "DSparkDraftModel",
            loss={
                "fused": True,
                "terms": {
                    "confidence": {"coefficient": 0.2},
                },
                "scheduling": {"chunk_size": 2},
            },
        )
        self.assertTrue(settings.losses.fused)
        self.assertEqual(settings.execution.chunk_size, 2)
        self.assertTrue(settings.execution.fused)
        cases = [
            {"terms": {"missing": {}}},
            {"terms": {"confidence": {"kind": "ce"}}},
            {"terms": {"token": {"execution": {"impl": "token_ce"}}}},
            {"terms": {"token": {"execution": {"impl": "x", "fused": True}}}},
            {"terms": {"token": {"execution": {"chunk_size": 1}}}},
            {"fused": "true"},
            {"scheduling": {"fused": True}},
            {"groups": {"g": {"terms": ["token", "confidence"]}}},
        ]
        for payload in cases:
            with (
                self.subTest(payload=payload),
                self.assertRaises((ValueError, TypeError)),
            ):
                resolve_objective("DSparkDraftModel", loss=payload)
        with self.assertRaisesRegex(ValueError, "loss_execution"):
            resolve_objective(
                "DSparkDraftModel", loss={"terms": {}}, execution={"fused": True}
            )
        with self.assertRaisesRegex(ValueError, "loss.terms"):
            resolve_objective(
                "DSparkDraftModel",
                loss={"terms": {}},
                options={"confidence_loss_alpha": 0.2},
            )

    def test_settings_round_trip(self):
        first = resolve_objective("DominoDraftModel", loss={"fused": True})
        saved = first.losses.to_dict()
        self.assertNotIn("groups", saved)
        self.assertNotIn("execution", saved["terms"]["base"])
        second = resolve_objective("DominoDraftModel", loss=saved)
        self.assertEqual(second.losses.to_dict(), saved)

    def test_defaults_preserve_all_algorithm_losses_and_gradients(self):
        for config in configurations():
            with self.subTest(architecture=config.architecture):
                from specforge.algorithms.objectives.registry import (
                    default_protocol,
                    protocol_registrations,
                )

                factory = protocol_registrations()[
                    default_protocol(config.architecture)
                ].factory
                selection = (
                    {"num_anchors": 2}
                    if "num_anchors" in factory.__dataclass_fields__
                    else {}
                )
                config, legacy = build_runtime(config.architecture, options=selection)
                _, named = build_runtime(
                    config.architecture, options=selection, loss={"terms": {}}
                )
                batch = training_batch(config)
                outputs = []
                for runtime in (legacy, named):
                    torch.manual_seed(31)
                    result = runtime(batch, step=1, total_steps=10)
                    result.loss.backward()
                    outputs.append(
                        (
                            result.loss,
                            {
                                name: p.grad
                                for name, p in runtime.named_parameters()
                                if p.grad is not None
                            },
                        )
                    )
                torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-5)

    def test_auxiliary_formula_coefficient_and_freezing(self):
        from unittest.mock import patch

        from specforge.algorithms.objectives.losses import _LOSSES, register_loss

        with patch.dict(_LOSSES):

            @register_loss("binary_square", input_kind="binary")
            def binary_square(logits, labels, teacher=None):
                return (logits.sigmoid() - labels).square()

            config, runtime = build_runtime(
                "DSparkDraftModel",
                options={"num_anchors": 2},
                loss={
                    "terms": {
                        "confidence": {"kind": "binary_square", "coefficient": 0.25}
                    }
                },
            )
            batch = training_batch(config)
            torch.manual_seed(11)
            p = next(
                iter(
                    runtime.objective(
                        runtime, batch, step=1, total_steps=10
                    ).predictions
                )
            )
            self.assertEqual(p.plan.terms[1].loss.kind, "binary_square")
            self.assertEqual(p.plan.terms[1].coefficient, 0.25)
            result = runtime(batch, step=1, total_steps=10)
            result.loss.backward()
            self.assertTrue(
                any(
                    p.grad is not None
                    for p in runtime.draft_model.confidence_head.parameters()
                )
            )
        _, disabled = build_runtime(
            "DSparkDraftModel", loss={"terms": {"confidence": {"coefficient": 0}}}
        )
        self.assertFalse(
            any(
                p.requires_grad
                for p in disabled.draft_model.confidence_head.parameters()
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_shared_fusion_preserves_losses_and_gradients(self):
        for architecture, terms in (
            ("DSparkDraftModel", {}),
            ("DominoDraftModel", {}),
            ("DominoDraftModel", {"final": {"kind": "kl"}}),
        ):
            outputs = []
            for fused in (False, True):
                with self.subTest(architecture=architecture, terms=terms, fused=fused):
                    config, runtime = build_runtime(
                        architecture,
                        options={"num_anchors": 2},
                        loss={
                            "fused": fused,
                            "terms": terms,
                            "scheduling": {"chunk_size": 1, "checkpoint": True},
                        },
                    )
                    runtime.cuda()
                    torch.manual_seed(17)
                    for name in config.target_bindings:
                        module = runtime.draft_model.get_submodule(name)
                        module.load_target_weight(torch.randn_like(module.weight))
                    batch = {k: v.cuda() for k, v in training_batch(config).items()}
                    torch.manual_seed(31)
                    result = runtime(batch, step=1, total_steps=10)
                    result.loss.backward()
                    outputs.append(
                        (
                            result.loss,
                            {
                                name: p.grad
                                for name, p in runtime.named_parameters()
                                if p.grad is not None
                            },
                        )
                    )
                    names = runtime.loss_implementations.names
                    if not fused:
                        self.assertFalse(names)
                    elif architecture == "DSparkDraftModel":
                        self.assertEqual(names["0:0"], "token_ce_l1")
                        self.assertEqual(names["0/compute/confidence:0"], "reference")
                    elif terms:
                        self.assertEqual(names["0:0,1"], "split")
                    else:
                        self.assertEqual(names["0:0,1"], "domino_suffix_ce")
            torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-4, atol=1e-5)
