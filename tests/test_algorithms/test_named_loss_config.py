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
            {"terms": {"token": {"options": {"ce_weight": -0.1}}}},
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

    def test_old_options_resolve_to_the_same_named_terms(self):
        cases = (
            (
                "DSparkDraftModel",
                {"ce_loss_alpha": 0.3, "confidence_loss_alpha": 0.2},
                {
                    "token": {"options": {"ce_weight": 0.3}},
                    "confidence": {"coefficient": 0.2},
                },
            ),
            (
                "DSpineDraftModel",
                {"ce_weight": 0.2, "alignment_weight": 0.4},
                {
                    "token": {"options": {"ce_weight": 0.2, "l1_weight": 0.8}},
                    "refinement": {"options": {"ce_weight": 0.2, "l1_weight": 0.8}},
                    "alignment": {"coefficient": 0.4},
                },
            ),
            (
                "DFlash2DraftModel",
                {"selector_loss_alpha": 0},
                {"selector": {"coefficient": 0}},
            ),
        )
        for architecture, options, terms in cases:
            with self.subTest(architecture=architecture):
                old = resolve_objective(
                    architecture, options=options, execution={"fused": True}
                )
                new = resolve_objective(
                    architecture, loss={"fused": True, "terms": terms}
                )
                self.assertEqual(old.losses, new.losses)
                self.assertEqual(old.objective, new.objective)
                self.assertEqual(old.frozen_parameters, new.frozen_parameters)
        domino = resolve_objective(
            "DominoDraftModel", options={"impl": "domino_suffix_ce"}
        )
        self.assertTrue(domino.execution.fused)
        self.assertFalse(hasattr(domino.objective, "impl"))
        with self.assertRaises(ValueError):
            resolve_objective("DSpineDraftModel", options={"ce_weight": 1.1})

    def test_previous_checkpoint_metadata_is_normalized_once(self):
        from specforge.algorithms.objectives.compatibility import (
            normalize_loss_checkpoint,
        )

        saved = {
            "canonical_objective": "dspine",
            "canonical_objective_options": {"ce_weight": 0.2, "alignment_weight": 0.4},
            "canonical_loss": {
                "kind": "ce_l1",
                "options": {"ce_weight": 0.2, "l1_weight": 0.8},
            },
            "canonical_loss_execution": {"fused": True, "chunk_size": 1},
            "canonical_loss_implementations": {
                "0/compute/refinement:0": "token_ce",
                "0:1": "value",
            },
        }
        normalized = normalize_loss_checkpoint(saved, "dspine")
        self.assertEqual(
            normalized["canonical_loss"]["terms"]["alignment"]["coefficient"], 0.4
        )
        self.assertNotIn("ce_weight", normalized["canonical_objective_options"])
        self.assertEqual(
            normalized["canonical_loss_implementations"],
            {
                "0/compute/refinement:0": "token_ce_l1",
                "0:1": "value",
                "0/compute/alignment:0": "reference",
            },
        )
        self.assertEqual(normalize_loss_checkpoint(normalized, "dspine"), normalized)
        self.assertIn("ce_weight", saved["canonical_objective_options"])
        dspark = {
            "canonical_objective_options": {
                "ce_loss_alpha": 0.3,
                "confidence_loss_alpha": 0.2,
            },
            "canonical_loss": {
                "kind": "ce_l1",
                "options": {"ce_weight": 0.3, "l1_weight": 0.9},
            },
        }
        migrated = normalize_loss_checkpoint(dspark, "dspark")
        self.assertEqual(
            migrated["canonical_loss"]["terms"]["confidence"]["coefficient"], 0.2
        )
        self.assertEqual(
            migrated["canonical_loss"]["terms"]["token"]["options"]["ce_weight"], 0.3
        )

    def test_flat_and_named_defaults_preserve_algorithm_losses_and_gradients(self):
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
