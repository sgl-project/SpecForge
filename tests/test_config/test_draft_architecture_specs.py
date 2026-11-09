"""Shared protocol values do not imply inheritance between algorithms."""

import dataclasses
import unittest

from specforge.config.draft.architecture import DFlash2Spec, DFlashSpec
from specforge.config.draft.architecture.block import (
    AnchoredBlockSpec,
    DraftOnlyBlockSpec,
    ExternalFirstBlockSpec,
)
from specforge.config.draft.architecture.features import TargetFeatureSpec


def dflash2(**overrides):
    return DFlash2Spec(
        **dict(
            block=AnchoredBlockSpec(block_size=16),
            target_features=TargetFeatureSpec(
                mask_token_id=31, target_layer_ids=(1, 4, 7)
            ),
            selector_rank=8,
            selector_top_k=4,
            output_multiplier=1.0,
            final_logit_softcapping=None,
            **overrides,
        )
    )


class ArchitectureSpecsTest(unittest.TestCase):
    def test_each_protocol_owns_its_counts(self):
        cases = (
            (AnchoredBlockSpec(block_size=16), 16, 15, 0, 16, tuple(range(1, 16))),
            (DraftOnlyBlockSpec(block_size=7), 7, 7, 0, 8, tuple(range(7))),
            (ExternalFirstBlockSpec(block_size=7), 7, 6, 1, 8, tuple(range(1, 7))),
        )
        for node, slots, predictions, external, window, positions in cases:
            with self.subTest(node=node):
                self.assertEqual(node.num_model_slots, slots)
                self.assertEqual(node.num_draft_tokens, predictions)
                self.assertEqual(node.num_external_candidates, external)
                self.assertEqual(node.verify_window, window)
                self.assertEqual(node.prediction_positions, positions)

    def test_invalid_block_sizes_are_rejected(self):
        for schema in (AnchoredBlockSpec, DraftOnlyBlockSpec, ExternalFirstBlockSpec):
            for value in (True, 0, -1, 2.0, "2", None):
                with (
                    self.subTest(schema=schema, value=value),
                    self.assertRaisesRegex(ValueError, "block_size"),
                ):
                    schema(block_size=value)
        for schema in (AnchoredBlockSpec, DraftOnlyBlockSpec, ExternalFirstBlockSpec):
            with self.assertRaisesRegex(ValueError, "at least 2"):
                schema(block_size=1)

    def test_target_feature_order_and_width(self):
        node = TargetFeatureSpec(mask_token_id=0, target_layer_ids=(1, 4, 7))
        self.assertEqual(node.target_layer_ids, (1, 4, 7))
        self.assertEqual(node.input_width(96), 288)
        for value in ([], (), (True,), (-1,), ("1",), [1, 2], (7, 1, 7)):
            with self.assertRaisesRegex(ValueError, "target_layer_ids"):
                TargetFeatureSpec(mask_token_id=0, target_layer_ids=value)
        for value in (True, -1, 0.0, "0", None):
            with self.assertRaisesRegex(ValueError, "mask_token_id"):
                TargetFeatureSpec(mask_token_id=value, target_layer_ids=(1,))

    def test_architectures_compose_nodes_without_inheriting_each_other(self):
        second = dflash2()
        first = DFlashSpec(block=second.block, target_features=second.target_features)
        self.assertNotIn(DFlashSpec, DFlash2Spec.__mro__)
        self.assertNotIn(DFlash2Spec, DFlashSpec.__mro__)
        self.assertIs(first.protocol, second.protocol)
        self.assertEqual(first.protocol.num_draft_tokens, 15)
        for node in (first, second, first.block, first.target_features):
            self.assertEqual(hash(node), hash(dataclasses.replace(node)))
            with self.assertRaises(dataclasses.FrozenInstanceError):
                node.new_setting = 1

    def test_algorithm_fields_cannot_leak_into_component_or_training_settings(self):
        fields = dataclasses.asdict(dflash2())
        # Reconstruct typed children, rather than letting dictionaries into a frozen node.
        fields.update(
            block=AnchoredBlockSpec(block_size=16),
            target_features=TargetFeatureSpec(mask_token_id=31, target_layer_ids=(1,)),
        )
        for name in ("conv_kernel_size", "ce_weight", "replacement_probability"):
            with self.assertRaisesRegex(TypeError, name):
                DFlash2Spec(**fields, **{name: 1})
        for name in fields:
            with self.assertRaisesRegex(TypeError, name):
                DFlash2Spec(**{k: v for k, v in fields.items() if k != name})

    def test_mismatched_protocols_and_raw_children_are_rejected(self):
        for block in (
            DraftOnlyBlockSpec(block_size=16),
            ExternalFirstBlockSpec(block_size=16),
            {"block_size": 16},
        ):
            with self.assertRaisesRegex(ValueError, "AnchoredBlockSpec"):
                DFlashSpec(
                    block=block,
                    target_features=TargetFeatureSpec(
                        mask_token_id=0, target_layer_ids=(1,)
                    ),
                )
        with self.assertRaisesRegex(ValueError, "TargetFeatureSpec"):
            DFlashSpec(block=AnchoredBlockSpec(block_size=16), target_features={})

    def test_selector_and_logit_parameters_are_explicit(self):
        node = dflash2()
        for name in ("selector_rank", "selector_top_k"):
            for value in (True, 0, -1, 1.5, "8"):
                with self.assertRaisesRegex(ValueError, name):
                    dataclasses.replace(node, **{name: value})
        for name in ("output_multiplier", "final_logit_softcapping"):
            for value in (True, 0, -1, "8", float("nan"), float("inf")):
                with self.assertRaisesRegex(ValueError, name):
                    dataclasses.replace(node, **{name: value})
        self.assertEqual(
            dataclasses.replace(
                node, final_logit_softcapping=30
            ).final_logit_softcapping,
            30,
        )


if __name__ == "__main__":
    unittest.main()
