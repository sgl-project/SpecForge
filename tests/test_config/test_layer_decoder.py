"""Layer decoding and registry contracts, independent of model wiring."""

import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

from specforge.config.draft import AutoDraftConfig, DFlashConfig
from specforge.config.draft.layers.decoder import decode_layers, legacy_layer_fields
from specforge.config.draft.layers.registry import build_component, register_component
from specforge.config.draft.layers.specs import GQASpec
from tests.test_modeling.test_all_draft_configs import architecture_payloads


class LayerDecoderTest(unittest.TestCase):
    def decode(self, **updates):
        config = AutoDraftConfig.from_inline(
            architecture_payloads()["DFlashDraftModel"]
        )
        return decode_layers(
            SimpleNamespace(**{**config.to_dict(), **updates}), DFlashConfig
        )

    def test_shared_layers_expand_to_immutable_tuple(self):
        shared = {"attention": {"sliding_window": 8}}
        layers = self.decode(num_hidden_layers=2, layers=shared)
        explicit = self.decode(num_hidden_layers=2, layers=[shared, shared])
        self.assertEqual(layers, explicit)
        self.assertIsInstance(layers, tuple)
        with self.assertRaises(FrozenInstanceError):
            layers[0].attention.sliding_window = 4
        shared["attention"]["sliding_window"] = 16
        self.assertEqual(layers[0].attention.sliding_window, 8)

    def test_unknown_fields_and_incompatible_contracts(self):
        for layer, message in (
            ({"attention": {"type": "unknown"}}, "Unknown"),
            ({"ffn": {"typo": 1}}, "Unknown"),
            ({"attention": {"is_causal": "true"}}, "boolean"),
        ):
            with self.subTest(layer=layer), self.assertRaisesRegex(ValueError, message):
                self.decode(layers=layer)

    def test_registered_builder_and_contract(self):
        register_component(
            "attention",
            "decoder_test",
            GQASpec,
            builders={"context_then_draft": lambda *, spec: spec.head_dim},
        )
        spec = self.decode(layers={"attention": {"type": "decoder_test"}})[0].attention
        self.assertEqual(build_component("attention", spec, "context_then_draft"), 4)
        with self.assertRaisesRegex(ValueError, "eagle_cache"):
            build_component("attention", spec, "eagle_cache")

    def test_legacy_export_preserves_full_swa_and_rejects_distinct_windows(self):
        layers = self.decode(
            num_hidden_layers=2,
            layers=[
                {"attention": {"is_causal": False}},
                {"attention": {"sliding_window": 8, "is_causal": True}},
            ],
        )
        flat, _ = legacy_layer_fields(layers)
        self.assertEqual(flat["layer_types"], ["full_attention", "sliding_attention"])
        self.assertEqual(flat["sliding_window"], 8)
        mixed = self.decode(
            num_hidden_layers=2,
            layers=[
                {"attention": {"sliding_window": 4}},
                {"attention": {"sliding_window": 8}},
            ],
        )
        with self.assertRaisesRegex(ValueError, "sliding windows"):
            legacy_layer_fields(mixed)
