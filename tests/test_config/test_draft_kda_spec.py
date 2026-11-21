"""KDA is a recurrent attention node that reads target context as state."""

import dataclasses
import json
import unittest
from pathlib import Path

import yaml

from specforge.config.draft.attention import KDASpec
from specforge.config.draft.codec import encode_config, parse_draft
from specforge.config.draft.codec.components import decode_component, encode_component
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_dspine_spec import dspine_config
from tests.test_config.test_draft_eagle3_spec import eagle_config
from tests.test_config.test_draft_mtp_spec import mtp_config
from tests.test_config.test_draft_peagle_spec import peagle_config

PRESET = (
    Path(__file__).resolve().parents[2]
    / "specforge/config/draft/presets/qwen3.8-27b-dflash2-kda.yaml"
)
KDA = {
    "type": "kda",
    "rope_parameters": None,
    "num_heads": 4,
    "head_dim": 8,
    "short_conv_kernel_size": 4,
    "gate_lower_bound": -5.0,
    "full_rank_output_gate": False,
}


def with_kda(config, layers=(0,), node=None):
    """Give the chosen layers a KDA node, dropping definitions left unused."""
    first = config.layer_configs[config.layer_types[0]]
    layer_types = tuple(
        "kda" if index in layers else name
        for index, name in enumerate(config.layer_types)
    )
    layer_configs = {
        name: layer
        for name, layer in (
            dict(config.layer_configs)
            | {"kda": dataclasses.replace(first, attention="kda")}
        ).items()
        if name in layer_types
    }
    attentions = {layer.attention for layer in layer_configs.values()}
    return dataclasses.replace(
        config,
        attention_configs={
            name: attention
            for name, attention in (
                dict(config.attention_configs)
                | {"kda": node or decode_component("attention", KDA)}
            ).items()
            if name in attentions
        },
        layer_configs=layer_configs,
        layer_types=layer_types,
    )


class KDASpecTest(unittest.TestCase):
    def test_round_trip_and_capabilities(self):
        for bound, full_rank in ((-5.0, False), (None, True), (-2, False)):
            payload = KDA | {
                "gate_lower_bound": bound,
                "full_rank_output_gate": full_rank,
            }
            with self.subTest(bound=bound, full_rank=full_rank):
                node = decode_component("attention", payload)
                self.assertIsInstance(node, KDASpec)
                self.assertEqual(node.projection_size, 32)
                self.assertIsNone(node.rotary_input_dim)
                self.assertEqual(node.capabilities(), {"target_context_state"})
                self.assertEqual(encode_component("attention", node), payload)
                self.assertEqual(
                    node,
                    decode_component("attention", json.loads(json.dumps(payload))),
                )

    def test_invalid_values_are_rejected(self):
        for name in ("num_heads", "head_dim", "short_conv_kernel_size"):
            for value in (False, 0, -1, 2.0, "2"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        decode_component("attention", KDA | {name: value})
        for value in (0.0, 1.0, float("nan"), "low", True):
            with self.subTest(gate_lower_bound=value):
                with self.assertRaisesRegex(ValueError, "gate_lower_bound"):
                    decode_component("attention", KDA | {"gate_lower_bound": value})
        with self.assertRaisesRegex(ValueError, "full_rank_output_gate"):
            decode_component("attention", KDA | {"full_rank_output_gate": 1})
        with self.assertRaisesRegex(ValueError, "unknown|unexpected"):
            decode_component("attention", KDA | {"context_state": "scan"})

    def test_rope_override_has_no_rotary_subspace(self):
        rope = {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "interleaved": False,
        }
        node = decode_component("attention", KDA | {"rope_parameters": rope})
        with self.assertRaisesRegex(ValueError, "no effect"):
            with_kda(draft_config(), node=node)


class KDAArchitectureTest(unittest.TestCase):
    def test_architectures_without_context_blocks_reject_kda(self):
        for config in (eagle_config(), peagle_config(), mtp_config(), dspine_config()):
            with self.subTest(architecture=config.architecture):
                with self.assertRaisesRegex(
                    ValueError, "variable_input_width|projected_kv|causal"
                ):
                    with_kda(config, layers=range(len(config.layer_types)))

    def test_preset_round_trips_the_hybrid_stack(self):
        config = parse_draft(yaml.safe_load(PRESET.read_text()), target_config=None)
        self.assertEqual(
            [config.layer_configs[name].attention for name in config.layer_types],
            ["kda", "kda", "sliding_attention", "kda", "kda"],
        )
        self.assertEqual(parse_draft(encode_config(config), target_config=None), config)
