"""Named configurations own structure without introducing layer dispatch."""

import copy
import dataclasses
import unittest

from specforge.config.draft.attention import GQASpec
from specforge.config.draft.conv import GroupedCausalConvSpec
from specforge.config.draft.immutable import FrozenMap
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.mlp import DenseFFNSpec
from specforge.config.draft.validate import validate_layers


def definitions():
    attention = GQASpec(
        qk_norm_position="before_rope",
        qk_norm_scope="head",
        qk_norm_config=None,
        rope_parameters=None,
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=16,
        qk_norm=True,
        attention_bias=False,
        output_gate=False,
        is_causal=False,
        sliding_window=None,
        attention_dropout=0.0,
    )
    return dict(
        hidden_size=96,
        attention_configs={
            "full": attention,
            "local": dataclasses.replace(attention, is_causal=True, sliding_window=128),
        },
        mlp_configs={"dense": DenseFFNSpec(intermediate_size=256, hidden_act="silu")},
        conv_configs={"grouped": GroupedCausalConvSpec(kernel_size=3, group_size=16)},
        layer_configs={
            "full_dense": LayerSpec(attention="full", mlp="dense"),
            "local_dense": LayerSpec(
                attention="local", mlp="dense", sublayer_conv="grouped"
            ),
        },
        layer_types=("full_dense", "local_dense", "full_dense"),
        num_model_slots=8,
    )


class LayerDefinitionsTest(unittest.TestCase):
    def test_named_maps_snapshot_input_and_ignore_definition_order(self):
        source = definitions()["layer_configs"]
        snapshot = FrozenMap(source)
        reverse = FrozenMap(dict(reversed(list(source.items()))))
        self.assertEqual(snapshot, reverse)
        self.assertEqual(hash(snapshot), hash(reverse))
        self.assertEqual(dict(snapshot), source)
        self.assertEqual(copy.deepcopy(snapshot), snapshot)
        before = hash(snapshot)
        source.clear()
        self.assertEqual(len(snapshot), 2)
        self.assertEqual(hash(snapshot), before)
        with self.assertRaises(TypeError):
            snapshot["new"] = LayerSpec(attention="full", mlp="dense")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snapshot._entries = ()
        with self.assertRaises(KeyError):
            snapshot["missing"]

    def test_invalid_name_tables(self):
        for source in ([], None, {1: 1}, {"": 1}, {"x": []}, {"x": {}}):
            with self.subTest(source=source), self.assertRaises(ValueError):
                FrozenMap(source)
        self.assertEqual(len(FrozenMap({})), 0)

    def test_hybrid_and_repeated_layers_are_valid(self):
        config = definitions()
        validate_layers(**config)
        for name in (
            "attention_configs",
            "mlp_configs",
            "conv_configs",
            "layer_configs",
        ):
            config[name] = FrozenMap(config[name])
        validate_layers(**config)
        self.assertEqual(len(config["layer_types"]), 3)
        self.assertEqual(len(config["layer_configs"]), 2)
        # Projection width may differ from hidden_size; linear projections handle it.
        self.assertNotEqual(config["hidden_size"], 8 * 16)

    def test_unknown_references_name_the_field(self):
        for field, expected in (
            ("attention", "attention"),
            ("mlp", "mlp"),
            ("sublayer_conv", "conv"),
        ):
            config = definitions()
            config["layer_configs"]["local_dense"] = dataclasses.replace(
                config["layer_configs"]["local_dense"], **{field: "missing"}
            )
            with self.assertRaisesRegex(
                ValueError, f"layer_configs.local_dense.{field}: unknown {expected}"
            ):
                validate_layers(**config)
        config = definitions()
        config["layer_types"] = ("missing",)
        with self.assertRaisesRegex(ValueError, r"layer_types\[0\].*unknown layer"):
            validate_layers(**config)

    def test_unused_definitions_are_rejected(self):
        for kind in ("attention", "mlp", "conv", "layer"):
            config = definitions()
            table = config[kind + "_configs"]
            table["unused"] = next(iter(table.values()))
            with self.assertRaisesRegex(ValueError, f"{kind}_configs: unused.*unused"):
                validate_layers(**config)

    def test_layer_order_must_be_explicit(self):
        for value in ((), [], "full_dense", (None,), ("",)):
            config = {**definitions(), "layer_types": value}
            with self.assertRaisesRegex(ValueError, "layer_types"):
                validate_layers(**config)

    def test_convolution_uses_model_slots_and_hidden_width(self):
        for changes, error in (
            ({"hidden_size": 95}, "group_size"),
            ({"num_model_slots": 2}, "kernel_size"),
            ({"num_model_slots": None}, "block protocol"),
        ):
            with self.assertRaisesRegex(ValueError, error):
                validate_layers(**{**definitions(), **changes})
        validate_layers(**{**definitions(), "num_model_slots": 3})

    def test_nodes_cannot_be_replaced_by_raw_dictionaries(self):
        for kind in ("attention", "mlp", "conv", "layer"):
            config = definitions()
            table = config[kind + "_configs"]
            table[next(iter(table))] = {}
            with self.assertRaises(ValueError):
                validate_layers(**config)

    def test_unregistered_subclasses_cannot_reuse_a_type_name(self):
        @dataclasses.dataclass(frozen=True, kw_only=True)
        class PatchedGQA(GQASpec):
            pass

        config = definitions()
        full = config["attention_configs"]["full"]
        config["attention_configs"]["full"] = PatchedGQA(**dataclasses.asdict(full))
        with self.assertRaisesRegex(
            ValueError, "attention_configs.full: expected GQASpec"
        ):
            validate_layers(**config)

    def test_an_unused_layer_is_reported_before_the_components_it_strands(self):
        config = definitions()
        attention = config["attention_configs"]
        attention["orphan_attention"] = attention["full"]
        config["layer_configs"]["orphan"] = LayerSpec(
            attention="orphan_attention", mlp="dense"
        )
        with self.assertRaisesRegex(
            ValueError,
            r"^layer_configs: unused definitions \['orphan'\]; "
            r"attention_configs: unused definitions \['orphan_attention'\]$",
        ):
            validate_layers(**config)


if __name__ == "__main__":
    unittest.main()
