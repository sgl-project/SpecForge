"""Structural nodes reject ambiguity without loading model implementations."""

import dataclasses
import subprocess
import sys
import unittest

from specforge.config.draft.attention import GQASpec, MHASpec
from specforge.config.draft.conv import GroupedCausalConvSpec
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.mlp import (
    DenseFFNSpec,
    MoEExpertSpec,
    MoEFFNSpec,
    MoERouterSpec,
)
from specforge.config.draft.norm import RMSNormSpec
from specforge.config.draft.residual import StandardResidualSpec


def gqa(**overrides):
    return GQASpec(
        **{
            "qk_norm_position": "before_rope",
            "qk_norm_scope": "head",
            "qk_norm_config": None,
            "rope_parameters": None,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "qk_norm": True,
            "attention_bias": False,
            "output_gate": False,
            "is_causal": False,
            "sliding_window": None,
            "attention_dropout": 0.0,
            **overrides,
        }
    )


def moe(**overrides):
    values = {
        "num_experts": 8,
        "num_experts_per_tok": 1,
        "experts": MoEExpertSpec(intermediate_size=96, hidden_act="silu", gated=True),
        "shared_expert": None,
        "shared_expert_gate": False,
        "router": MoERouterSpec(
            scoring="softmax", normalize_top_k=False, route_scale=1.0, expert_bias=True
        ),
    }
    return MoEFFNSpec(**{**values, **overrides})


class DraftNodesTest(unittest.TestCase):
    def test_nodes_are_immutable_hashable_values(self):
        nodes = (
            gqa(),
            moe(),
            DenseFFNSpec(intermediate_size=256, hidden_act="silu"),
            RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=0),
            StandardResidualSpec(),
            GroupedCausalConvSpec(kernel_size=3, group_size=16),
            LayerSpec(attention="full", mlp="dense"),
        )
        for node in nodes:
            with self.subTest(node=node):
                clone = dataclasses.replace(node)
                self.assertEqual(node, clone)
                self.assertEqual(hash(node), hash(clone))
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    node.type = "target"

    def test_scalar_types_are_not_coerced(self):
        for name in ("num_attention_heads", "num_key_value_heads", "head_dim"):
            for value in (True, 0, -1, 2.0, "2", None):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        gqa(**{name: value})
        for name in ("qk_norm", "attention_bias", "output_gate", "is_causal"):
            for value in (None, 0, 1, "false"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        gqa(**{name: value})

    def test_attention_geometry(self):
        self.assertEqual(gqa().num_key_value_groups, 4)
        with self.assertRaisesRegex(ValueError, "divisible"):
            gqa(num_key_value_heads=3)
        values = dataclasses.asdict(gqa())
        with self.assertRaisesRegex(ValueError, "equal"):
            MHASpec(**values)
        values["num_key_value_heads"] = 8
        self.assertEqual(MHASpec(**values).num_key_value_groups, 1)

    def test_window_and_causality_are_independent(self):
        for causal in (True, False):
            for window in (None, 128):
                node = gqa(is_causal=causal, sliding_window=window)
                self.assertIs(node.is_causal, causal)
                self.assertEqual(node.sliding_window, window)
        for value in (False, 0, -1, 2.0, "128"):
            with self.assertRaisesRegex(ValueError, "sliding_window"):
                gqa(sliding_window=value)

    def test_missing_and_unknown_fields_are_rejected(self):
        values = dataclasses.asdict(gqa())
        for name in values:
            with self.subTest(name=name), self.assertRaisesRegex(TypeError, name):
                GQASpec(**{key: value for key, value in values.items() if key != name})
        for factory, values in (
            (GQASpec, values),
            (DenseFFNSpec, {"intermediate_size": 64, "hidden_act": "silu"}),
            (RMSNormSpec, {"eps": 1e-6, "weight_offset": 0}),
            (StandardResidualSpec, {}),
            (GroupedCausalConvSpec, {"kernel_size": 3, "group_size": 8}),
            (LayerSpec, {"attention": "full", "mlp": "dense"}),
        ):
            with (
                self.subTest(factory=factory),
                self.assertRaisesRegex(TypeError, "typo"),
            ):
                factory(**values, typo=True)

    def test_numeric_semantics_are_explicit(self):
        for value in (True, "0.1", float("nan"), float("inf"), -0.1, 1):
            with self.assertRaisesRegex(ValueError, "attention_dropout"):
                gqa(attention_dropout=value)
        for value in (True, 0, -1, "1e-6", float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "eps"):
                RMSNormSpec(has_weight=True, eps=value, weight_offset=0)
        for value in (True, -1, 2, 0.0, "0", None):
            with self.assertRaisesRegex(ValueError, "weight_offset"):
                RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=value)
        ordinary = RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=0)
        shifted = RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=1)
        self.assertNotEqual(ordinary, shifted)

    def test_moe_ffn_node(self):
        expert = dict(intermediate_size=48, hidden_act="silu", gated=True)
        router = dict(
            scoring="softmax", normalize_top_k=False, route_scale=1.0, expert_bias=True
        )
        for build, message in (
            (lambda: moe(num_experts_per_tok=9), "exceeds num_experts"),
            (
                lambda: moe(experts=MoEExpertSpec(**{**expert, "gated": False})),
                "gated FFNs only",
            ),
            (lambda: moe(num_experts=0), "num_experts must be a positive integer"),
            (lambda: moe(num_experts_per_tok=True), "num_experts_per_tok"),
            (
                lambda: MoEExpertSpec(**{**expert, "intermediate_size": -1}),
                "intermediate_size",
            ),
            (
                lambda: MoEExpertSpec(**{**expert, "hidden_act": "gelu_new"}),
                "hidden_act",
            ),
            (lambda: MoEExpertSpec(**{**expert, "gated": 1}), "gated"),
            (lambda: moe(shared_expert_gate=True), "requires a shared expert"),
            (lambda: MoERouterSpec(**{**router, "scoring": "relu"}), "scoring"),
            (lambda: MoERouterSpec(**{**router, "route_scale": 0}), "route_scale"),
            (
                lambda: MoERouterSpec(**{**router, "route_scale": float("inf")}),
                "route_scale",
            ),
            (
                lambda: MoERouterSpec(**{**router, "normalize_top_k": 1}),
                "normalize_top_k",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build()
        for build, message in (
            (lambda: moe(router=None), "router"),
            (lambda: moe(shared_expert=expert), "shared_expert"),
            (lambda: moe(experts=expert), "experts"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(TypeError, message):
                    build()

    def test_ffn_convolution_and_references(self):
        for value in (True, 0, -1, 2.0, "2", None):
            with self.assertRaisesRegex(ValueError, "intermediate_size"):
                DenseFFNSpec(intermediate_size=value, hidden_act="silu")
            for name in ("kernel_size", "group_size"):
                with self.assertRaisesRegex(ValueError, name):
                    GroupedCausalConvSpec(
                        **{"kernel_size": 3, "group_size": 8, name: value}
                    )
        for value in (None, 3, "", " "):
            with self.assertRaisesRegex(ValueError, "hidden_act"):
                DenseFFNSpec(intermediate_size=64, hidden_act=value)
            for name in ("attention", "mlp"):
                with self.assertRaisesRegex(ValueError, name):
                    LayerSpec(**{"attention": "full", "mlp": "dense", name: value})
        for value in (3, "", " "):
            with self.assertRaisesRegex(ValueError, "sublayer_conv"):
                LayerSpec(attention="full", mlp="dense", sublayer_conv=value)
        layer = LayerSpec(attention="full", mlp="dense")
        self.assertFalse(hasattr(layer, "type"))
        with self.assertRaisesRegex(TypeError, "type"):
            LayerSpec(attention="full", mlp="dense", type="decoder")

    def test_no_draft_module_loads_run_schema_or_compute_packages(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import importlib
import pkgutil
import sys
import specforge.config.draft as draft
for module in pkgutil.walk_packages(draft.__path__, draft.__name__ + '.'):
    importlib.import_module(module.name)
for name in ('torch', 'transformers', 'pydantic', 'specforge.runtime',
             'specforge.config.schema', 'specforge.modeling',
             'specforge.algorithms'):
    assert name not in sys.modules, name
""",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_existing_run_config_exports_still_work(self):
        import specforge.config as config
        from specforge.config import schema

        for name in config.__all__:
            self.assertIs(getattr(config, name), getattr(schema, name))
        with self.assertRaises(AttributeError):
            config.not_a_config_export


if __name__ == "__main__":
    unittest.main()
