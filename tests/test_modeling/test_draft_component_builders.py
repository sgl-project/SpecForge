"""Modeling dispatch is extensible without linking algorithms or formats."""

import ast
import inspect
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, get_type_hints
from unittest.mock import patch

import torch
from torch import nn

from specforge.config.draft.attention import GQASpec, MHASpec, MLASpec
from specforge.config.draft.conv import GroupedCausalConvSpec
from specforge.config.draft.logits import MuPLogitTransformSpec, NoLogitTransformSpec
from specforge.config.draft.mlp import DenseFFNSpec
from specforge.config.draft.norm import NormSpec, RMSNormSpec
from specforge.config.draft.registry import _SCHEMAS, register_component_schema
from specforge.config.draft.residual import StandardResidualSpec
from specforge.modeling.draft import layers
from specforge.modeling.draft.layers import (
    build_component,
    component_builder,
    register_component_builder,
)
from specforge.modeling.draft.layers.registry import _BUILDERS


@dataclass(frozen=True, kw_only=True)
class TestNormSpec(NormSpec):
    type: ClassVar[str] = "test_builder_norm"


class TestNorm(nn.Module):
    def __init__(
        self, config: TestNormSpec, *, hidden_size: int, device=None, dtype=None
    ):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))

    def forward(self, hidden):
        return hidden * self.weight


class ComponentBuilderTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(_SCHEMAS))
        self.enterContext(patch.dict(_BUILDERS))
        register_component_schema("norm", TestNormSpec)

    def test_builtin_schemas_have_matching_builders(self):
        schemas = {
            key: schema
            for key, schema in _SCHEMAS.items()
            if schema.__module__.startswith("specforge.config.draft.")
        }
        exceptions = {
            ("rope", "none"),  # build_rotary returns None; there is no module.
        }
        self.assertLessEqual(exceptions, schemas.keys())
        self.assertLessEqual(_BUILDERS.keys(), _SCHEMAS.keys())
        self.assertEqual(schemas.keys() - exceptions, _BUILDERS.keys() & schemas.keys())
        for key in schemas.keys() - exceptions:
            with self.subTest(component=key):
                builder = component_builder(*key)
                annotation = get_type_hints(builder.__init__)["config"]
                self.assertTrue(issubclass(schemas[key], annotation))

    def test_builtins_keep_nodes_and_support_composition(self):
        norm_node = RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=1)
        mlp_node = DenseFFNSpec(intermediate_size=24, hidden_act="silu")
        residual_node = StandardResidualSpec()
        norm = build_component("norm", norm_node, hidden_size=16)
        mlp = build_component("mlp", mlp_node, hidden_size=16)
        residual = build_component("residual", residual_node)
        self.assertIs(norm.config, norm_node)
        self.assertIs(mlp.config, mlp_node)
        self.assertIs(residual.config, residual_node)
        hidden = torch.randn(2, 3, 16, requires_grad=True)
        output = norm(residual.reduce(mlp(residual.expand(hidden))))
        output.square().sum().backward()
        for parameter in (*norm.parameters(), *mlp.parameters(), hidden):
            self.assertIsNotNone(parameter.grad)

    def test_standard_residual_connections_are_independent_and_preserve_gradients(self):
        residual = build_component("residual", StandardResidualSpec())
        attention = residual.build_connection(
            hidden_size=4, layer_index=0, sublayer="attention"
        )
        ffn = residual.build_connection(hidden_size=4, layer_index=0, sublayer="mlp")
        self.assertIsNot(attention, ffn)
        self.assertIs(attention.config, residual.config)
        self.assertIs(ffn.config, residual.config)
        self.assertEqual(list(residual.children()), [])
        self.assertEqual(attention.state_dict(), {})
        hidden = torch.randn(2, 3, 4, requires_grad=True)
        sublayer = nn.Linear(4, 4)
        expected = hidden + sublayer(hidden)
        actual = attention(hidden, sublayer)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "input shape"):
            attention(hidden, lambda x: x[:, :1])
        tensors = (hidden, *sublayer.parameters())
        torch.testing.assert_close(
            torch.autograd.grad(actual.sum(), tensors),
            torch.autograd.grad(expected.sum(), tensors),
            rtol=0,
            atol=0,
        )

    def test_external_schema_and_builder_can_register_without_shared_edits(self):
        register_component_builder("norm", TestNormSpec, TestNorm)
        register_component_builder("norm", TestNormSpec, TestNorm)
        node = TestNormSpec()
        module = build_component("norm", node, hidden_size=4)
        self.assertIs(module.config, node)
        self.assertIs(component_builder("norm", node.type), TestNorm)
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_component_builder(
                "norm", TestNormSpec, lambda config: nn.Identity()
            )

    def test_invalid_builder_signatures_fail_at_registration(self):
        for builder in (
            lambda: nn.Identity(),
            lambda config: nn.Identity(),
            lambda config, hidden_size, /, **kwargs: nn.Identity(),
            lambda config, *, required_extra, **kwargs: nn.Identity(),
        ):
            with self.assertRaisesRegex(TypeError, "builder signature"):
                register_component_builder("norm", TestNormSpec, builder)
        self.assertNotIn(("norm", TestNormSpec.type), _BUILDERS)

    def test_unknown_or_unimplemented_kinds_fail_before_allocation(self):
        with patch("torch.nn.Parameter") as allocate:
            with self.assertRaisesRegex(ValueError, "no modeling builder"):
                build_component(
                    "norm",
                    TestNormSpec(),
                    hidden_size=32,
                )
            with self.assertRaises(ValueError):
                component_builder("norm", "missing")
            with self.assertRaises(ValueError):
                component_builder("layer", "decoder")
            allocate.assert_not_called()

    def test_wrong_nodes_schemas_and_builder_results_are_rejected(self):
        with self.assertRaises(TypeError):
            register_component_builder("norm", RMSNormSpec, None)

        class Impostor:
            type = "rms"

        with self.assertRaises(ValueError):
            register_component_builder("norm", Impostor, nn.Identity)
        with self.assertRaises(TypeError):
            build_component("norm", Impostor(), hidden_size=4)
        for builder, error in (
            (lambda config, **kwargs: object(), TypeError),
            (lambda config, **kwargs: nn.Identity(), ValueError),
        ):
            with patch.dict(_BUILDERS):
                register_component_builder("norm", TestNormSpec, builder)
                with self.assertRaises(error):
                    build_component("norm", TestNormSpec())

    def test_builtin_constructor_and_package_boundaries(self):
        for kind, node_type, extras in (
            ("norm", RMSNormSpec, {"hidden_size", "device", "dtype"}),
            ("mlp", DenseFFNSpec, {"hidden_size", "device", "dtype"}),
            ("residual", StandardResidualSpec, set()),
            ("logits", NoLogitTransformSpec, set()),
            ("logits", MuPLogitTransformSpec, set()),
            *(
                (
                    "attention",
                    node,
                    {
                        "input_size",
                        "hidden_size",
                        "norm",
                        "rope",
                        "backend",
                        "training",
                        "device",
                        "dtype",
                    },
                )
                for node in (GQASpec, MHASpec, MLASpec)
            ),
            (
                "conv",
                GroupedCausalConvSpec,
                {"hidden_size", "block_size", "device", "dtype"},
            ),
        ):
            constructor = component_builder(kind, node_type.type)
            parameters = inspect.signature(constructor).parameters
            self.assertEqual(set(parameters), {"config"} | extras)
            self.assertTrue(issubclass(node_type, parameters["config"].annotation))
            for name in extras:
                self.assertEqual(parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        root = Path(layers.__file__).parent
        forbidden = (
            "specforge.algorithms",
            "specforge.training",
            "specforge.checkpoint",
            "specforge.loading",
            "transformers.models",
        )
        for path in root.rglob("*.py"):
            for item in ast.walk(ast.parse(path.read_text())):
                if isinstance(item, ast.ImportFrom):
                    imports = [item.module or ""]
                elif isinstance(item, ast.Import):
                    imports = [alias.name for alias in item.names]
                else:
                    continue
                for name in imports:
                    self.assertFalse(name.startswith(forbidden), (path, name))
                    self.assertNotIn("architecture", name.split("."), (path, name))
                    self.assertNotIn("heads", name.split("."), (path, name))
                    self.assertNotIn("models", name.split("."), (path, name))


if __name__ == "__main__":
    unittest.main()
