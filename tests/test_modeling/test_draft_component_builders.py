"""Modeling dispatch is extensible without linking algorithms or formats."""

import ast
import inspect
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import torch
from torch import nn

from specforge.config.draft.conv import GroupedCausalConvSpec
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
    def __init__(self, config: TestNormSpec, *, hidden_size: int):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden):
        return hidden * self.weight


class ComponentBuilderTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(_SCHEMAS))
        self.enterContext(patch.dict(_BUILDERS))
        register_component_schema("norm", TestNormSpec)

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

    def test_unknown_or_unimplemented_kinds_fail_before_allocation(self):
        with patch("torch.nn.Parameter") as allocate:
            with self.assertRaisesRegex(ValueError, "no modeling builder"):
                build_component(
                    "conv",
                    GroupedCausalConvSpec(kernel_size=3, group_size=8),
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
            (lambda config: object(), TypeError),
            (lambda config: nn.Identity(), ValueError),
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
        ):
            constructor = component_builder(kind, node_type.type)
            parameters = inspect.signature(constructor).parameters
            self.assertEqual(set(parameters), {"config"} | extras)
            self.assertIs(parameters["config"].annotation, node_type)
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
