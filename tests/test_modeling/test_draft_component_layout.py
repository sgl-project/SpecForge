"""Component layouts move elements between fused and per-tensor forms without change."""

import unittest
from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import patch

import torch
from torch import nn

from specforge.config.draft.mlp import FFNSpec
from specforge.config.draft.registry import _SCHEMAS, register_component_schema
from specforge.modeling.draft.layers import (
    ComponentLayout,
    StackedParameter,
    component_layout,
    register_component_builder,
    register_component_layout,
)
from specforge.modeling.draft.layers.layout import join, split
from specforge.modeling.draft.layers.registry import _BUILDERS, _LAYOUTS
from tests.test_config.test_draft_moe_spec import moe_spec

EXPERTS = ComponentLayout(
    fused={
        "experts.gate_up_proj": StackedParameter(
            parts=("experts.{e}.gate_proj.weight", "experts.{e}.up_proj.weight"),
            dim=0,
        ),
        "experts.down_proj": StackedParameter(
            parts=("experts.{e}.down_proj.weight",), dim=0
        ),
    },
    count="num_experts",
)


def expert_tensors(num_experts, hidden, width, dtype=torch.float32):
    parts = {}
    for e in range(num_experts):
        parts[f"experts.{e}.gate_proj.weight"] = torch.randn(width, hidden, dtype=dtype)
        parts[f"experts.{e}.up_proj.weight"] = torch.randn(width, hidden, dtype=dtype)
        parts[f"experts.{e}.down_proj.weight"] = torch.randn(hidden, width, dtype=dtype)
    return parts


@dataclass(frozen=True, kw_only=True)
class TestFFNSpec(FFNSpec):
    type: ClassVar[str] = "test_layout_ffn"
    num_experts: int


class TestFFN(nn.Module):
    def __init__(self, config: TestFFNSpec, *, hidden_size: int):
        super().__init__()
        self.config = config


class ComponentLayoutTest(unittest.TestCase):
    def test_join_and_split_are_exact_inverses(self):
        config = moe_spec(num_experts=3, intermediate_size=5)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                parts = expert_tensors(3, 8, 5, dtype)
                fused = join(EXPERTS, config, parts)
                self.assertEqual(set(fused), set(EXPERTS.fused))
                self.assertEqual(tuple(fused["experts.gate_up_proj"].shape), (3, 10, 8))
                self.assertEqual(tuple(fused["experts.down_proj"].shape), (3, 8, 5))
                self.assertTrue(all(t.dtype == dtype for t in fused.values()))
                restored = split(EXPERTS, config, fused)
                self.assertEqual(set(restored), set(parts))
                for name, tensor in parts.items():
                    self.assertTrue(torch.equal(restored[name], tensor), name)
                rejoined = join(EXPERTS, config, restored)
                for name, tensor in fused.items():
                    self.assertTrue(torch.equal(rejoined[name], tensor), name)
        self.assertEqual(
            EXPERTS.part_names(moe_spec(num_experts=2, intermediate_size=5))[
                "experts.down_proj"
            ],
            (("experts.0.down_proj.weight",), ("experts.1.down_proj.weight",)),
        )

    def test_incomplete_or_mismatched_tensors_are_rejected(self):
        config = moe_spec(num_experts=2, intermediate_size=5)
        parts = expert_tensors(2, 8, 5)
        with self.assertRaisesRegex(ValueError, "missing checkpoint tensors"):
            join(
                EXPERTS, config, {k: v for k, v in parts.items() if "up_proj" not in k}
            )
        with self.assertRaisesRegex(ValueError, "unused checkpoint tensors"):
            join(
                EXPERTS,
                config,
                parts
                | {"experts.2.gate_proj.weight": parts["experts.0.gate_proj.weight"]},
            )
        fused = join(EXPERTS, config, parts)
        with self.assertRaisesRegex(ValueError, "expected 2 stacked entries"):
            split(
                EXPERTS,
                config,
                fused | {"experts.down_proj": fused["experts.down_proj"][:1]},
            )
        with self.assertRaisesRegex(ValueError, "missing fused parameter"):
            split(EXPERTS, config, {"experts.down_proj": fused["experts.down_proj"]})
        with self.assertRaisesRegex(ValueError, "unknown fused parameters"):
            split(
                EXPERTS, config, fused | {"experts.extra": fused["experts.down_proj"]}
            )
        odd = fused | {"experts.gate_up_proj": fused["experts.gate_up_proj"][:, :9]}
        with self.assertRaisesRegex(ValueError, "equal parts"):
            split(EXPERTS, config, odd)
        with self.assertRaises(ValueError):
            StackedParameter(parts=("experts.gate_proj.weight",), dim=0)
        with self.assertRaises(ValueError):
            ComponentLayout(fused={}, count="num_experts")

    def test_layouts_register_only_against_built_components(self):
        with patch.dict(_SCHEMAS), patch.dict(_BUILDERS), patch.dict(_LAYOUTS):
            register_component_schema("mlp", TestFFNSpec)
            with self.assertRaisesRegex(ValueError, "no modeling builder"):
                register_component_layout("mlp", TestFFNSpec, EXPERTS)
            register_component_builder("mlp", TestFFNSpec, TestFFN)
            with self.assertRaisesRegex(ValueError, "not a node field"):
                register_component_layout(
                    "mlp", TestFFNSpec, ComponentLayout(EXPERTS.fused, "num_entries")
                )
            with self.assertRaises(TypeError):
                register_component_layout("mlp", TestFFNSpec, {"experts": "stack"})
            register_component_layout("mlp", TestFFNSpec, EXPERTS)
            register_component_layout("mlp", TestFFNSpec, EXPERTS)  # same value: fine
            self.assertIs(component_layout("mlp", "test_layout_ffn"), EXPERTS)
            other = ComponentLayout(
                {"experts.down_proj": EXPERTS.fused["experts.down_proj"]}, "num_experts"
            )
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_component_layout("mlp", TestFFNSpec, other)
        self.assertIsNone(component_layout("mlp", "dense"))
        with self.assertRaisesRegex(ValueError, "unknown mlp type"):
            component_layout("mlp", "test_layout_ffn")


if __name__ == "__main__":
    unittest.main()
