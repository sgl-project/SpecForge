"""Node-configured building blocks shared by draft architectures."""

from .attention import GQAAttention, MLAAttention
from .conv import GroupedCausalConv
from .decoder import DecoderLayer
from .layout import ComponentLayout, StackedParameter
from .logits import MuPLogitTransform, NoLogitTransform, TargetLogitTransform
from .mlp import DenseFFN, MoEFFN
from .norm import RMSNorm
from .registry import (
    build_component,
    component_builder,
    component_layout,
    register_component_builder,
    register_component_layout,
)
from .residual import StandardResidual

__all__ = [
    "ComponentLayout",
    "DenseFFN",
    "DecoderLayer",
    "GQAAttention",
    "GroupedCausalConv",
    "MLAAttention",
    "MoEFFN",
    "MuPLogitTransform",
    "NoLogitTransform",
    "RMSNorm",
    "StackedParameter",
    "StandardResidual",
    "TargetLogitTransform",
    "build_component",
    "component_builder",
    "component_layout",
    "register_component_builder",
    "register_component_layout",
]
