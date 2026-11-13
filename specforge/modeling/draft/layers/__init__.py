"""Node-configured building blocks shared by draft architectures."""

from .attention import GQAAttention, MLAAttention
from .conv import GroupedCausalConv
from .decoder import DecoderLayer
from .logits import MuPLogitTransform, NoLogitTransform, TargetLogitTransform
from .mlp import DenseFFN, MoEFFN
from .norm import RMSNorm
from .registry import build_component, component_builder, register_component_builder
from .residual import StandardResidual

__all__ = [
    "DenseFFN",
    "DecoderLayer",
    "GQAAttention",
    "GroupedCausalConv",
    "MLAAttention",
    "MoEFFN",
    "MuPLogitTransform",
    "NoLogitTransform",
    "RMSNorm",
    "StandardResidual",
    "TargetLogitTransform",
    "build_component",
    "component_builder",
    "register_component_builder",
]
