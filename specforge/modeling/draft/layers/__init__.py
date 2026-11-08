"""Node-configured building blocks shared by draft architectures."""

from .attention import GQAAttention
from .decoder import DecoderLayer
from .mlp import DenseFFN
from .norm import RMSNorm
from .registry import build_component, component_builder, register_component_builder
from .residual import StandardResidual

__all__ = [
    "DenseFFN",
    "DecoderLayer",
    "GQAAttention",
    "RMSNorm",
    "StandardResidual",
    "build_component",
    "component_builder",
    "register_component_builder",
]
