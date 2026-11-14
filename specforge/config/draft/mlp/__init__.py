"""Feed-forward structure, serialized as mlp_configs."""

from .base import FFNSpec
from .dense import DenseFFNSpec
from .moe import MoEExpertSpec, MoEFFNSpec, MoERouterSpec

__all__ = [
    "FFNSpec",
    "DenseFFNSpec",
    "MoEExpertSpec",
    "MoEFFNSpec",
    "MoERouterSpec",
]
