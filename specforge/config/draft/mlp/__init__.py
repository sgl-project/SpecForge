"""Feed-forward structure, serialized as mlp_configs."""

from .base import FFNSpec
from .dense import DenseFFNSpec
from .moe import MoEFFNSpec, MoERouterSpec, MoESharedExpertSpec

__all__ = [
    "FFNSpec",
    "DenseFFNSpec",
    "MoEFFNSpec",
    "MoERouterSpec",
    "MoESharedExpertSpec",
]
