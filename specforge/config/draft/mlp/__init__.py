"""Feed-forward structure, serialized as mlp_configs."""

from .base import FFNSpec
from .dense import DenseFFNSpec

__all__ = ["FFNSpec", "DenseFFNSpec"]
