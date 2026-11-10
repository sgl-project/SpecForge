"""Rotary position components driven by resolved RoPE nodes."""

from .apply import apply_rotary
from .build import build_rotary
from .default import DefaultRotaryEmbedding
from .dynamic import DynamicNTKRotaryEmbedding
from .linear import LinearRotaryEmbedding
from .llama3 import Llama3RotaryEmbedding
from .multiaxis import MultiAxisRotaryEmbedding
from .yarn import YaRNRotaryEmbedding

# Exact types whose tables depend only on (node, head_dim, positions); a
# subclass may add state, so it is not listed by inheritance.
NODE_DETERMINED_ROTARY_TYPES = frozenset(
    {
        DefaultRotaryEmbedding,
        DynamicNTKRotaryEmbedding,
        LinearRotaryEmbedding,
        Llama3RotaryEmbedding,
        MultiAxisRotaryEmbedding,
        YaRNRotaryEmbedding,
    }
)

__all__ = [
    "NODE_DETERMINED_ROTARY_TYPES",
    "DefaultRotaryEmbedding",
    "DynamicNTKRotaryEmbedding",
    "LinearRotaryEmbedding",
    "Llama3RotaryEmbedding",
    "MultiAxisRotaryEmbedding",
    "YaRNRotaryEmbedding",
    "apply_rotary",
    "build_rotary",
]
