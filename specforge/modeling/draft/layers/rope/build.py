"""Build implemented RoPE settings without silently downgrading other rules."""

from specforge.config.draft.rope import (
    DefaultRoPESpec,
    DynamicNTKRoPESpec,
    LinearRoPESpec,
    Llama3RoPESpec,
    MultiAxisRoPESpec,
    NoRoPESpec,
    RopeSpec,
    YaRNSpec,
)

from .base import RotaryEmbedding
from .default import DefaultRotaryEmbedding
from .dynamic import DynamicNTKRotaryEmbedding
from .linear import LinearRotaryEmbedding
from .llama3 import Llama3RotaryEmbedding
from .multiaxis import MultiAxisRotaryEmbedding
from .yarn import YaRNRotaryEmbedding

_BUILDERS = {
    DefaultRoPESpec: DefaultRotaryEmbedding,
    DynamicNTKRoPESpec: DynamicNTKRotaryEmbedding,
    LinearRoPESpec: LinearRotaryEmbedding,
    Llama3RoPESpec: Llama3RotaryEmbedding,
    MultiAxisRoPESpec: MultiAxisRotaryEmbedding,
    YaRNSpec: YaRNRotaryEmbedding,
}


def build_rotary(config: RopeSpec, *, head_dim: int) -> RotaryEmbedding | None:
    """Accept one resolved node; None is the backbone's disabled-RoPE entry."""
    if type(config) is NoRoPESpec:
        config.rotary_dim(head_dim)
        return None
    try:
        builder = _BUILDERS[type(config)]
    except KeyError:
        raise ValueError(f"unsupported rotary node: {type(config).__name__}") from None
    return builder(config, head_dim=head_dim)
