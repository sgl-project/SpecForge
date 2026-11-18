"""Build registered RoPE implementations from resolved configuration nodes."""

from torch import nn

from specforge.config.draft.rope import NoRoPESpec, RopeSpec

from ..registry import build_component


def build_rotary(config: RopeSpec, *, head_dim: int) -> nn.Module | None:
    """Accept one resolved node; None is the backbone's disabled-RoPE entry."""
    if not isinstance(config, RopeSpec):
        raise ValueError(f"unsupported rotary node: {type(config).__name__}")
    if type(config) is NoRoPESpec:
        config.rotary_dim(head_dim)
        return None
    return build_component("rope", config, head_dim=head_dim)
