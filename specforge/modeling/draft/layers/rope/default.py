"""Position-selected rotary embeddings, without family configuration or caches."""

from specforge.config.draft.rope import DefaultRoPESpec

from ..registry import register_component_builder
from .base import RotaryEmbedding


class DefaultRotaryEmbedding(RotaryEmbedding):
    """Unscaled full or partial rotation."""

    def __init__(self, config: DefaultRoPESpec, *, head_dim: int):
        if type(config) is not DefaultRoPESpec:
            raise TypeError("DefaultRotaryEmbedding requires DefaultRoPESpec")
        super().__init__(config, head_dim=head_dim)


register_component_builder("rope", DefaultRoPESpec, DefaultRotaryEmbedding)
