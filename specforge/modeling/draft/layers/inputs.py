"""Shared execution inputs, independent of the backbone and architecture models."""

from dataclasses import dataclass

from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from .attention.masks import AttentionLayout


@dataclass(frozen=True, eq=False, kw_only=True)
class LayerInputs:
    """Runtime context, visibility, and positions without algorithm settings.

    New components consume attention_layout and apply their node's constraints.
    Existing adapters may instead use a prepared tensor/BlockMask. Supplying both
    is ambiguous and rejected. RoPE can be supplied directly or prepared by the
    backbone from position_ids. Tensor contents must not be mutated while their
    autograd graph is live; mutable serving KV caches are not part of this API.
    """

    target_context: Tensor | None = None
    attention_mask: Tensor | BlockMask | None = None
    position_ids: Tensor | None = None
    position_embeddings: tuple[Tensor, Tensor] | None = None
    attention_layout: AttentionLayout | None = None

    def __post_init__(self):
        if self.attention_layout is not None:
            if not isinstance(self.attention_layout, AttentionLayout):
                raise TypeError("attention_layout must be AttentionLayout")
            if self.attention_mask is not None:
                raise ValueError("provide attention_layout or attention_mask, not both")
