"""Shared execution inputs, independent of the backbone and architecture models."""

from dataclasses import dataclass

from torch import Tensor
from torch.nn.attention.flex_attention import BlockMask

from .attention.backends import PreparedAttention
from .attention.masks import AttentionLayout


@dataclass(frozen=True, eq=False, kw_only=True)
class LayerInputs:
    """Runtime context, visibility, and positions without algorithm settings.

    Decoder components consume attention prepared before the layer loop.
    Existing adapters may instead use a layout or tensor/BlockMask.
    Supply only one attention representation. RoPE can be supplied directly or
    prepared by the backbone from position_ids. Tensor contents must not be mutated while their
    autograd graph is live; mutable serving KV caches are not part of this API.
    """

    attention: PreparedAttention | None = None
    target_context: Tensor | None = None
    attention_mask: Tensor | BlockMask | None = None
    position_ids: Tensor | None = None
    position_embeddings: tuple[Tensor, Tensor] | None = None
    attention_layout: AttentionLayout | None = None

    def __post_init__(self):
        if self.attention is not None:
            if not isinstance(self.attention, PreparedAttention):
                raise TypeError("attention must be PreparedAttention")
            if self.attention_layout is not None or self.attention_mask is not None:
                raise ValueError(
                    "provide prepared attention or a legacy layout/mask, not both"
                )
        if self.attention_layout is not None:
            if not isinstance(self.attention_layout, AttentionLayout):
                raise TypeError("attention_layout must be AttentionLayout")
            if self.attention_mask is not None:
                raise ValueError("provide attention_layout or attention_mask, not both")
