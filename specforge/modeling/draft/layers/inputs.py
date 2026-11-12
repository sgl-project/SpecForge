"""Shared execution inputs, independent of the backbone and architecture models."""

from dataclasses import dataclass

from torch import Tensor

from .attention.backends import PreparedAttention
from .attention.masks import AttentionLayout


@dataclass(frozen=True, eq=False, kw_only=True)
class LayerInputs:
    """Runtime context, visibility, and positions without algorithm settings.

    Decoder components consume attention prepared before the layer loop.
    Existing adapters may instead use a layout.
    Supply only one attention representation. RoPE can be supplied directly or
    prepared by the backbone from position_ids. Tensor contents must not be mutated while their
    autograd graph is live; mutable serving KV caches are not part of this API.
    """

    attention: PreparedAttention | None = None
    target_context: Tensor | None = None
    position_ids: Tensor | None = None
    position_embeddings: tuple[Tensor, Tensor] | None = None
    attention_layout: AttentionLayout | None = None

    def __post_init__(self):
        if self.attention is not None:
            if not isinstance(self.attention, PreparedAttention):
                raise TypeError("attention must be PreparedAttention")
            if self.attention_layout is not None:
                raise ValueError("provide prepared attention or a layout, not both")
        if self.attention_layout is not None:
            if not isinstance(self.attention_layout, AttentionLayout):
                raise TypeError("attention_layout must be AttentionLayout")


@dataclass(frozen=True, eq=False)
class LayerOutput:
    """A layer's hidden result; specialized stages may add typed tensor fields."""

    hidden: Tensor


@dataclass(frozen=True, eq=False)
class ExpertChoice:
    """One MoE layer's expert choice, carried out of the forward for balancing."""

    expert_ids: Tensor  # [tokens, top_k] chosen experts


@dataclass(frozen=True, eq=False)
class MoEOutput:
    """A MoE FFN's stream update together with its expert choice."""

    update: Tensor
    expert_choice: ExpertChoice


@dataclass(frozen=True, eq=False)
class MoELayerOutput(LayerOutput):
    """A decoder layer whose FFN routed: the hidden result plus its expert choice."""

    expert_choice: ExpertChoice
