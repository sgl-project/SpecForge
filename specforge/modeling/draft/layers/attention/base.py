"""Attention contract and the explicit-KV dense implementation scaffold.

Only DenseAttention owns projections/cache dispatch. Sparse implementations can
implement Attention directly without inheriting the dense execution path.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask
from transformers import PretrainedConfig
from transformers.cache_utils import Cache

from ..config import AttentionConfig, LayerKernels

State = TypeVar("State")


@dataclass(frozen=True)
class AttentionContext:
    """Inputs and visibility supplied by the algorithm.

    For dense attention, new keys are ordered [key_value_prefix, hidden_states]. RoPE embeddings
    describe those new keys; queries use the trailing hidden_states positions.
    Omitting the prefix gives ordinary self attention. Cached positions are
    already rotated and are not included in position_embeddings.

    Sparse implementations own their KV representation and map these logical
    input positions/masks to each source. This context does not require a dense
    KV cache, nor does it describe physical slots or compressed indexer keys.
    """

    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    attention_mask: torch.Tensor | BlockMask | None = None
    key_value_prefix: torch.Tensor | None = None
    cache_position: torch.Tensor | None = None
    # Visibility coordinates for all keys (including cached keys) and queries.
    # Defaults align queries to the trailing key positions, not SDPA's top-left.
    query_positions: torch.Tensor | None = None
    key_positions: torch.Tensor | None = None
    # The algorithm built attention_mask from this layer's causal and window
    # rules already (DFlash's per-layer-type masks). Dense backends then use the
    # mask as is instead of rebuilding an identical one in every layer.
    mask_encodes_visibility: bool = False


@dataclass(frozen=True)
class AttentionAuxiliaryLoss:
    """An unweighted scalar objective, with its autograd graph retained.

    The algorithm owns weighting/aggregation; the controller owns backward.
    Return these only when requested by the algorithm's training integration.
    """

    name: str
    value: torch.Tensor

    def __post_init__(self) -> None:
        if not self.name or self.value.ndim != 0:
            raise ValueError("An auxiliary loss needs a name and a scalar tensor")


@dataclass(frozen=True)
class AttentionOutput(Generic[State]):
    hidden_states: torch.Tensor
    state: State | None = None
    attention_weights: torch.Tensor | None = None
    auxiliary_losses: tuple[AttentionAuxiliaryLoss, ...] = ()


class Attention(nn.Module, ABC, Generic[State]):
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        context: AttentionContext,
        state: State | None = None,
        **backend_kwargs,
    ) -> AttentionOutput[State]:
        raise NotImplementedError


class DenseAttention(Attention[Cache]):
    """Shared execution for GQA and expanded MLA, not a sparse attention base."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_idx: int,
        kernels: LayerKernels,
        options: AttentionConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        if config._attn_implementation == "flex_attention":
            if self.attention_dropout != 0.0:
                raise ValueError("FlexAttention requires attention_dropout=0.0")
        self.sliding_window = options.sliding_window
        self.is_causal = options.is_causal
        self._init_projections(config, kernels)

    @abstractmethod
    def _init_projections(
        self, config: PretrainedConfig, kernels: LayerKernels
    ) -> None:
        """Build projections, scaling, num_key_value_groups and o_proj."""
        raise NotImplementedError

    @abstractmethod
    def _compute_qkv(
        self,
        hidden_states: torch.Tensor,
        key_value_prefix: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return rotated Q/K/V in [batch, heads, sequence, channels] layout."""
        raise NotImplementedError

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: AttentionContext,
        state: Cache | None = None,
        *,
        output_attentions: bool = False,
        is_causal: bool | None = None,
        **backend_kwargs,
    ) -> AttentionOutput[Cache]:
        from .backends.dense import dense_attention

        prefix = context.key_value_prefix
        if prefix is None:
            prefix = hidden_states[:, :0]
        q, k, v = self._compute_qkv(hidden_states, prefix, context.position_embeddings)
        if state is not None:
            cos, sin = context.position_embeddings
            k, v = state.update(
                k,
                v,
                self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": context.cache_position},
            )
        output, weights, valid_queries = dense_attention(
            self,
            q,
            k,
            v,
            context.attention_mask,
            query_positions=context.query_positions,
            key_positions=context.key_positions,
            mask_encodes_visibility=context.mask_encodes_visibility,
            is_causal=self.is_causal if is_causal is None else is_causal,
            **backend_kwargs,
        )
        output = self.o_proj(output.reshape(*hidden_states.shape[:2], -1))
        # Zero after o_proj as well, so its bias cannot revive a masked query.
        if valid_queries is not None:
            output = output.masked_fill(~valid_queries.any(dim=1), 0)
        return AttentionOutput(output, state, weights if output_attentions else None)
