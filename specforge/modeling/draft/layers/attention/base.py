"""Attention contract and the explicit-KV dense implementation scaffold."""

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

    New keys are ordered [key_value_prefix, hidden_states]. RoPE embeddings
    describe those new keys; queries use the trailing hidden_states positions.
    Omitting the prefix gives ordinary self attention. Cached positions are
    already rotated and are not included in position_embeddings.
    """

    position_embeddings: tuple[torch.Tensor, torch.Tensor]
    attention_mask: torch.Tensor | BlockMask | None = None
    key_value_prefix: torch.Tensor | None = None
    cache_position: torch.Tensor | None = None


@dataclass(frozen=True)
class AttentionOutput(Generic[State]):
    hidden_states: torch.Tensor
    state: State | None = None
    attention_weights: torch.Tensor | None = None


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
    """Shared execution for GQA and expanded MLA.

    Subclasses own only the projection parameterization: ``_init_projections``
    builds the weights and must define ``scaling``, ``num_key_value_groups``
    and ``o_proj``; ``_compute_qkv`` returns rotated ``(q, k, v)`` with keys
    ordered prefix-then-self. KV-cache updates, backend dispatch, fully masked
    query zeroing and the output projection live here.
    """

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
        # Read by mask-free backends (flash_attention_2) and SDPA without a mask.
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
            self, q, k, v, context.attention_mask, **backend_kwargs
        )
        output = self.o_proj(output.reshape(*hidden_states.shape[:2], -1))
        if valid_queries is not None:
            output = output.masked_fill(~valid_queries.any(dim=1), 0)
        return AttentionOutput(output, state, weights if output_attentions else None)
