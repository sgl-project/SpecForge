"""Grouped-query projections, reusable by dense and sparse attention."""

import torch
from torch import nn
from transformers import PretrainedConfig

from ..config import LayerKernels
from .base import DenseAttention
from .rotary import apply_rotary_pos_emb


class GQAProjections:
    """GQA/MHA projections for self attention with an optional KV prefix."""

    def _init_projections(
        self, config: PretrainedConfig, kernels: LayerKernels
    ) -> None:
        if (
            config.num_key_value_heads <= 0
            or config.num_attention_heads % config.num_key_value_heads
        ):
            raise ValueError(
                "num_key_value_heads must be positive and divide num_attention_heads"
            )
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = kernels.make_rms_norm(self.head_dim, config.rms_norm_eps)
        self.k_norm = kernels.make_rms_norm(self.head_dim, config.rms_norm_eps)

    def _compute_qkv(
        self,
        hidden_states: torch.Tensor,
        key_value_prefix: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = key_value_prefix.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(key_value_prefix)
        k_self = self.k_proj(hidden_states)
        v_ctx = self.v_proj(key_value_prefix)
        v_self = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_self], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_self], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v


class GQAAttention(GQAProjections, DenseAttention):
    """Grouped-query attention with a dense execution backend."""
