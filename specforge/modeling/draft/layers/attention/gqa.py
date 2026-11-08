"""Node-driven GQA/MHA projections over self or context-then-self KV."""

import torch
from torch import Tensor, nn

from specforge.config.draft._validation import positive_int
from specforge.config.draft.attention import GQASpec, MHASpec
from specforge.config.draft.norm import NormSpec
from specforge.config.draft.rope import RopeSpec

from ..registry import build_component, register_component_builder
from ..rope import apply_rotary
from .backends import eager_attention
from .inputs import validate_attention_inputs
from .masks import AttentionLayout, dense_attention_mask


class GQAAttention(nn.Module):
    def __init__(
        self,
        config: GQASpec,
        *,
        input_size: int,
        hidden_size: int,
        norm: NormSpec,
        rope: RopeSpec,
        backend: str,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if type(config) not in (GQASpec, MHASpec):
            raise TypeError("GQAAttention requires GQASpec or MHASpec")
        positive_int("input_size", input_size)
        positive_int("hidden_size", hidden_size)
        if not isinstance(norm, NormSpec) or not isinstance(rope, RopeSpec):
            raise TypeError("norm and rope must be configuration nodes")
        if config.rope_parameters is not None and rope != config.rope_parameters:
            raise ValueError("rope must match the attention RoPE override")
        rotary_dim = rope.rotary_dim(config.head_dim)
        if backend != "eager":
            raise ValueError("GQAAttention currently supports only the eager backend")
        self.config = config
        self.rotary_dim = rotary_dim
        self.q_norm = (
            build_component(
                "norm", norm, hidden_size=config.head_dim, device=device, dtype=dtype
            )
            if config.qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            build_component(
                "norm", norm, hidden_size=config.head_dim, device=device, dtype=dtype
            )
            if config.qk_norm
            else nn.Identity()
        )
        q_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        options = dict(bias=config.attention_bias, device=device, dtype=dtype)
        self.q_proj = nn.Linear(
            input_size, q_width * (2 if config.output_gate else 1), **options
        )
        self.k_proj = nn.Linear(input_size, kv_width, **options)
        self.v_proj = nn.Linear(input_size, kv_width, **options)
        self.o_proj = nn.Linear(q_width, hidden_size, **options)

    def forward(
        self,
        hidden: Tensor,
        *,
        layout: AttentionLayout,
        position_embeddings: tuple[Tensor, Tensor] | None,
        target_context: Tensor | None = None,
    ) -> Tensor:
        """Use an explicit attention layout and joint context-then-draft RoPE inputs.

        Context is already fused to input_size by the owning architecture. The
        layout's query positions describe hidden; key positions describe context
        followed by hidden. No cache, algorithm settings, or family config enters
        this component. rotary_dim=0 requires position_embeddings=None.
        """
        context_length = validate_attention_inputs(
            hidden,
            layout,
            position_embeddings,
            target_context,
            input_size=self.q_proj.in_features,
            num_heads=self.config.num_attention_heads,
            rotary_dim=self.rotary_dim,
        )
        batch, length = hidden.shape[:2]
        keys = context_length + length
        node = self.config
        mask = dense_attention_mask(
            layout, is_causal=node.is_causal, sliding_window=node.sliding_window
        )
        query = self.q_proj(hidden).view(
            batch,
            length,
            node.num_attention_heads,
            node.head_dim * (2 if node.output_gate else 1),
        )
        gate = None
        if node.output_gate:
            query, gate = query.chunk(2, dim=-1)
            gate = gate.reshape(batch, length, node.num_attention_heads * node.head_dim)
        query = self.q_norm(query).transpose(1, 2)
        key, value = self.k_proj(hidden), self.v_proj(hidden)
        if target_context is not None:
            # Keep separate projections to preserve the existing context path's
            # GEMM shapes and accumulation order, including its input gradients.
            key = torch.cat((self.k_proj(target_context), key), dim=1)
            value = torch.cat((self.v_proj(target_context), value), dim=1)
        key = self.k_norm(
            key.view(batch, keys, node.num_key_value_heads, node.head_dim)
        ).transpose(1, 2)
        value = value.view(
            batch, keys, node.num_key_value_heads, node.head_dim
        ).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query = apply_rotary(
                query, (cos[:, context_length:], sin[:, context_length:])
            )
            key = apply_rotary(key, (cos, sin))
        # FP32 norms/RoPE can promote Q/K under autocast. Preserve their precision
        # through softmax/dropout; the matmuls still follow the autocast policy.
        value = value.to(query.dtype)
        output = eager_attention(
            query,
            key,
            value,
            attention_mask=mask,
            scale=node.head_dim**-0.5,
            dropout_p=node.attention_dropout,
            training=self.training,
        )
        output = output.transpose(1, 2).reshape(
            batch, length, node.num_attention_heads * node.head_dim
        )
        if gate is not None:
            output = output * gate.sigmoid()
        output = self.o_proj(output)
        # Inactive queries must stay zero even when the output projection has bias.
        return output.masked_fill(~mask.any(dim=-1).any(dim=1).unsqueeze(-1), 0)


register_component_builder("attention", GQASpec, GQAAttention)
register_component_builder("attention", MHASpec, GQAAttention)
