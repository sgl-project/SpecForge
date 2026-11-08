"""Node-driven GQA/MHA projections over self or context-then-self KV."""

import torch
from torch import Tensor, nn

from specforge.config.draft._validation import positive_int
from specforge.config.draft.attention import GQASpec, MHASpec
from specforge.config.draft.norm import NormSpec

from ..registry import build_component, register_component_builder
from ..rope import apply_rotary
from .backends import eager_attention
from .masks import AttentionLayout, dense_attention_mask


class GQAAttention(nn.Module):
    def __init__(
        self,
        config: GQASpec,
        *,
        input_size: int,
        hidden_size: int,
        norm: NormSpec,
        rotary_dim: int,
        backend: str,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if type(config) not in (GQASpec, MHASpec):
            raise TypeError("GQAAttention requires GQASpec or MHASpec")
        positive_int("input_size", input_size)
        positive_int("hidden_size", hidden_size)
        if not isinstance(norm, NormSpec):
            raise TypeError("norm must be a normalization node")
        if (
            type(rotary_dim) is not int
            or rotary_dim < 0
            or rotary_dim % 2
            or rotary_dim > config.head_dim
        ):
            raise ValueError(
                "rotary_dim must be zero or an even dimension within the head"
            )
        if (
            config.rope_parameters is not None
            and rotary_dim != config.rope_parameters.rotary_dim(config.head_dim)
        ):
            raise ValueError("rotary_dim must match the attention RoPE override")
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
        if hidden.ndim != 3 or hidden.shape[-1] != self.q_proj.in_features:
            raise ValueError("hidden must have shape [batch, query_length, input_size]")
        batch, length = hidden.shape[:2]
        if target_context is not None and (
            target_context.ndim != 3
            or target_context.shape[0] != batch
            or target_context.shape[-1] != hidden.shape[-1]
            or target_context.device != hidden.device
            or not target_context.is_floating_point()
            or (
                target_context.dtype != hidden.dtype
                and not (
                    torch.is_autocast_enabled(hidden.device.type)
                    and hidden.dtype in (torch.float32, torch.float16, torch.bfloat16)
                    and target_context.dtype
                    in (torch.float32, torch.float16, torch.bfloat16)
                )
            )
        ):
            raise ValueError(
                "target_context must match hidden batch, input width, and device; "
                "different floating dtypes require autocast"
            )
        context_length = 0 if target_context is None else target_context.shape[1]
        keys = context_length + length
        if not isinstance(layout, AttentionLayout):
            raise TypeError("GQAAttention requires AttentionLayout")
        if (
            layout.query_positions.shape[0] not in (1, batch)
            or layout.query_positions.shape[1] != length
            or layout.key_positions.shape[1] != keys
            or layout.query_positions.device != hidden.device
            or layout.visibility.shape[1] not in (1, self.config.num_attention_heads)
        ):
            raise ValueError(
                "layout must describe the query and context-then-query KV tensors"
            )
        if self.rotary_dim:
            if position_embeddings is None or len(position_embeddings) != 2:
                raise ValueError("rotary attention requires a joint cosine/sine pair")
            for value in position_embeddings:
                if (
                    value.ndim != 3
                    or value.shape[0] not in (1, batch)
                    or value.shape[1:] != (keys, self.rotary_dim)
                    or value.device != hidden.device
                    or not value.is_floating_point()
                ):
                    raise ValueError(
                        "RoPE tensors must describe all context-then-query positions"
                    )
        elif position_embeddings is not None:
            raise ValueError("disabled RoPE requires position_embeddings=None")
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
