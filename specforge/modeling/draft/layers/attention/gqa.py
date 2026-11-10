"""Node-driven GQA/MHA projections over self or context-then-self KV."""

import torch
from torch import Tensor, nn

from specforge.config.draft._validation import boolean, positive_int
from specforge.config.draft.attention import GQASpec, MHASpec
from specforge.config.draft.norm import NormSpec
from specforge.config.draft.rope import RopeSpec

from ..registry import build_component, register_component_builder
from ..rope import apply_rotary
from .backends import (
    AttentionBackend,
    AttentionPreparationScope,
    PreparedAttention,
    resolve_attention_backend,
)
from .masks import AttentionLayout
from .validation import validate_hidden, validate_rotary


class GQAAttention(nn.Module):
    def __init__(
        self,
        config: GQASpec,
        *,
        input_size: int,
        hidden_size: int,
        norm: NormSpec,
        rope: RopeSpec,
        backend: str | AttentionBackend,
        training: bool = True,
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
        if not isinstance(rope, RopeSpec):
            raise TypeError("rope must be the resolved RopeSpec")
        if config.rope_parameters is not None and rope != config.rope_parameters:
            raise ValueError("rope must match the attention RoPE override")
        rotary_dim = rope.rotary_dim(config.head_dim)
        self.backend = resolve_attention_backend(backend)
        self.config = config
        self.train(training)  # Validate the backend before allocating parameters.
        self.rotary_dim = rotary_dim
        self.rope = rope
        norm = config.qk_norm_config or norm
        self.q_norm = (
            build_component(
                "norm",
                norm,
                hidden_size=(
                    config.head_dim
                    if config.qk_norm_scope == "head"
                    else config.num_attention_heads * config.head_dim
                ),
                device=device,
                dtype=dtype,
            )
            if config.qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            build_component(
                "norm",
                norm,
                hidden_size=(
                    config.head_dim
                    if config.qk_norm_scope == "head"
                    else config.num_key_value_heads * config.head_dim
                ),
                device=device,
                dtype=dtype,
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
        self.train(training)

    def train(self, mode: bool = True):
        boolean("training", mode)
        self.backend.validate(
            head_dim=self.config.head_dim,
            value_head_dim=self.config.head_dim,
            dropout_p=self.config.attention_dropout if mode else 0.0,
        )
        return super().train(mode)

    def _normalize(self, hidden: Tensor, norm: nn.Module) -> Tensor:
        if self.config.qk_norm_scope == "projection":
            shape = hidden.shape
            return (
                norm(hidden.transpose(1, 2).flatten(2))
                .view(shape[0], shape[2], shape[1], shape[3])
                .transpose(1, 2)
            )
        return norm(hidden.transpose(1, 2)).transpose(1, 2)

    def prepare(
        self,
        layout: AttentionLayout,
        preparation: AttentionPreparationScope,
        *,
        batch_size: int,
    ) -> PreparedAttention:
        """Prepare outside the layer loop; equivalent layers share the same result."""
        return preparation.prepare(
            self.backend,
            layout,
            batch_size=batch_size,
            num_heads=self.config.num_attention_heads,
            is_causal=self.config.is_causal,
            sliding_window=self.config.sliding_window,
        )

    def forward(
        self,
        hidden: Tensor,
        *,
        attention: PreparedAttention,
        position_embeddings: tuple[Tensor, Tensor] | None,
        target_context: Tensor | None = None,
    ) -> Tensor:
        """Use an explicit attention layout and joint context-then-draft RoPE inputs.

        Context is already fused to input_size by the owning architecture. The
        layout's query positions describe hidden; key positions describe context
        followed by hidden. No cache, algorithm settings, or family config enters
        this component. rotary_dim=0 requires position_embeddings=None. Supplied
        cosine/sine tables must be FP32; rotation casts back to the projected Q/K dtype.
        """
        batch, length, context_length = validate_hidden(
            hidden, target_context, self.q_proj.in_features
        )
        keys = context_length + length
        if not isinstance(attention, PreparedAttention):
            raise TypeError("GQAAttention requires PreparedAttention")
        layout = attention.layout
        if (
            attention.batch_size != batch
            or attention.num_heads != self.config.num_attention_heads
            or attention.backend is not self.backend
            or attention.is_causal != self.config.is_causal
            or attention.sliding_window != self.config.sliding_window
            or layout.query_positions.shape[1] != length
            or layout.key_positions.shape[1] != keys
            or layout.query_positions.device != hidden.device
        ):
            raise ValueError(
                "prepared attention must match the layer and query/KV tensors"
            )
        validate_rotary(position_embeddings, hidden, keys, self.rotary_dim)
        node = self.config
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
        query = query.transpose(1, 2)
        key, value = self.k_proj(hidden), self.v_proj(hidden)
        if target_context is not None:
            # Keep separate projections to preserve the existing context path's
            # GEMM shapes and accumulation order, including its input gradients.
            key = torch.cat((self.k_proj(target_context), key), dim=1)
            value = torch.cat((self.v_proj(target_context), value), dim=1)
        key = key.view(batch, keys, node.num_key_value_heads, node.head_dim).transpose(
            1, 2
        )
        value = value.view(
            batch, keys, node.num_key_value_heads, node.head_dim
        ).transpose(1, 2)
        if node.qk_norm_position == "before_rope":
            query, key = self._normalize(query, self.q_norm), self._normalize(
                key, self.k_norm
            )
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query = apply_rotary(
                query,
                (cos[:, context_length:], sin[:, context_length:]),
                interleaved=self.rope.interleaved,
            )
            key = apply_rotary(key, (cos, sin), interleaved=self.rope.interleaved)
        if node.qk_norm_position == "after_rope":
            query, key = self._normalize(query, self.q_norm), self._normalize(
                key, self.k_norm
            )
        result = attention.attend(
            query,
            key,
            value,
            scale=node.head_dim**-0.5,
            dropout_p=node.attention_dropout,
            training=self.training,
        )
        output = result.output
        output = output.transpose(1, 2).reshape(
            batch, length, node.num_attention_heads * node.head_dim
        )
        if gate is not None:
            output = output * gate.sigmoid()
        output = self.o_proj(output)
        # Inactive queries must stay zero even when the output projection has bias.
        return result.mask_output(output)


register_component_builder("attention", GQASpec, GQAAttention)
register_component_builder("attention", MHASpec, GQAAttention)
