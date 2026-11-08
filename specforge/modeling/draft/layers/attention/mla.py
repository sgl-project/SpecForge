"""Expanded multi-head latent attention for training, without a KV cache."""

import torch
from torch import Tensor, nn

from specforge.config.draft._validation import positive_int
from specforge.config.draft.attention import MLASpec
from specforge.config.draft.norm import NormSpec
from specforge.config.draft.rope import RopeSpec, YaRNSpec

from ..registry import build_component, register_component_builder
from ..rope import apply_rotary
from .backends import eager_attention
from .inputs import validate_attention_inputs
from .masks import AttentionLayout, dense_attention_mask


class MLAAttention(nn.Module):
    """Own latent projections; reuse the same layout and backend as GQA.

    ``rope`` is the resolved attention override or model default. It determines
    the rotary width and YaRN full-QK multiplier; cosine/sine preparation stays
    in the backbone. K/V expand per head here, not into a compressed serving cache.
    """

    def __init__(
        self,
        config: MLASpec,
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
        if type(config) is not MLASpec:
            raise TypeError("MLAAttention requires MLASpec")
        positive_int("input_size", input_size)
        positive_int("hidden_size", hidden_size)
        if not isinstance(norm, NormSpec) or not isinstance(rope, RopeSpec):
            raise TypeError("norm and rope must be configuration nodes")
        if config.rope_parameters is not None and rope != config.rope_parameters:
            raise ValueError("rope must match the attention RoPE override")
        rotary_dim = rope.rotary_dim(config.qk_rope_head_dim)
        if backend != "eager":
            raise ValueError("MLAAttention currently supports only the eager backend")
        self.config = config
        self.rotary_dim = rotary_dim
        self.scaling = config.qk_head_dim**-0.5
        if isinstance(rope, YaRNSpec):
            self.scaling *= rope.mla_qk_scale_multiplier
        options = dict(device=device, dtype=dtype)
        if config.q_lora_rank is None:
            self.q_proj = nn.Linear(
                input_size,
                config.num_attention_heads * config.qk_head_dim,
                bias=False,
                **options,
            )
        else:
            self.q_a_proj = nn.Linear(
                input_size, config.q_lora_rank, bias=config.attention_bias, **options
            )
            self.q_a_layernorm = build_component(
                "norm", norm, hidden_size=config.q_lora_rank, **options
            )
            self.q_b_proj = nn.Linear(
                config.q_lora_rank,
                config.num_attention_heads * config.qk_head_dim,
                bias=False,
                **options,
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            input_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
            **options,
        )
        self.kv_a_layernorm = build_component(
            "norm", norm, hidden_size=config.kv_lora_rank, **options
        )
        self.kv_b_proj = nn.Linear(
            config.kv_lora_rank,
            config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
            bias=False,
            **options,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.v_head_dim,
            hidden_size,
            bias=config.attention_bias,
            **options,
        )

    def _project_kv(self, hidden: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        node = self.config
        latent, k_rope = self.kv_a_proj_with_mqa(hidden).split(
            (node.kv_lora_rank, node.qk_rope_head_dim), dim=-1
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(latent)).view(
            *hidden.shape[:2],
            node.num_attention_heads,
            node.qk_nope_head_dim + node.v_head_dim,
        )
        k_nope, value = kv.split((node.qk_nope_head_dim, node.v_head_dim), dim=-1)
        return k_nope, k_rope, value

    def forward(
        self,
        hidden: Tensor,
        *,
        layout: AttentionLayout,
        position_embeddings: tuple[Tensor, Tensor] | None,
        target_context: Tensor | None = None,
    ) -> Tensor:
        node = self.config
        context_length = validate_attention_inputs(
            hidden,
            layout,
            position_embeddings,
            target_context,
            input_size=self.kv_a_proj_with_mqa.in_features,
            num_heads=node.num_attention_heads,
            rotary_dim=self.rotary_dim,
        )
        batch, length = hidden.shape[:2]
        query = (
            self.q_proj(hidden)
            if node.q_lora_rank is None
            else self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden)))
        ).view(batch, length, node.num_attention_heads, node.qk_head_dim)
        q_nope, q_rope = query.split(
            (node.qk_nope_head_dim, node.qk_rope_head_dim), dim=-1
        )
        # Keep separate context/draft projections to retain existing GEMM shapes.
        if target_context is None:
            k_nope, k_rope, value = self._project_kv(hidden)
        else:
            context = self._project_kv(target_context)
            draft = self._project_kv(hidden)
            k_nope, k_rope, value = (
                torch.cat((left, right), dim=1) for left, right in zip(context, draft)
            )
        q_rope, k_rope = q_rope.transpose(1, 2), k_rope.unsqueeze(1)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q_rope = apply_rotary(
                q_rope,
                (cos[:, context_length:], sin[:, context_length:]),
                interleaved=node.rope_interleave,
            )
            k_rope = apply_rotary(k_rope, (cos, sin), interleaved=node.rope_interleave)
        query = torch.cat((q_nope.transpose(1, 2), q_rope), dim=-1)
        key = torch.cat(
            (
                k_nope.transpose(1, 2),
                k_rope.expand(-1, node.num_attention_heads, -1, -1),
            ),
            dim=-1,
        )
        mask = dense_attention_mask(
            layout, is_causal=node.is_causal, sliding_window=node.sliding_window
        )
        output = eager_attention(
            query,
            key,
            value.transpose(1, 2).to(query.dtype),
            attention_mask=mask,
            scale=self.scaling,
            dropout_p=node.attention_dropout,
            training=self.training,
        )
        output = self.o_proj(
            output.transpose(1, 2).reshape(
                batch, length, node.num_attention_heads * node.v_head_dim
            )
        )
        return output.masked_fill(~mask.any(dim=-1).any(dim=1).unsqueeze(-1), 0)


register_component_builder("attention", MLASpec, MLAAttention)
