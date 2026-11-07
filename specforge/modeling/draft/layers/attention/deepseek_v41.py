"""DeepSeek-V4.1 attention: low-rank MQA, partial RoPE, sinks, grouped output."""

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.cache_utils import Cache
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4GroupedLinear
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    apply_rotary_pos_emb as apply_v4_rotary_pos_emb,
)

from ..config import LayerKernels
from .backends.dense import dense_attention
from .base import AttentionContext, AttentionOutput, DenseAttention


def validate_deepseek_v41_config(config: PretrainedConfig) -> None:
    """Validate the DeepSeek-V4.1 attention fields carried by a draft config."""

    required = (
        "head_dim",
        "qk_rope_head_dim",
        "q_lora_rank",
        "o_groups",
        "o_lora_rank",
    )
    missing = [name for name in required if getattr(config, name, None) is None]
    if missing:
        raise ValueError(
            f"DeepSeek-V4.1 draft config is missing required fields: {missing}"
        )
    if int(config.num_key_value_heads) != 1:
        raise ValueError("DeepSeek-V4.1 attention has one KV head shared by every head")
    rope_head_dim = int(config.qk_rope_head_dim)
    if rope_head_dim % 2 or not 0 < rope_head_dim < int(config.head_dim):
        raise ValueError("qk_rope_head_dim must be even and smaller than head_dim")
    if int(config.num_attention_heads) % int(config.o_groups):
        raise ValueError("o_groups must divide num_attention_heads")
    for name in ("q_lora_rank", "o_lora_rank"):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")


class DeepseekV41Projections:
    """V4.1: q_a/q_b, one KV head shared by all heads, o_a/o_b; no per-head q norm."""

    def _init_projections(
        self, config: PretrainedConfig, kernels: LayerKernels
    ) -> None:
        validate_deepseek_v41_config(config)
        self.num_heads = int(config.num_attention_heads)
        # One KV head serves every query head; the eager path repeats it this often.
        self.num_key_value_groups = self.num_heads
        self.head_dim = int(config.head_dim)
        self.scaling = self.head_dim**-0.5
        hidden_size, eps = int(config.hidden_size), config.rms_norm_eps
        q_lora_rank, o_groups = int(config.q_lora_rank), int(config.o_groups)
        o_rank = o_groups * int(config.o_lora_rank)
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_norm = kernels.make_rms_norm(q_lora_rank, eps)
        self.q_b_proj = nn.Linear(
            q_lora_rank, self.num_heads * self.head_dim, bias=False
        )
        self.kv_proj = nn.Linear(hidden_size, self.head_dim, bias=False)
        self.kv_norm = kernels.make_rms_norm(self.head_dim, eps)
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // o_groups, o_rank, o_groups
        )
        self.o_b_proj = nn.Linear(o_rank, hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.zeros(self.num_heads))

    def o_proj(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Heads in o_groups blocks through o_a_proj, then one mixing linear."""
        grouped = attn_output.unflatten(-1, (self.o_a_proj.n_groups, -1))
        return self.o_b_proj(self.o_a_proj(grouped).flatten(-2))

    def _compute_qkv(
        self,
        hidden_states: torch.Tensor,
        key_value_prefix: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len = hidden_states.shape[:2]
        # The model-level rotary carries qk_rope_head_dim; one angle per pair here.
        cos, sin = position_embeddings
        half = cos.shape[-1] // 2
        cos, sin = cos[..., :half], sin[..., :half]
        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        # V4.1 dropped V4's parameter-free per-head RMS norm on q.
        q = apply_v4_rotary_pos_emb(q, cos[:, -q_len:], sin[:, -q_len:])
        kv = torch.cat([key_value_prefix, hidden_states], dim=1)
        kv = self.kv_norm(self.kv_proj(kv)).unsqueeze(1)
        kv = apply_v4_rotary_pos_emb(kv, cos, sin)
        return q, kv, kv


class DeepseekV41Attention(DeepseekV41Projections, DenseAttention):
    """DeepSeek-V4.1 attention on the dense backends; its sinks need eager or flex."""

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
        prefix = context.key_value_prefix
        if prefix is None:
            prefix = hidden_states[:, :0]
        q, kv, _ = self._compute_qkv(hidden_states, prefix, context.position_embeddings)
        if state is not None:
            cos, sin = context.position_embeddings
            kv, _ = state.update(
                kv,
                kv,
                self.layer_idx,
                {"sin": sin, "cos": cos, "cache_position": context.cache_position},
            )
        output, weights, _ = dense_attention(
            self,
            q,
            kv,
            kv,
            context.attention_mask,
            query_positions=context.query_positions,
            key_positions=context.key_positions,
            mask_encodes_visibility=context.mask_encodes_visibility,
            is_causal=self.is_causal if is_causal is None else is_causal,
            **backend_kwargs,
        )
        # K is V, so the output's rope slice carries the keys' rotation; the conjugate
        # rotation at the query positions turns it into DeepSeek's relative rotation.
        q_len = output.shape[1]
        cos, sin = context.position_embeddings
        half = cos.shape[-1] // 2
        cos, sin = cos[:, -q_len:, :half], sin[:, -q_len:, :half]
        output = apply_v4_rotary_pos_emb(output, cos, -sin, unsqueeze_dim=2)
        output = self.o_proj(output.flatten(-2))
        return AttentionOutput(output, state, weights if output_attentions else None)
