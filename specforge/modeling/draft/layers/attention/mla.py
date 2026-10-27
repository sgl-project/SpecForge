"""Low-rank attention projections with an explicit expanded-KV training path."""

import math

import torch
from torch import nn
from transformers import PretrainedConfig

from ..config import LayerKernels
from .base import DenseAttention
from .rotary import apply_mla_rope


def resolve_rope_interleaved(config: PretrainedConfig) -> bool:
    value = getattr(config, "rope_interleave", True)
    if not isinstance(value, bool):
        raise ValueError("rope_interleave must be a boolean")
    return value


def validate_mla_config(config: PretrainedConfig) -> None:
    """Validate the standard MLA dimension fields carried by a draft config."""

    required = (
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
    )
    missing = [name for name in required if getattr(config, name, None) is None]
    if missing:
        raise ValueError(f"MLA draft config is missing required fields: {missing}")

    q_lora_rank = getattr(config, "q_lora_rank", None)
    if q_lora_rank is not None and int(q_lora_rank) <= 0:
        raise ValueError(f"q_lora_rank must be positive or null, got {q_lora_rank}")

    for name in ("kv_lora_rank", "qk_rope_head_dim", "v_head_dim"):
        value = int(getattr(config, name))
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    qk_nope_head_dim = int(config.qk_nope_head_dim)
    if qk_nope_head_dim < 0:
        raise ValueError(
            f"qk_nope_head_dim must be non-negative, got {qk_nope_head_dim}"
        )
    qk_rope_head_dim = int(config.qk_rope_head_dim)
    if qk_rope_head_dim % 2:
        raise ValueError(f"qk_rope_head_dim must be even, got {qk_rope_head_dim}")
    resolve_rope_interleaved(config)


class MLAProjections:
    """Low-rank Q/KV with partial RoPE and expanded per-head K/V."""

    def _init_projections(
        self, config: PretrainedConfig, kernels: LayerKernels
    ) -> None:
        validate_mla_config(config)
        self.num_heads = int(config.num_attention_heads)
        self.num_key_value_groups = 1
        self.q_lora_rank = (
            None
            if getattr(config, "q_lora_rank", None) is None
            else int(config.q_lora_rank)
        )
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.v_head_dim = int(config.v_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.head_dim = self.qk_head_dim
        self.scaling = self.qk_head_dim**-0.5
        # DeepSeek YaRN applies mscale_all_dim to the full QK logits; the
        # rotary attention factor separately scales only the partial-RoPE slice.
        rope_parameters = config.rope_parameters
        if rope_parameters.get("rope_type", "default") != "default":
            mscale_all_dim = rope_parameters.get("mscale_all_dim", 0)
            if mscale_all_dim:
                factor = rope_parameters["factor"]
                mscale = (
                    1.0
                    if factor <= 1
                    else 0.1 * mscale_all_dim * math.log(factor) + 1.0
                )
                self.scaling *= mscale * mscale
        self.rope_interleaved = resolve_rope_interleaved(config)

        hidden_size = int(config.hidden_size)
        bias = bool(config.attention_bias)
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )
        else:
            self.q_a_proj = nn.Linear(hidden_size, self.q_lora_rank, bias=bias)
            self.q_a_layernorm = kernels.make_rms_norm(
                self.q_lora_rank,
                config.rms_norm_eps,
            )
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=False,
            )
        self.kv_a_proj_with_mqa = nn.Linear(
            hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=bias,
        )
        self.kv_a_layernorm = kernels.make_rms_norm(
            self.kv_lora_rank,
            config.rms_norm_eps,
        )
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            hidden_size,
            bias=bias,
        )

    def _project_q(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.q_lora_rank is None:
            return self.q_proj(hidden_states)
        return self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

    def _project_kv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len = hidden_states.shape[:2]
        kv_compressed, k_rope = self.kv_a_proj_with_mqa(hidden_states).split(
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_compressed)).view(
            bsz,
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, value = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return k_nope, k_rope, value

    def _compute_qkv(
        self,
        hidden_states: torch.Tensor,
        key_value_prefix: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len = hidden_states.shape[:2]
        query = self._project_q(hidden_states).view(
            bsz,
            q_len,
            self.num_heads,
            self.qk_head_dim,
        )
        q_nope, q_rope = query.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )
        ctx_k_nope, ctx_k_rope, ctx_value = self._project_kv(key_value_prefix)
        self_k_nope, self_k_rope, self_value = self._project_kv(hidden_states)
        k_nope = torch.cat((ctx_k_nope, self_k_nope), dim=1)
        k_rope = torch.cat((ctx_k_rope, self_k_rope), dim=1)
        v = torch.cat((ctx_value, self_value), dim=1)
        # The model-level rotary carries qk_rope_head_dim; queries take the
        # trailing positions exactly like apply_rotary_pos_emb.
        cos, sin = position_embeddings
        q_rope = apply_mla_rope(
            q_rope.transpose(1, 2),
            cos[:, -q_len:],
            sin[:, -q_len:],
            interleaved=self.rope_interleaved,
        )
        k_rope = apply_mla_rope(
            k_rope.unsqueeze(1),
            cos,
            sin,
            interleaved=self.rope_interleaved,
        ).expand(-1, self.num_heads, -1, -1)
        q = torch.cat((q_nope.transpose(1, 2), q_rope), dim=-1)
        k = torch.cat((k_nope.transpose(1, 2), k_rope), dim=-1)
        return q, k, v.transpose(1, 2)


class MLAAttention(MLAProjections, DenseAttention):
    """MLA using expanded K/V; this is not a latent-cache inference backend."""
