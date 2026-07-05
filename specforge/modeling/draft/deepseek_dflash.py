# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Portions of the MLA attention math are adapted from TorchSpec
# (https://github.com/torchspec-project/TorchSpec,
# torchspec/models/draft/deepseek_eagle.py), MIT License,
# Copyright (c) 2026 LightSeek Foundation. The above permission notice of the
# MIT license applies to those portions.
"""DeepSeek MLA (Multi-head Latent Attention) DFlash draft model.

The draft-ARCHITECTURE counterpart to ``dflash.py`` (a Qwen3-style GQA draft),
just as ``deepseek_eagle3.py`` is the MLA counterpart to ``llama3_eagle.py``:
the DFlash *algorithm* is unchanged — same block-parallel context+noise
attention, same ``OnlineDFlashModel`` training wrapper, same
``fc``/``hidden_norm``/``norm`` surface and ``target_layer_ids`` capture
contract — only the attention block is MLA (DeepSeek-V2/V3).

DFlash attention differs from the eagle3 suffix-cache: the query stream is the
per-block *noise* embeddings and the keys/values are ``cat(context, noise)``
along the sequence axis (see ``dflash.Qwen3DFlashAttention``). The MLA port
keeps that exactly and only swaps the projection geometry:

  Q:  [optional q_lora down/up] -> per-head [nope | rope]  (from noise)
  KV: kv_a_proj (compressed + shared k_rope) -> layernorm -> kv_b_proj
      -> per-head [k_nope | value]                        (from context AND noise)
  RoPE is applied ONLY to the rope head dims with DeepSeek's INTERLEAVED pair
  rotation; the query gets the block (draft) positions and the key gets the
  full ``cat(context, noise)`` positions. ``o_proj`` maps
  ``num_heads*v_head_dim -> hidden_size`` (v_head_dim != qk_head_dim in MLA).

Backend support: the ``sdpa`` dense-mask training path only. flex/fa/usp for
MLA are not wired (asymmetric q/k vs v head dims need MLA-shaped kernels), and
``spec_generate`` (the DynamicCache decode helper) is not yet ported — both
raise with a pointer.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # transformers >= 4.51 exports it at top level
    from transformers import DeepseekV3Config
except ImportError:  # pragma: no cover - older layouts
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
        DeepseekV3Config,
    )

try:  # canonical location since transformers ~4.53
    from transformers.modeling_layers import GradientCheckpointingLayer
except ImportError:  # pragma: no cover - older layouts
    from transformers.models.qwen3.modeling_qwen3 import GradientCheckpointingLayer

from transformers.modeling_utils import PreTrainedModel

from specforge.modeling.draft.deepseek_eagle3 import (
    _rope_config_get,
    _rotate_half_interleaved,
)
from specforge.modeling.draft.dflash import build_target_layer_ids
from specforge.modeling.draft.llama3_eagle import (
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaYarnRotaryEmbedding,
    yarn_get_mscale,
)
from specforge.modeling.draft.registry import register_draft


def _apply_interleaved_rope_single(x, cos, sin, position_ids, unsqueeze_dim=1):
    """Interleaved-pair RoPE for ONE tensor at its own positions.

    DFlash's query and key live at different positions (block/draft positions
    vs the full ``cat(context, noise)`` positions), so — unlike the eagle3
    helper — this rotates a single tensor with a single ``position_ids``.

    ``cos``/``sin`` are neox-layout caches from the shared
    ``LlamaRotaryEmbedding`` family ([1, 1, seq, dim]); interleaved layout
    repeats each frequency pairwise, so ``cache[..., :d/2].repeat_interleave(2)``
    is the exact conversion.
    """
    cos = cos.squeeze(1).squeeze(0)  # [seq, dim]
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # [B, 1, L, dim]
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    half = cos.shape[-1] // 2
    cos = cos[..., :half].repeat_interleave(2, dim=-1)
    sin = sin[..., :half].repeat_interleave(2, dim=-1)
    return (x * cos) + (_rotate_half_interleaved(x) * sin)


class DeepseekDFlashAttention(nn.Module):
    """MLA attention on the DFlash context+noise seam.

    Query comes from the per-block noise stream; keys/values come from
    ``cat(context, noise)`` along the sequence axis. Only ``attn_output`` is
    returned (the DFlash decoder layer owns the residual/MLP), matching
    ``dflash.Qwen3DFlashAttention``'s contract.
    """

    def __init__(self, config: DeepseekV3Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.max_position_embeddings = config.max_position_embeddings

        # DFlash: the attention input is a single hidden-width stream (noise or
        # context), NOT the eagle3 cat(input_emb, hidden). So input_dim == hidden.
        input_dim = self.hidden_size

        if self.q_lora_rank is not None:
            self.q_a_proj = nn.Linear(input_dim, self.q_lora_rank, bias=False)
            self.q_a_layernorm = LlamaRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(
                self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False
            )
        else:
            self.q_proj = nn.Linear(
                input_dim, self.num_heads * self.qk_head_dim, bias=False
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            input_dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = LlamaRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        # v_head_dim per head, NOT qk_head_dim — MLA's asymmetric value width.
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim, self.hidden_size, bias=False
        )

        self._init_rope()
        self.softmax_scale = self._compute_softmax_scale()

    def _init_rope(self):
        """Rotary embedding over qk_rope_head_dim only (the MLA rope slice)."""
        rope_dim = self.qk_rope_head_dim
        rope_scaling = self.config.rope_scaling
        rope_theta = getattr(self.config, "rope_theta", 10000)
        scaling_type = _rope_config_get(
            rope_scaling, "rope_type", _rope_config_get(rope_scaling, "type")
        )

        if rope_scaling is None or scaling_type in (None, "default"):
            self.rotary_emb = LlamaRotaryEmbedding(
                rope_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=rope_theta,
            )
            return
        scaling_factor = _rope_config_get(rope_scaling, "factor")
        if scaling_type == "linear":
            self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                rope_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
            )
        elif scaling_type == "dynamic":
            self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                rope_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
            )
        elif scaling_type == "yarn":
            self.rotary_emb = LlamaYarnRotaryEmbedding(
                rope_dim,
                max_position_embeddings=self.max_position_embeddings,
                original_max_position_embeddings=_rope_config_get(
                    rope_scaling, "original_max_position_embeddings"
                ),
                scaling_factor=scaling_factor,
                beta_fast=_rope_config_get(rope_scaling, "beta_fast"),
                beta_slow=_rope_config_get(rope_scaling, "beta_slow"),
                mscale=_rope_config_get(rope_scaling, "mscale"),
                mscale_all_dim=_rope_config_get(rope_scaling, "mscale_all_dim"),
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type} for MLA draft")

    def _compute_softmax_scale(self) -> float:
        """1/sqrt(qk_head_dim), with the YaRN mscale correction when configured."""
        rope_scaling = self.config.rope_scaling
        scaling_type = _rope_config_get(
            rope_scaling, "rope_type", _rope_config_get(rope_scaling, "type")
        )
        if scaling_type == "yarn":
            factor = _rope_config_get(rope_scaling, "factor", 1.0)
            mscale_all_dim = _rope_config_get(rope_scaling, "mscale_all_dim", 0)
            mscale = yarn_get_mscale(factor, mscale_all_dim)
            return (mscale * mscale) / math.sqrt(self.qk_head_dim)
        return 1.0 / math.sqrt(self.qk_head_dim)

    def _project_q(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """noise -> per-head Q [B, H, q_len, qk_head_dim]."""
        bsz, q_len, _ = hidden_states.size()
        if self.q_lora_rank is not None:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q = self.q_proj(hidden_states)
        return q.view(bsz, q_len, self.num_heads, self.qk_head_dim).transpose(1, 2)

    def _project_kv(self, hidden_states: torch.Tensor):
        """stream -> K_nope [B,H,L,nope], shared K_rope [B,1,L,rope], V [B,H,L,v]."""
        bsz, seq_len, _ = hidden_states.size()
        kv_combined = self.kv_a_proj_with_mqa(hidden_states)
        kv_compressed, k_rope_raw = torch.split(
            kv_combined, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_compressed))
        kv = kv.view(
            bsz, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, value = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        return (
            k_nope.transpose(1, 2),
            k_rope_raw.unsqueeze(1),  # [B,1,L,rope]: one shared rope head
            value.transpose(1, 2),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        ctx_len = target_hidden.shape[1]

        # Q from noise; K/V from cat(context, noise) along the sequence axis.
        query_states = self._project_q(hidden_states)
        k_nope_ctx, k_rope_ctx, v_ctx = self._project_kv(target_hidden)
        k_nope_noise, k_rope_noise, v_noise = self._project_kv(hidden_states)
        k_nope = torch.cat([k_nope_ctx, k_nope_noise], dim=2)
        k_rope_raw = torch.cat([k_rope_ctx, k_rope_noise], dim=2)
        value_states = torch.cat([v_ctx, v_noise], dim=2)

        # RoPE the rope slice: query at the block (draft) positions, key at the
        # full cat(context, noise) positions. position_ids is the concatenated
        # [context | draft] index vector of length ctx_len + q_len.
        max_pos = int(position_ids.max().item()) + 1
        cos, sin = self.rotary_emb(hidden_states, seq_len=max_pos)
        cos, sin = cos.to(hidden_states.device), sin.to(hidden_states.device)
        q_pos = position_ids[:, -q_len:]
        k_pos = position_ids

        q_nope = query_states[..., : self.qk_nope_head_dim]
        q_rope = query_states[..., self.qk_nope_head_dim :]
        q_rope = _apply_interleaved_rope_single(q_rope, cos, sin, q_pos)
        k_rope = _apply_interleaved_rope_single(k_rope_raw, cos, sin, k_pos)
        k_rope = k_rope.expand(-1, self.num_heads, -1, -1)
        query_states = torch.cat([q_nope, q_rope], dim=-1)
        key_states = torch.cat([k_nope, k_rope], dim=-1)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            is_causal=attention_mask is None,
            dropout_p=0.0,
            scale=self.softmax_scale,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        return self.o_proj(attn_output)


class DeepseekDFlashDecoderLayer(GradientCheckpointingLayer):
    """DFlash decoder layer with MLA attention (mirrors Qwen3DFlashDecoderLayer)."""

    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = DeepseekDFlashAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # target_hidden is the shared, model-level pre-normed context; the noise
        # stream is normed per layer (matches dflash.Qwen3DFlashDecoderLayer).
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@register_draft
class DeepseekDFlashDraftModel(PreTrainedModel):
    """MLA (DeepSeek/Kimi family) DFlash draft on the unchanged DFlash surface."""

    config_class = DeepseekV3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekDFlashDecoderLayer"]
    # The forward always uses torch SDPA; declare it so PreTrainedModel's attn
    # validation accepts a config with `_attn_implementation="sdpa"`. The other
    # flags keep construction from failing, but only sdpa is actually wired —
    # __init__ raises for anything else (flex/fa need MLA-shaped kernels).
    _supports_sdpa = True
    _supports_flash_attn = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        attn_impl = getattr(config, "_attn_implementation", None)
        if attn_impl not in (None, "sdpa", "eager"):
            raise ValueError(
                f"DeepseekDFlashDraftModel supports the 'sdpa' attention backend "
                f"only, got {attn_impl!r}. flex/fa need an MLA-shaped kernel "
                f"treatment (asymmetric q/k vs v head dims); pass "
                f"--attention-backend sdpa."
            )
        self.layers = nn.ModuleList(
            [
                DeepseekDFlashDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.target_layer_ids = dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = dflash_config.get("mask_token_id", None)
        self.projector_type = dflash_config.get("projector_type", None)
        self.pure_draft_prefix_len = dflash_config.get("pure_draft_prefix_len", 0)
        self.shift_label = dflash_config.get("shift_label", False)

        if self.projector_type is not None:
            # The domino projector (GRU prefix + embed head) is not ported to MLA.
            raise ValueError(
                f"DeepseekDFlashDraftModel supports the standard DFlash head only; "
                f"projector_type={self.projector_type!r} is not supported for MLA."
            )
        self.post_init()

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", 0.02)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        for layer in self.layers:
            hidden_states = layer(
                target_hidden=target_hidden,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(self, *args, **kwargs):
        raise NotImplementedError(
            "spec_generate (the DynamicCache decode helper) is not yet ported to "
            "the MLA DFlash draft — MLA's compressed-KV cache needs a separate "
            "treatment. Training (OnlineDFlashModel) uses the dense forward and "
            "is fully supported. See deepseek_dflash.py module docstring."
        )


__all__ = [
    "DeepseekDFlashAttention",
    "DeepseekDFlashDecoderLayer",
    "DeepseekDFlashDraftModel",
]
