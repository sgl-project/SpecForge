# coding=utf-8
# Copyright 2025 SpecForge Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Qwen3 MoE MTP-Compatible Draft Model.

This module implements a draft model that is **structurally identical** to the
target Qwen3.5 model's native MTP (Multi-Token Prediction) head. After training,
the weights can be directly merged back into the target model's MTP head for
native MTP inference, with no separate Eagle3 framework needed at inference time.

Key differences from Eagle3 (qwen3_moe_eagle.py):
- fc: Linear(hidden_size*2, hidden_size) — fuses embedding + 1 hidden state
  (vs Eagle3's 3*hidden_size for 3 target layers)
- q_proj: Linear(hidden_size, num_heads*head_dim*2) — standard input, output
  includes sigmoid gate for attn_output_gate
  (vs Eagle3's Linear(hidden_size*2, num_heads*head_dim))
- k_proj/v_proj: Linear(hidden_size, ...) — standard input
  (vs Eagle3's Linear(hidden_size*2, ...))
- Output gating: attn_output *= sigmoid(gate) where gate comes from q_proj's
  second half (not present in Eagle3)
- Decoder layer: standard pre-norm transformer (no input_emb concatenation)
  (vs Eagle3's cat(input_emb, hidden) → attention)
- project_hidden_states: identity pass-through (fc fusion moved to backbone)
  (vs Eagle3's fc(3_layer_hidden) called once before TTT loop)
- backbone: fc(cat(norm(emb), norm(hidden))) → midlayer(fused)
  (vs Eagle3's midlayer(input_emb, hidden))
- lm_head: Linear(hidden_size, vocab_size) — full vocab (248320)
  (vs Eagle3's Linear(hidden_size, draft_vocab_size=32000))
- Hidden states: 1 layer from target (last transformer layer)
  (vs Eagle3's 3 layers from low/mid/high)
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from specforge.utils import print_with_rank

from .base import Eagle3DraftModel
from .llama3_eagle import (
    LlamaRMSNorm,
    prepare_decoder_attention_mask,
)

# Re-export MoE components from the Eagle3 module (identical structure)
from .qwen3_moe_eagle import (
    Qwen3MoeDraftMLP,
    Qwen3MoeDraftSparseMoeBlock,
)


# ---------------------------------------------------------------------------
# MTP Attention classes
# ---------------------------------------------------------------------------
# These differ from Eagle3 attention in three fundamental ways:
# 1. q/k/v_proj input is hidden_size (not 2*hidden_size)
# 2. q_proj output is num_heads * head_dim * 2 (includes gate for output gating)
# 3. Output gating: attn_output *= sigmoid(gate)
#
# Because the projection dimensions are fundamentally different from
# LlamaAttention (which uses hidden_size*2 input), we implement these
# attention classes from scratch, reusing only RoPE and cache_hidden logic.
# ---------------------------------------------------------------------------


class Qwen3MoeMTPAttention(nn.Module):
    """MTP-compatible attention (SDPA) with output gating, q_norm and k_norm."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        # MTP q_proj: input=hidden_size, output=num_heads*head_dim*2 (includes gate)
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim * 2, bias=False
        )
        # MTP k/v_proj: input=hidden_size (standard)
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        # q_norm / k_norm (matching target MTP attention)
        self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self._init_rope()

    def _init_rope(self):
        from .llama3_eagle import (
            LlamaRotaryEmbedding,
            LlamaLinearScalingRotaryEmbedding,
            LlamaDynamicNTKScalingRotaryEmbedding,
            LlamaMutiRotaryEmbedding,
            LlamaYarnRotaryEmbedding,
        )

        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=getattr(self.config, "rope_theta", 10000),
            )
        else:
            rope_scaling = self.config.rope_scaling

            def rope_get(key, default=None):
                if isinstance(rope_scaling, dict):
                    return rope_scaling.get(key, default)
                return getattr(rope_scaling, key, default)

            scaling_type = rope_get("rope_type", rope_get("type"))
            scaling_factor = rope_get("factor")

            if scaling_type == "default":
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=getattr(self.config, "rope_theta", 10000),
                )
            elif scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "llama3":
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=getattr(self.config, "rope_theta", 10000),
                    scaling_factor=(
                        scaling_factor if scaling_factor is not None else 1.0
                    ),
                    low_freq_factor=rope_get("low_freq_factor"),
                    high_freq_factor=rope_get("high_freq_factor"),
                    orig_max_position=rope_get("original_max_position_embeddings"),
                )
            elif scaling_type == "mrope":
                self.rotary_emb = LlamaMutiRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                )
            elif scaling_type == "yarn":
                self.rotary_emb = LlamaYarnRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    original_max_position_embeddings=rope_get(
                        "original_max_position_embeddings"
                    ),
                    scaling_factor=scaling_factor,
                    beta_fast=rope_get("beta_fast"),
                    beta_slow=rope_get("beta_slow"),
                    mscale=rope_get("mscale"),
                    mscale_all_dim=rope_get("mscale_all_dim"),
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _split_qkv_and_gate(self, query_states, bsz, q_len):
        """Split q_proj output into query states and output gate."""
        # q_proj output: (bsz, q_len, num_heads * head_dim * 2)
        # Split into query (num_heads * head_dim) and gate (num_heads * head_dim)
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim * 2
        )
        # Split along head_dim: first half is query, second half is gate
        query, gate = query_states.split(self.head_dim, dim=-1)
        # gate shape: (bsz, q_len, num_heads, head_dim) → will be reshaped later
        return query, gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        import math
        from .llama3_eagle import (
            apply_rotary_pos_emb,
            apply_multimodal_rotary_pos_emb,
            repeat_kv,
            LlamaMutiRotaryEmbedding,
        )

        bsz, q_len, _ = hidden_states.size()

        # Project and split query + gate
        q_out = self.q_proj(hidden_states)
        query_states, gate = self._split_qkv_and_gate(q_out, bsz, q_len)
        # query_states: (bsz, q_len, num_heads, head_dim)
        # gate: (bsz, q_len, num_heads, head_dim)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Transpose to (bsz, num_heads, q_len, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # q_norm / k_norm (before RoPE)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        if cache_hidden is None:
            if isinstance(self.rotary_emb, LlamaMutiRotaryEmbedding):
                cos, sin = self.rotary_emb(query_states, position_ids)
                cos, sin = cos.to(query_states.device), sin.to(query_states.device)
                query_states, key_states = apply_multimodal_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                    self.config.rope_scaling["mrope_section"],
                )
            else:
                cos, sin = self.rotary_emb(query_states, seq_len=q_len)
                cos, sin = cos.to(query_states.device), sin.to(query_states.device)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, position_ids
                )

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
                dropout_p=0.0,
            )

        else:
            lck = len(cache_hidden[0])
            if isinstance(self.rotary_emb, LlamaMutiRotaryEmbedding):
                cos, sin = self.rotary_emb(query_states, position_ids + lck)
                cos, sin = cos.to(query_states.device), sin.to(query_states.device)
                query_states, key_states = apply_multimodal_rotary_pos_emb(
                    query_states,
                    key_states,
                    cos,
                    sin,
                    self.config.rope_scaling["mrope_section"],
                )
            else:
                cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
                cos, sin = cos.to(query_states.device), sin.to(query_states.device)
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, position_ids + lck
                )

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            cache_hidden[0] = cache_hidden[0] + [key_states]
            cache_hidden[1] = cache_hidden[1] + [value_states]

            cache_k = cache_hidden[0]
            cache_v = cache_hidden[1]

            k0 = cache_k[0]
            v0 = cache_v[0]

            # causal
            attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(
                self.head_dim
            )
            lck = len(cache_k)

            attn_weights = attn_weights + attention_mask

            for i in range(1, lck):
                ki = cache_k[i]
                qi = query_states
                kiq = ki

                attn_weightsi = (qi * kiq).sum(-1) / math.sqrt(self.head_dim)
                attn_weights = torch.cat(
                    (attn_weights, attn_weightsi[..., None]), dim=-1
                )

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights0 = attn_weights[..., :q_len]

            attn_output = torch.matmul(attn_weights0, v0)

            for i in range(1, lck):
                vi = cache_v[i]
                attn_weightsi = attn_weights[..., q_len + i - 1]
                attn_outputi = attn_weightsi[..., None] * vi
                attn_output = attn_output + attn_outputi

        # attn_output: (bsz, num_heads, q_len, head_dim) → (bsz, q_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # --- Output gating (MTP-specific) ---
        # gate shape: (bsz, q_len, num_heads, head_dim)
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)

        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen3MoeMTPFlexAttention(nn.Module):
    """MTP-compatible FlexAttention with output gating, q_norm and k_norm."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim * 2, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Share RoPE init with SDPA attention
        _attn = Qwen3MoeMTPAttention.__new__(Qwen3MoeMTPAttention)
        _attn.config = config
        _attn.head_dim = self.head_dim
        _attn.max_position_embeddings = self.max_position_embeddings
        _attn._init_rope()
        self.rotary_emb = _attn.rotary_emb

    def _split_qkv_and_gate(self, query_states, bsz, q_len):
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim * 2
        )
        query, gate = query_states.split(self.head_dim, dim=-1)
        return query, gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        from .llama3_eagle import (
            apply_rotary_pos_emb,
            apply_multimodal_rotary_pos_emb,
            LlamaMutiRotaryEmbedding,
            generate_eagle3_mask,
            create_block_mask,
            compile_friendly_create_block_mask,
            flex_attention,
            compile_friendly_flex_attention,
        )

        bsz, q_len, _ = hidden_states.size()

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )

        q_out = self.q_proj(hidden_states)
        query_states, gate = self._split_qkv_and_gate(q_out, bsz, q_len)
        # query_states: (bsz, q_len, num_heads, head_dim)

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Transpose: (bsz, num_heads, q_len, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # q_norm / k_norm (before RoPE)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        lck = past_seen_tokens // q_len
        if isinstance(self.rotary_emb, LlamaMutiRotaryEmbedding):
            cos, sin = self.rotary_emb(query_states, position_ids + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                self.config.rope_scaling["mrope_section"],
            )
        else:
            cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids + lck
            )

        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + q_len, device=hidden_states.device
        )
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        key_cache, value_cache = past_key_values.update(
            key_states,
            value_states,
            layer_idx=0,
            cache_kwargs=cache_kwargs,
        )

        seq_lengths = attention_mask.sum(dim=-1)
        seq_lengths -= lck
        if q_len <= 128:
            create_block_mask_func = create_block_mask
            flex_attention_func = flex_attention
        else:
            create_block_mask_func = compile_friendly_create_block_mask
            flex_attention_func = compile_friendly_flex_attention

        block_mask = create_block_mask_func(
            mask_mod=generate_eagle3_mask(
                seq_lengths=seq_lengths,
                Q_LEN=q_len,
                KV_LEN=key_cache.shape[-2],
                lck=lck,
            ),
            B=bsz,
            H=1,
            Q_LEN=q_len,
            KV_LEN=key_cache.shape[-2],
            device=query_states.device,
        )
        attn_output = flex_attention_func(
            query=query_states,
            key=key_cache.contiguous(),
            value=value_cache.contiguous(),
            block_mask=block_mask,
            enable_gqa=True,
        )
        # (bsz, num_heads, q_len, head_dim) → (bsz, q_len, num_heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Output gating
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)
        return attn_output


class Qwen3MoeMTPFlashAttention(nn.Module):
    """MTP-compatible FlashAttention with output gating, q_norm and k_norm."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim * 2, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        _attn = Qwen3MoeMTPAttention.__new__(Qwen3MoeMTPAttention)
        _attn.config = config
        _attn.head_dim = self.head_dim
        _attn.max_position_embeddings = self.max_position_embeddings
        _attn._init_rope()
        self.rotary_emb = _attn.rotary_emb

    def _split_qkv_and_gate(self, query_states, bsz, q_len):
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim * 2
        )
        query, gate = query_states.split(self.head_dim, dim=-1)
        return query, gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        import math
        from .llama3_eagle import (
            apply_rotary_pos_emb,
            apply_multimodal_rotary_pos_emb,
            LlamaMutiRotaryEmbedding,
            flash_attn_func,
        )

        bsz, q_len, _ = hidden_states.size()

        q_out = self.q_proj(hidden_states)
        query_states, gate = self._split_qkv_and_gate(q_out, bsz, q_len)
        # query_states: (bsz, q_len, num_heads, head_dim) — flash attn layout

        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

        # q_norm / k_norm (before RoPE)
        # flash attention uses (bsz, q_len, num_heads, head_dim) layout
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        lck = 0 if cache_hidden is None else len(cache_hidden[0])
        if isinstance(self.rotary_emb, LlamaMutiRotaryEmbedding):
            cos, sin = self.rotary_emb(query_states, position_ids + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_multimodal_rotary_pos_emb(
                query_states,
                key_states,
                cos,
                sin,
                self.config.rope_scaling["mrope_section"],
                unsqueeze_dim=2,
            )
        else:
            cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
            cos, sin = cos.to(query_states.device), sin.to(query_states.device)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin, position_ids + lck, unsqueeze_dim=2
            )

        if cache_hidden is not None:
            cache_hidden[0] = cache_hidden[0] + [key_states]
            cache_hidden[1] = cache_hidden[1] + [value_states]

            cache_k = cache_hidden[0]
            cache_v = cache_hidden[1]
        else:
            cache_k = [key_states]
            cache_v = [value_states]

        k0 = cache_k[0]
        v0 = cache_v[0]

        assert (
            flash_attn_func is not None
        ), "flash_attn is not installed, please install flash_attn if you want to use the flash attention backend"
        attn_output, lse, _ = flash_attn_func(
            query_states,
            k0,
            v0,
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
            causal=True,
            return_attn_probs=True,
        )
        lse = lse.transpose(1, 2)

        lck = len(cache_k)
        if lck > 1:
            q_shape_expanded = (
                bsz,
                q_len,
                self.num_key_value_heads,
                self.num_key_value_groups,
                self.head_dim,
            )
            attn_outputs = [attn_output.view(q_shape_expanded)]
            lses = [lse.view(q_shape_expanded[:-1])]

            for i in range(1, lck):
                ki = cache_k[i].unsqueeze(-2)
                qi = query_states.view(q_shape_expanded)
                vi = cache_v[i].unsqueeze(-2)

                attn_outputs.append(vi)
                lses.append((qi * ki).sum(-1) / math.sqrt(self.head_dim))

            lse = torch.logsumexp(torch.stack(lses, dim=-1), dim=-1)
            attn_output = sum(
                attn_outputi * torch.exp(lsei - lse).unsqueeze(-1)
                for attn_outputi, lsei in zip(attn_outputs, lses)
            )
            # lse is fp32, downcast attn_output back
            attn_output = attn_output.to(self.o_proj.weight.dtype)

        # attn_output: (bsz, q_len, num_heads, head_dim) in flash attn layout
        # Output gating (applied before reshape)
        attn_output = attn_output * torch.sigmoid(gate)

        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)

        attn_output = self.o_proj(attn_output)

        return attn_output


class Qwen3MoeMTPUSPFlashAttention(nn.Module):
    """MTP-compatible USP FlashAttention with output gating, q_norm and k_norm."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        if hasattr(config, "head_dim"):
            self.head_dim = config.head_dim
        else:
            self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim * 2, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.q_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = LlamaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        _attn = Qwen3MoeMTPAttention.__new__(Qwen3MoeMTPAttention)
        _attn.config = config
        _attn.head_dim = self.head_dim
        _attn.max_position_embeddings = self.max_position_embeddings
        _attn._init_rope()
        self.rotary_emb = _attn.rotary_emb

        # USP-specific: import SP group info
        from .llama3_eagle import get_sp_ring_group, get_sp_ulysses_group

        self.ring_pg = get_sp_ring_group()
        self.ulysses_pg = get_sp_ulysses_group()
        self.sp_ring_degree = self.ring_pg.size()
        self.sp_ulysses_degree = self.ulysses_pg.size()

        self.scatter_idx = 1  # seq dim
        self.gather_idx = 2  # head dim
        self.use_sync = True

    def _split_qkv_and_gate(self, query_states, bsz, q_len):
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim * 2
        )
        query, gate = query_states.split(self.head_dim, dim=-1)
        return query, gate

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        import math
        from .llama3_eagle import (
            apply_rotary_pos_emb,
            LlamaMutiRotaryEmbedding,
            SeqAllToAll4D,
            ring_flash_attn_func,
            get_sp_ring_group,
            get_sp_ulysses_group,
        )

        bsz, q_len, _ = hidden_states.size()
        local_q_len = q_len

        # 1. Projections & split gate
        q_out = self.q_proj(hidden_states)
        query_states, gate = self._split_qkv_and_gate(q_out, bsz, q_len)
        # query_states: (bsz, q_len, num_heads, head_dim)

        # Ulysses scatter for query
        query_states = SeqAllToAll4D.apply(
            self.ulysses_pg,
            query_states,
            self.scatter_idx,
            self.gather_idx,
            self.use_sync,
        )

        # Ulysses scatter for gate (same seq/head split as query)
        gate = SeqAllToAll4D.apply(
            self.ulysses_pg,
            gate,
            self.scatter_idx,
            self.gather_idx,
            self.use_sync,
        )

        key_states = self.k_proj(hidden_states)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        key_states = SeqAllToAll4D.apply(
            self.ulysses_pg,
            key_states,
            self.scatter_idx,
            self.gather_idx,
            self.use_sync,
        )

        value_states = self.v_proj(hidden_states)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )
        value_states = SeqAllToAll4D.apply(
            self.ulysses_pg,
            value_states,
            self.scatter_idx,
            self.gather_idx,
            self.use_sync,
        )

        current_q_len = query_states.shape[1]
        local_num_heads = query_states.shape[2]

        # q_norm / k_norm (before RoPE)
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        # Global length calculation (for RoPE)
        global_q_len = q_len * self.sp_ring_degree * self.sp_ulysses_degree

        # 2. RoPE & Cache Management
        lck = 0 if cache_hidden is None else len(cache_hidden[0])

        cos, sin = self.rotary_emb(query_states, seq_len=global_q_len + lck)
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids + lck, unsqueeze_dim=2
        )

        # Update Cache
        if cache_hidden is not None:
            cache_hidden[0] = cache_hidden[0] + [key_states]
            cache_hidden[1] = cache_hidden[1] + [value_states]
            cache_k = cache_hidden[0]
            cache_v = cache_hidden[1]
        else:
            cache_k = [key_states]
            cache_v = [value_states]

        # 3. Hybrid Attention Computation
        # 3.1 Main Sequence (Ring Attention)
        out_ring, lse_ring, _ = ring_flash_attn_func(
            query_states,
            cache_k[0],
            cache_v[0],
            dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
            return_attn_probs=True,
            group=self.ring_pg,
        )

        if lse_ring.dim() == 3 and lse_ring.shape[1] == local_num_heads:
            acc_lse = lse_ring.transpose(1, 2).contiguous()
        else:
            acc_lse = lse_ring

        assert (
            acc_lse.shape[1] == current_q_len
        ), f"LSE seq_len {acc_lse.shape[1]} mismatch with Query seq_len {current_q_len}"

        acc_out = out_ring

        # 3.2 Extras Branches (Eagle3 Point-wise Update)
        if len(cache_k) > 1:
            num_kv_heads_local = cache_k[0].shape[2]
            local_groups = local_num_heads // num_kv_heads_local

            q_shape_expanded = (
                bsz,
                current_q_len,
                num_kv_heads_local,
                local_groups,
                self.head_dim,
            )
            qi_reshaped = query_states.view(q_shape_expanded)

            for i in range(1, len(cache_k)):
                ki = cache_k[i]
                vi = cache_v[i]

                ki_expanded = ki.unsqueeze(-2)

                score_i = (qi_reshaped * ki_expanded).sum(-1) / math.sqrt(self.head_dim)

                step_lse = score_i.view(bsz, current_q_len, -1)

                vi_expanded = vi.unsqueeze(-2)
                step_out = vi_expanded.expand(q_shape_expanded).reshape(acc_out.shape)

                new_lse = torch.logaddexp(acc_lse, step_lse)

                acc_out = acc_out * torch.exp(acc_lse - new_lse).unsqueeze(
                    -1
                ) + step_out * torch.exp(step_lse - new_lse).unsqueeze(-1)

                acc_lse = new_lse

        attn_output = acc_out.to(query_states.dtype)

        # Output gating (before Ulysses gather)
        # gate has already been scattered, same layout as attn_output
        attn_output = attn_output * torch.sigmoid(gate)

        # 4. Ulysses Gather & Output Projection
        attn_output = SeqAllToAll4D.apply(
            self.ulysses_pg,
            attn_output,
            self.gather_idx,
            self.scatter_idx,
            self.use_sync,
        )

        attn_output = attn_output.reshape(
            bsz, local_q_len, self.head_dim * self.num_heads
        )
        attn_output = self.o_proj(attn_output)

        return attn_output


# ---------------------------------------------------------------------------
# MTP Decoder layer
# ---------------------------------------------------------------------------

class Qwen3MoeDecoderLayerMTP(nn.Module):
    """
    MTP-compatible decoder layer.

    Key difference from Eagle3's Qwen3MoeDecoderLayerEagle3:
    - Standard pre-norm transformer: forward takes ONLY hidden_states
      (no input_emb parameter, no concatenation)
    - Uses MTP attention classes with output gating
    """

    def __init__(self, config, attention_backend: str = "sdpa"):
        super().__init__()
        self.hidden_size = config.hidden_size

        if attention_backend == "sdpa":
            self.self_attn = Qwen3MoeMTPAttention(config=config)
        elif attention_backend == "flex_attention":
            print_with_rank("Using flex attention on draft model training!")
            self.self_attn = Qwen3MoeMTPFlexAttention(config=config)
        elif attention_backend == "fa":
            self.self_attn = Qwen3MoeMTPFlashAttention(config=config)
        elif attention_backend == "usp":
            self.self_attn = Qwen3MoeMTPUSPFlashAttention(config=config)
        else:
            raise ValueError(f"Unknown attention backend {attention_backend}")

        self.attention_backend = attention_backend

        # Use MoE block (identical to Eagle3)
        self.mlp = Qwen3MoeDraftSparseMoeBlock(config)

        # Standard pre-attention norm
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: List[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        """
        Standard pre-norm transformer decoder layer.
        NO input_emb parameter — fc fusion is done upstream in the model's backbone().
        """
        residual = hidden_states

        # Pre-attention norm
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention (with output gating, q_norm/k_norm inside)
        hidden_states = self.self_attn(
            cache_hidden=cache_hidden,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # MoE MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Full MTP-compatible draft model
# ---------------------------------------------------------------------------

class Qwen3MoeForCausalLMMTP(Eagle3DraftModel):
    """
    MTP-compatible draft model for Qwen3.5 MoE target models.

    This model is structurally identical to the target model's native MTP head.
    After training, all weights (including fc, q/k/v_proj, lm_head) can be
    directly merged back into the target model's MTP for native MTP inference.

    Key differences from Qwen3MoeForCausalLMEagle3:
    - fc: 2*hidden → hidden (not 3*hidden)
    - project_hidden_states: identity (fc fusion moved to backbone)
    - backbone: does fc(cat(norm(emb), norm(hidden))) → midlayer(fused)
    - lm_head: full vocab (not draft_vocab)
    - q/k/v_proj: standard hidden_size input (not 2*hidden_size)
    - Output gating in attention
    """

    config_class = Qwen3MoeConfig

    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        super().__init__(config)
        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size
        self.draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.midlayer = Qwen3MoeDecoderLayerMTP(
            config, attention_backend=attention_backend
        )

        # fc: Linear(2*hidden, hidden) — fuses embedding + 1 hidden state
        self.fc = torch.nn.Linear(
            config.hidden_size * 2, config.hidden_size, bias=False
        )

        # Pre-FC norms (matching target MTP)
        self.pre_fc_norm_embedding = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # lm_head: FULL target vocab (matches target MTP's shared_head.head exactly).
        # We deliberately keep the full [vocab_size, hidden] matrix so that:
        #   1) load_mtp_weights can copy weights with no shape mismatch
        #   2) export_mtp_weights round-trips losslessly back to the target slot
        #   3) when training on a t2d subset, we still chunk-matmul the full vocab
        #      and project to draft_vocab_size via index_select(d2t) (see compute_logits)
        self.lm_head = nn.Linear(
            config.hidden_size, self.vocab_size, bias=False
        )

        # Vocab mapping buffers — when draft_vocab_size == vocab_size,
        # t2d is all-True and d2t is identity, effectively a pass-through.
        # When training on a subset, base.load_vocab_mapping(file_path) overwrites both.
        t2d = torch.ones(self.vocab_size, dtype=torch.bool)
        d2t = torch.arange(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)

        # Optional row-chunk size for compute_logits (lm_head matmul over full vocab).
        # None ⇒ run the full [N, hidden] @ [hidden, vocab] in one shot.
        # Set via set_lm_head_chunk_size(n) to cap activation memory peak.
        self._lm_head_chunk_size: Optional[int] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ttt_length: int = 1,
    ):
        if ttt_length == 1:
            print_with_rank("using ttt_length 1, no need to cache hidden states")
            cache_hidden = None
        else:
            print_with_rank(f"using ttt_length {ttt_length}, caching hidden states")
            cache_hidden = [[], []]

        batch_size, seq_length, _ = hidden_states.size()

        # make position ids
        device = hidden_states.device
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        # make attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=hidden_states.device
            )
        attention_mask = prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, 0
        )

        # MTP forward: fc(cat(norm(emb), norm(hidden))) → midlayer
        emb_normed = self.pre_fc_norm_embedding(inputs_embeds)
        hid_normed = self.pre_fc_norm_hidden(hidden_states)
        fused = self.fc(torch.cat([emb_normed, hid_normed], dim=-1))

        hidden_states = self.midlayer(
            hidden_states=fused,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
        )

        # norm
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Identity pass-through — fc fusion is done inside backbone() at each TTT step."""
        return hidden_states

    def set_lm_head_chunk_size(self, chunk_size: Optional[int]) -> None:
        """
        Cap the row-chunk size of the lm_head matmul inside compute_logits.

        Set this to a small int (e.g. 1024) when training on huge vocabs
        (e.g. Qwen3.5 248K) to keep the per-step logits activation memory
        bounded. Pass None to disable chunking.
        """
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(
                f"lm_head chunk_size must be a positive int or None, got {chunk_size}"
            )
        self._lm_head_chunk_size = chunk_size

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Project final hidden states to logits.

        Strategy:
          1) Run lm_head over the FULL target vocab (matches target MTP exactly).
          2) If training on a draft vocab subset (draft_vocab_size < vocab_size),
             immediately project each chunk to [..., draft_vocab_size] via d2t
             so the downstream loss only sees the subset.
          3) Optionally chunk along the flattened (B*S) dim to bound peak memory.

        The chunked outputs are concatenated (NOT scatter-assigned) to keep
        autograd clean.
        """
        norm_h = self.norm(hidden_states)
        project = self.draft_vocab_size < self.vocab_size

        orig_shape = norm_h.shape  # (..., hidden)
        H = orig_shape[-1]
        flat_h = norm_h.reshape(-1, H)
        N = flat_h.shape[0]

        chunk = self._lm_head_chunk_size if self._lm_head_chunk_size is not None else N
        if chunk >= N:
            logits = self.lm_head(flat_h)
            if project:
                logits = logits.index_select(-1, self.d2t.to(logits.device))
        else:
            chunks = []
            for start in range(0, N, chunk):
                end = min(start + chunk, N)
                cl = self.lm_head(flat_h[start:end])
                if project:
                    cl = cl.index_select(-1, self.d2t.to(cl.device))
                chunks.append(cl)
            logits = torch.cat(chunks, dim=0)

        out_dim = self.draft_vocab_size if project else self.vocab_size
        return logits.reshape(*orig_shape[:-1], out_dim)

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        MTP-compatible backbone: fc fusion at every TTT step.

        Flow: fc(cat(norm(emb), norm(hidden))) → midlayer(fused)
        """
        emb_normed = self.pre_fc_norm_embedding(input_embeds)
        hid_normed = self.pre_fc_norm_hidden(hidden_states)
        fused = self.fc(torch.cat([emb_normed, hid_normed], dim=-1))

        return self.midlayer(
            hidden_states=fused,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=False,
        )

    @torch.no_grad()
    def load_mtp_weights(
        self,
        model_path: str,
        mtp_layer_idx: int = 0,
        only_lm_head: bool = False,
    ) -> None:
        """
        Load weights from the target model's native MTP head.

        Unlike the Eagle3 version, this MTP-compatible model has identical
        dimensions for ALL parameters (fc, q/k/v_proj, lm_head), so every
        single weight can be loaded without any shape mismatch.

        Args:
            model_path: Path to the target model directory.
            mtp_layer_idx: Index of the MTP block to load from (default: 0).
            only_lm_head: If True, ONLY reload `lm_head.weight` (resume path —
                the rest of the draft weights come from the checkpoint).
        """
        import glob
        import json
        import os

        from safetensors import safe_open

        if not os.path.exists(model_path):
            print_with_rank(
                f"Warning: model_path {model_path} does not exist, skipping MTP weight loading"
            )
            return

        # Find the weight index file
        glob_path = os.path.join(model_path, "*.index.json")
        index_json_paths = glob.glob(glob_path)

        weight_files = {}
        if len(index_json_paths) > 0:
            index_json_path = index_json_paths[0]
            with open(index_json_path, "r") as f:
                index_json = json.load(f)
            weight_map = index_json["weight_map"]
        else:
            # Single file model
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(safetensors_path):
                print_with_rank(
                    f"Warning: No model files found in {model_path}, skipping MTP weight loading"
                )
                return
            weight_map = None

        # Define the MTP weight key patterns
        mtp_prefix = "mtp"
        layer_prefix = f"{mtp_prefix}.layers.{mtp_layer_idx}"
        moe_prefix = f"{layer_prefix}.mlp"

        # Map from MTP weight keys to draft model keys
        weight_mapping = {}

        # --- Input embedding ---
        # Target's MTP head always materializes `mtp.layers.{idx}.embed_tokens.weight`
        # in its sharded checkpoint (regardless of whether target uses the nested
        # multimodal layout `model.language_model.embed_tokens.weight` or the
        # flat `model.embed_tokens.weight`). Sourcing from the MTP-side key keeps
        # this loader agnostic to the target's main-trunk module layout and is
        # also the perfect dual of `export_mtp_weights` (which writes the same key).
        weight_mapping[f"{layer_prefix}.embed_tokens.weight"] = "embed_tokens.weight"

        # --- fc (now dimensions match: 2*hidden → hidden) ---
        weight_mapping[f"{mtp_prefix}.fc.weight"] = "fc.weight"

        # --- Pre-FC norms ---
        weight_mapping[f"{mtp_prefix}.pre_fc_norm_embedding.weight"] = "pre_fc_norm_embedding.weight"
        weight_mapping[f"{mtp_prefix}.pre_fc_norm_hidden.weight"] = "pre_fc_norm_hidden.weight"

        # --- Final norm ---
        weight_mapping[f"{mtp_prefix}.norm.weight"] = "norm.weight"

        # --- Decoder layer norms ---
        weight_mapping[f"{layer_prefix}.input_layernorm.weight"] = "midlayer.input_layernorm.weight"
        weight_mapping[f"{layer_prefix}.post_attention_layernorm.weight"] = "midlayer.post_attention_layernorm.weight"

        # --- Attention: ALL projections (now dimensions match) ---
        weight_mapping[f"{layer_prefix}.self_attn.q_proj.weight"] = "midlayer.self_attn.q_proj.weight"
        weight_mapping[f"{layer_prefix}.self_attn.k_proj.weight"] = "midlayer.self_attn.k_proj.weight"
        weight_mapping[f"{layer_prefix}.self_attn.v_proj.weight"] = "midlayer.self_attn.v_proj.weight"
        weight_mapping[f"{layer_prefix}.self_attn.o_proj.weight"] = "midlayer.self_attn.o_proj.weight"

        # --- Attention: q_norm / k_norm ---
        weight_mapping[f"{layer_prefix}.self_attn.q_norm.weight"] = "midlayer.self_attn.q_norm.weight"
        weight_mapping[f"{layer_prefix}.self_attn.k_norm.weight"] = "midlayer.self_attn.k_norm.weight"

        # --- MoE router ---
        weight_mapping[f"{moe_prefix}.gate.weight"] = "midlayer.mlp.gate.weight"

        # --- MoE experts ---
        for expert_idx in range(self.config.num_experts):
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                src_key = f"{moe_prefix}.experts.{expert_idx}.{proj}.weight"
                dst_key = f"midlayer.mlp.experts.{expert_idx}.{proj}.weight"
                weight_mapping[src_key] = dst_key

        # --- Shared expert ---
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            weight_mapping[f"{moe_prefix}.shared_expert.{proj}.weight"] = f"midlayer.mlp.shared_expert.{proj}.weight"

        # --- Shared expert gate ---
        weight_mapping[f"{moe_prefix}.shared_expert_gate.weight"] = "midlayer.mlp.shared_expert_gate.weight"

        # --- lm_head (now dimensions match: full vocab) ---
        weight_mapping[f"{layer_prefix}.shared_head.head.weight"] = "lm_head.weight"

        # Resume path: only refresh lm_head (everything else comes from the
        # training checkpoint and must NOT be overwritten).
        if only_lm_head:
            lm_head_src = f"{layer_prefix}.shared_head.head.weight"
            weight_mapping = {lm_head_src: "lm_head.weight"}

        # Load and copy weights
        loaded_count = 0
        skipped_shape_mismatch = []
        for src_key, dst_key in weight_mapping.items():
            try:
                if weight_map is not None:
                    if src_key not in weight_map:
                        print_with_rank(
                            f"Warning: {src_key} not found in weight map, skipping"
                        )
                        continue
                    ckpt_file = weight_map[src_key]
                    file_path = os.path.join(model_path, ckpt_file)
                else:
                    file_path = safetensors_path

                with safe_open(file_path, framework="pt") as f:
                    if src_key in f.keys():
                        src_tensor = f.get_tensor(src_key)
                    else:
                        print_with_rank(
                            f"Warning: {src_key} not found in {file_path}, skipping"
                        )
                        continue

                # Navigate to the target parameter
                parts = dst_key.split(".")
                target = self
                for part in parts[:-1]:
                    if part.isdigit():
                        target = target[int(part)]
                    else:
                        target = getattr(target, part)
                param_name = parts[-1]
                param = getattr(target, param_name)

                if param.shape != src_tensor.shape:
                    skipped_shape_mismatch.append(
                        f"{src_key}: src={src_tensor.shape}, dst={param.shape}"
                    )
                    continue

                param.copy_(src_tensor)
                loaded_count += 1

            except Exception as e:
                print_with_rank(
                    f"Warning: Failed to load {src_key} -> {dst_key}: {e}"
                )
                continue

        print_with_rank(
            f"Loaded {loaded_count}/{len(weight_mapping)} MTP weights from {model_path} "
            f"(mtp layer {mtp_layer_idx})"
        )
        if skipped_shape_mismatch:
            print_with_rank(
                f"WARNING: Skipped {len(skipped_shape_mismatch)} weights due to shape mismatch "
                f"(this should NOT happen for MTP-compatible model):"
            )
            for msg in skipped_shape_mismatch:
                print_with_rank(f"  - {msg}")

    @torch.no_grad()
    def export_mtp_weights(
        self,
        output_path: str,
        mtp_layer_idx: int = 0,
    ) -> dict:
        """
        Export trained weights as a state_dict with MTP weight key format.

        This produces a state_dict that can be directly merged back into the
        target model's MTP head weights.

        Args:
            output_path: Path to save the exported weights.
            mtp_layer_idx: MTP block index for key naming (default: 0).

        Returns:
            The state_dict with MTP-format keys.
        """
        mtp_prefix = "mtp"
        layer_prefix = f"{mtp_prefix}.layers.{mtp_layer_idx}"
        moe_prefix = f"{layer_prefix}.mlp"

        # Reverse mapping: draft model key → MTP key
        export_mapping = {
            "fc.weight": f"{mtp_prefix}.fc.weight",
            "pre_fc_norm_embedding.weight": f"{mtp_prefix}.pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight": f"{mtp_prefix}.pre_fc_norm_hidden.weight",
            "norm.weight": f"{mtp_prefix}.norm.weight",
            "midlayer.input_layernorm.weight": f"{layer_prefix}.input_layernorm.weight",
            "midlayer.post_attention_layernorm.weight": f"{layer_prefix}.post_attention_layernorm.weight",
            "midlayer.self_attn.q_proj.weight": f"{layer_prefix}.self_attn.q_proj.weight",
            "midlayer.self_attn.k_proj.weight": f"{layer_prefix}.self_attn.k_proj.weight",
            "midlayer.self_attn.v_proj.weight": f"{layer_prefix}.self_attn.v_proj.weight",
            "midlayer.self_attn.o_proj.weight": f"{layer_prefix}.self_attn.o_proj.weight",
            "midlayer.self_attn.q_norm.weight": f"{layer_prefix}.self_attn.q_norm.weight",
            "midlayer.self_attn.k_norm.weight": f"{layer_prefix}.self_attn.k_norm.weight",
            "midlayer.mlp.gate.weight": f"{moe_prefix}.gate.weight",
            "midlayer.mlp.shared_expert_gate.weight": f"{moe_prefix}.shared_expert_gate.weight",
            "lm_head.weight": f"{layer_prefix}.shared_head.head.weight",
        }

        # Add experts
        for expert_idx in range(self.config.num_experts):
            for proj in ["gate_proj", "up_proj", "down_proj"]:
                src = f"midlayer.mlp.experts.{expert_idx}.{proj}.weight"
                dst = f"{moe_prefix}.experts.{expert_idx}.{proj}.weight"
                export_mapping[src] = dst

        # Add shared expert
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            src = f"midlayer.mlp.shared_expert.{proj}.weight"
            dst = f"{moe_prefix}.shared_expert.{proj}.weight"
            export_mapping[src] = dst

        # Build export state_dict
        mtp_state_dict = {}
        model_state = self.state_dict()
        for draft_key, mtp_key in export_mapping.items():
            if draft_key in model_state:
                mtp_state_dict[mtp_key] = model_state[draft_key]
            else:
                print_with_rank(f"Warning: {draft_key} not found in model state_dict")

        # Also include embedding (it's shared with the target model)
        mtp_state_dict[f"{layer_prefix}.embed_tokens.weight"] = model_state["embed_tokens.weight"]

        torch.save(mtp_state_dict, output_path)
        print_with_rank(
            f"Exported {len(mtp_state_dict)} MTP weights to {output_path} "
            f"(mtp layer {mtp_layer_idx})"
        )
        return mtp_state_dict
