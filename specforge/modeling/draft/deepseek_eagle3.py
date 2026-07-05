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
"""DeepSeek MLA (Multi-head Latent Attention) Eagle3 draft model.

The draft-ARCHITECTURE axis counterpart to ``llama3_eagle.py`` (plan.md G4):
the eagle3 TTT *algorithm* is unchanged — same ``Eagle3DraftModel`` interface,
same suffix-attention ``cache_hidden`` convention, same fc/norm/lm_head/t2d
surface — only the attention block is MLA (DeepSeek-V2/V3):

  Q:  [optional q_lora down/up] -> per-head [nope | rope] split
  KV: kv_a_proj (compressed + shared k_rope) -> layernorm -> kv_b_proj
      -> per-head [k_nope | value]
  RoPE is applied ONLY to the rope head dims, with DeepSeek's INTERLEAVED
  pair rotation (dims (0,1),(2,3),... rotate together — not the neox
  half-split rotation Llama uses). We reuse the neox-layout cos/sin caches
  from ``llama3_eagle`` and convert layout at apply time.

Because the eagle3 attention input is ``cat(input_emb, hidden)``, every
down-projection takes ``2*hidden_size`` inputs. ``o_proj`` maps
``num_heads*v_head_dim -> hidden_size`` (v_head_dim != qk_head_dim in MLA).

Backend support: ``sdpa`` (dense mask path + the TTT suffix-cache path).
flex/fa/usp backends for MLA are not wired yet and raise with a pointer —
they need an MLA-shaped kernel treatment (asymmetric q/k vs v head dims).
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn

try:  # transformers >= 4.51 exports it at top level
    from transformers import DeepseekV3Config
except ImportError:  # pragma: no cover - older layouts
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import (
        DeepseekV3Config,
    )

from specforge.modeling.draft.base import Eagle3DraftModel
from specforge.modeling.draft.registry import register_draft
from specforge.modeling.draft.llama3_eagle import (
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaYarnRotaryEmbedding,
    prepare_decoder_attention_mask,
    yarn_get_mscale,
)
from specforge.utils import print_with_rank


def _rope_config_get(rope_scaling, key, default=None):
    """Read a value out of a rope_scaling config (dict or object)."""
    if rope_scaling is None:
        return default
    if isinstance(rope_scaling, dict):
        return rope_scaling.get(key, default)
    return getattr(rope_scaling, key, default)


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Interleaved-pair rotation: pairs dims (0,1), (2,3), ..."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rotary_pos_emb_interleaved(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Apply RoPE with DeepSeek's interleaved rotation from neox-layout caches.

    The cos/sin caches come from the shared ``LlamaRotaryEmbedding`` family
    (neox layout: [t0..t{d/2-1}, t0..t{d/2-1}]); interleaved layout repeats each
    frequency pairwise instead, so ``cache[..., :d/2].repeat_interleave(2)`` is
    the exact conversion.
    """
    cos = cos.squeeze(1).squeeze(0)
    sin = sin.squeeze(1).squeeze(0)
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    half = cos.shape[-1] // 2
    cos = cos[..., :half].repeat_interleave(2, dim=-1)
    sin = sin[..., :half].repeat_interleave(2, dim=-1)
    q_embed = (q * cos) + (_rotate_half_interleaved(q) * sin)
    k_embed = (k * cos) + (_rotate_half_interleaved(k) * sin)
    return q_embed, k_embed


class DeepseekMLAAttention(nn.Module):
    """MLA attention for eagle3 draft training, on the shared TTT cache seam.

    Mirrors ``LlamaAttention``'s contract exactly: input is the layer's
    ``cat(input_emb, hidden)`` (``2*hidden_size``), ``cache_hidden`` is the
    mutable ``[[k_1..k_n], [v_1..v_n]]`` suffix cache appended in place per TTT
    step, and only ``attn_output`` is returned.
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

        # eagle3: the attention input is cat(input_emb, hidden) = 2*hidden.
        input_dim = self.hidden_size * 2

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

    def _project_qkv(self, hidden_states: torch.Tensor):
        """Project to per-head Q [B,H,S,qk], K_nope [B,H,S,nope],
        shared K_rope [B,1,S,rope], V [B,H,S,v]."""
        bsz, q_len, _ = hidden_states.size()

        if self.q_lora_rank is not None:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim).transpose(1, 2)

        kv_combined = self.kv_a_proj_with_mqa(hidden_states)
        kv_compressed, k_rope_raw = torch.split(
            kv_combined, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_compressed))
        kv = kv.view(
            bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, value = torch.split(
            kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        return (
            q,
            k_nope.transpose(1, 2),
            k_rope_raw.unsqueeze(1),  # [B,1,S,rope]: one shared rope head
            value.transpose(1, 2),
        )

    def _rope_and_assemble(self, query_states, k_nope, k_rope_raw, position_ids, lck):
        """RoPE the rope slice (interleaved) and assemble full-width Q/K."""
        q_len = query_states.shape[2]
        q_nope = query_states[..., : self.qk_nope_head_dim]
        q_rope = query_states[..., self.qk_nope_head_dim :]

        cos, sin = self.rotary_emb(q_rope, seq_len=q_len + lck)
        cos, sin = cos.to(q_rope.device), sin.to(q_rope.device)
        q_rope, k_rope = _apply_rotary_pos_emb_interleaved(
            q_rope, k_rope_raw, cos, sin, position_ids + lck
        )
        k_rope = k_rope.expand(-1, self.num_heads, -1, -1)
        query_states = torch.cat([q_nope, q_rope], dim=-1)
        key_states = torch.cat([k_nope, k_rope], dim=-1)
        return query_states, key_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[List[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()
        query_states, k_nope, k_rope_raw, value_states = self._project_qkv(
            hidden_states
        )

        if cache_hidden is None:
            query_states, key_states = self._rope_and_assemble(
                query_states, k_nope, k_rope_raw, position_ids, lck=0
            )
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                is_causal=attention_mask is None,
                dropout_p=0.0,
                scale=self.softmax_scale,
            )
        else:
            # The shared TTT suffix-cache semantics (see LlamaAttention): the
            # first cache entry attends causally, later entries diagonally.
            lck = len(cache_hidden[0])
            query_states, key_states = self._rope_and_assemble(
                query_states, k_nope, k_rope_raw, position_ids, lck=lck
            )
            cache_hidden[0] = cache_hidden[0] + [key_states]
            cache_hidden[1] = cache_hidden[1] + [value_states]
            cache_k = cache_hidden[0]
            cache_v = cache_hidden[1]

            k0 = cache_k[0]
            v0 = cache_v[0]
            attn_weights = (
                torch.matmul(query_states, k0.transpose(2, 3)) * self.softmax_scale
            )
            lck = len(cache_k)
            attn_weights = attn_weights + attention_mask

            for i in range(1, lck):
                ki = cache_k[i]
                attn_weightsi = (query_states * ki).sum(-1) * self.softmax_scale
                attn_weights = torch.cat(
                    (attn_weights, attn_weightsi[..., None]), dim=-1
                )

            attn_weights = nn.functional.softmax(
                attn_weights, dim=-1, dtype=torch.float32
            ).to(query_states.dtype)
            attn_weights0 = attn_weights[..., :q_len]
            attn_output = torch.matmul(attn_weights0, v0)
            for i in range(1, lck):
                vi = cache_v[i]
                attn_weightsi = attn_weights[..., q_len + i - 1]
                attn_output = attn_output + attn_weightsi[..., None] * vi

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        return self.o_proj(attn_output)


class DeepseekEagle3DecoderLayer(nn.Module):
    """Identical layer shape to ``LlamaDecoderLayer`` with MLA attention."""

    def __init__(self, config: DeepseekV3Config, attention_backend: str = "sdpa"):
        super().__init__()
        self.hidden_size = config.hidden_size
        if attention_backend == "sdpa":
            self.self_attn = DeepseekMLAAttention(config=config)
        else:
            raise ValueError(
                f"MLA draft supports the 'sdpa' attention backend only for now, "
                f"got {attention_backend!r}. flex/fa/usp need an MLA-shaped "
                f"kernel treatment (asymmetric q/k vs v head dims)."
            )
        self.attention_backend = attention_backend
        self.mlp = LlamaMLP(config)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[List[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
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

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@register_draft
class DeepseekV3ForCausalLMEagle3(Eagle3DraftModel):
    """MLA Eagle3 draft (DeepSeek/Kimi family) on the unchanged eagle3 surface."""

    config_class = DeepseekV3Config

    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        super().__init__(config)
        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.midlayer = DeepseekEagle3DecoderLayer(
            config, attention_backend=attention_backend
        )

        if hasattr(config, "target_hidden_size") and config.target_hidden_size:
            self.fc = torch.nn.Linear(
                config.target_hidden_size * 3, config.hidden_size, bias=False
            )
        else:
            self.fc = torch.nn.Linear(
                config.hidden_size * 3, config.hidden_size, bias=False
            )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )

        t2d = torch.ones(self.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)

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
        device = hidden_states.device
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), dtype=torch.bool, device=device
            )
        attention_mask = prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), hidden_states, 0
        )

        hidden_states = self.fc(hidden_states)
        hidden_states = self.midlayer(
            input_emb=inputs_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
        )
        return self.norm(hidden_states)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.size(-1) == self.fc.in_features
        return self.fc(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        norm_hidden_states = self.norm(hidden_states)
        return self.lm_head(norm_hidden_states)

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        return self.midlayer(
            input_emb=input_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=False,
            use_cache=False,
        )


__all__ = [
    "DeepseekMLAAttention",
    "DeepseekEagle3DecoderLayer",
    "DeepseekV3ForCausalLMEagle3",
]
