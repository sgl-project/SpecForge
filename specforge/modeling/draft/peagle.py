"""P-EAGLE (Parallel EAGLE) draft model with multi-layer architecture."""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.models.llama.configuration_llama import LlamaConfig

from specforge.modeling.draft.base import Eagle3DraftModel
from specforge.modeling.draft.flex_attention import compile_friendly_flex_attention
from specforge.modeling.draft.llama3_eagle import (
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    rotate_half,
)


class PEagleAttention(nn.Module):
    """Flex-attention layer for P-EAGLE. Accepts pre-computed BlockMask and position embeddings."""

    def __init__(self, config: LlamaConfig, input_size: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = nn.Linear(input_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(
            input_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            input_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = (
            self.q_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_key_value_heads, self.head_dim)
            .transpose(1, 2)
        )

        # position_embeddings: (cos, sin), each [batch, seq_len, head_dim], pre-indexed
        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = sin.unsqueeze(1)
        query_states = (query_states * cos) + (rotate_half(query_states) * sin)
        key_states = (key_states * cos) + (rotate_half(key_states) * sin)

        attn_output = compile_friendly_flex_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            block_mask=attention_mask,
            enable_gqa=True,
            kernel_options={
                "FORCE_USE_FLEX_ATTENTION": True,
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "num_stages": 2,
                "bwd_BLOCK_M1": 32,
                "bwd_BLOCK_N1": 64,
                "bwd_BLOCK_M2": 32,
                "bwd_BLOCK_N2": 32,
            },
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class PEagleFirstLayer(nn.Module):
    """Eagle3-style first decoder layer: splits 2*hidden_size input into embeds and hidden,
    normalizes separately, then runs attention with 2*hidden_size Q/K/V projections."""

    def __init__(self, config: LlamaConfig, norm_before_residual: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.norm_before_residual = norm_before_residual

        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = PEagleAttention(config, input_size=2 * config.hidden_size)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        mid = hidden_states.shape[2] // 2
        embeds, hidden = hidden_states.split(mid, dim=-1)
        residual = hidden

        embeds = self.input_layernorm(embeds)
        hidden = self.hidden_norm(hidden)
        if self.norm_before_residual:
            residual = hidden
        hidden_states = torch.cat([embeds, hidden], dim=-1)

        hidden_states = self.self_attn(
            hidden_states, attention_mask, position_embeddings
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class PEagleStandardLayer(nn.Module):
    """Standard decoder layer for subsequent P-EAGLE layers: hidden_size input."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = PEagleAttention(config, input_size=config.hidden_size)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = LlamaMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, attention_mask, position_embeddings
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class PEagleDraftModel(Eagle3DraftModel):
    """P-EAGLE draft model with multi-layer architecture.

    Architecture follows speculators Eagle3DraftModel:
    - First layer: Eagle3-style with 2*hidden_size Q/K/V (splits embeds + hidden)
    - Subsequent layers: Standard decoder layers with hidden_size Q/K/V
    - External rotary embeddings shared across all layers
    """

    config_class = LlamaConfig

    def __init__(
        self,
        config: LlamaConfig,
        norm_before_residual: bool = False,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.draft_vocab_size = getattr(config, "draft_vocab_size", config.vocab_size)
        self.norm_before_residual = norm_before_residual

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )

        if hasattr(config, "target_hidden_size"):
            fc_input_size = config.target_hidden_size * 3
        else:
            fc_input_size = config.hidden_size * 3
        self.fc = nn.Linear(fc_input_size, config.hidden_size, bias=False)
        self.mask_hidden = nn.Parameter(torch.randn(1, 1, fc_input_size))

        num_layers = config.num_hidden_layers
        layers: List[nn.Module] = [
            PEagleFirstLayer(config, norm_before_residual=norm_before_residual)
        ]
        for _ in range(1, num_layers):
            layers.append(PEagleStandardLayer(config))
        self.layers = nn.ModuleList(layers)

        self.rotary_emb = LlamaRotaryEmbedding(
            dim=getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            ),
            max_position_embeddings=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 10000),
        )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, self.draft_vocab_size, bias=False)

        t2d = torch.ones(self.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden_states))

    def backbone(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: torch.Tensor = None,
        attention_mask=None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Run multi-layer forward pass.

        Args:
            input_embeds: [batch, seq_len, hidden_size] - token embeddings
            hidden_states: [batch, seq_len, hidden_size] - projected aux hidden states
            attention_mask: BlockMask from flex_attention
            position_ids: [batch, seq_len] - position indices
        """
        layer_input = torch.cat([input_embeds, hidden_states], dim=-1)

        cos, sin = self.rotary_emb(layer_input, seq_len=position_ids.max().item() + 1)
        cos = cos.squeeze(0).squeeze(0)
        sin = sin.squeeze(0).squeeze(0)
        cos = cos[position_ids]
        sin = sin[position_ids]
        position_embeddings = (cos, sin)

        h = layer_input
        for layer in self.layers:
            h = layer(h, attention_mask, position_embeddings)
        return h

    def forward(
        self,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask=None,
        position_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Convenience forward that projects hidden states and runs backbone."""
        projected = self.fc(hidden_states)
        h = self.backbone(
            input_embeds=inputs_embeds,
            hidden_states=projected,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        return self.norm(h)
