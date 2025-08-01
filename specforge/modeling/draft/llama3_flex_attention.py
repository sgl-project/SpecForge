import torch
import torch.nn as nn
import math
from typing import Optional, List, Tuple
from transformers import LlamaConfig
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from specforge.modeling.draft.llama3_eagle import (
    LlamaRotaryEmbedding, 
    LlamaLinearScalingRotaryEmbedding, 
    LlamaDynamicNTKScalingRotaryEmbedding,
    apply_rotary_pos_emb
)
from transformers.cache_utils import Cache, DynamicCache
from specforge.modeling.draft.flex_attention import generate_eagle3_mask




class LlamaFlexAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
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
            self.hidden_size * 2, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size * 2, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim, max_position_embeddings=self.max_position_embeddings
            )
        else:
            scaling_type = self.config.rope_scaling["rope_type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
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
                # for nv type
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Cache,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        
        cos, sin = self.rotary_emb(query_states, seq_len=q_len)
        cos, sin = cos.to(query_states.device), sin.to(query_states.device)
        # Keep positions ids aligned when padding so the KV cache is unaffected.
        lck = past_seen_tokens % q_len
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids + lck
        )

        cache_position: torch.Tensor = torch.arange(
            past_seen_tokens, past_seen_tokens + q_len, device=hidden_states.device
        )
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        
        key_cache, value_cache = past_key_values.update(
            key_states,
            value_states,
            layer_idx=self.layer_idx,
            cache_kwargs=cache_kwargs,
        )


        seq_lengths = attention_mask.sum(dim=-1)
        # Shrink the attention mask to align with the padding to the right.
        seq_lengths -= lck
        # Flex Attention
        block_mask = create_block_mask(
            mask_mod=generate_eagle3_mask(
                seq_lengths=seq_lengths, Q_LEN=q_len, KV_LEN=key_cache.shape[-2]
            ), 
            B=bsz, # Rely on broadcast
            H=1, # Rely on broadcast
            Q_LEN=q_len, 
            KV_LEN=key_cache.shape[-2],
            device=query_states.device,
        )

        compiled_flex_attention = torch.compile(
            flex_attention, dynamic=True
        )
        attn_output = compiled_flex_attention(
            query=query_states,
            key=key_cache,
            value=value_cache,
            block_mask=block_mask,
            enable_gqa=True,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)
        return attn_output
    




if __name__ == "__main__":
    config_dict = {
        "hidden_size": 512,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-05,
        "vocab_size": 32000,
        "intermediate_size": 1376,
        "hidden_act": "silu",
    }
    config = LlamaConfig(**config_dict)
    llama_flex_attention = LlamaFlexAttention(config)
    llama_flex_attention.eval()
    batch_size = 1
    seq_len = 128 * 4
    hidden_size = config.hidden_size * 2
    # Simulate inputs
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    # Switch to StaticCache
    past_key_values = DynamicCache()
    attention_mask = torch.ones(batch_size, seq_len)
    # First 128 is padding
    attention_mask[:, 512:] = False
    for i in range(4):
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        flex_output = llama_flex_attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
    print(flex_output.shape)
    