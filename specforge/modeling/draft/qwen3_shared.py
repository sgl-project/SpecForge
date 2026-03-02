# coding=utf-8
"""Single-Model Speculative Decoding with Shared Backend for Qwen3."""

from collections.abc import Callable
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Unpack


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary positional embeddings to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Build target layer IDs for KV extraction.

    Args:
        num_target_layers: Total number of layers in the target model
        num_draft_layers: Number of draft layers

    Returns:
        List of target layer IDs evenly spaced across the target model
    """
    if num_draft_layers == 1:
        return [num_target_layers // 2]

    # Evenly space target layers from early to middle of the network
    # Target layer mapping for 5 draft layers in 32-layer model: [0, 6, 13, 20, 26]
    start = 0
    end = num_target_layers - 5  # Leave some buffer before the final layer
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


class Qwen3SharedDraftAttention(nn.Module):
    """Multi-headed attention for shared backend speculative decoding.

    This layer implements bi-directional attention where:
    - Q comes from the draft layer's hidden states
    - K/V come from both the target layer's hidden states (context) and draft layer's hidden states
    """

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False  # Bi-directional attention

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_kv: Tuple[torch.Tensor, torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass with bi-directional attention.

        Args:
            hidden_states: Draft layer hidden states [batch, seq_len, hidden_size]
            target_kv: Tuple of (target_K, target_V) from the corresponding target layer
            position_embeddings: Cos and sin for rotary embeddings
            attention_mask: Attention mask for bi-directional attention
            past_key_values: Past key values for caching
            cache_position: Cache positions

        Returns:
            Tuple of (output_hidden_states, (output_K, output_V))
        """
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_kv[0].shape[1]

        # Project Q from draft hidden states
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)

        # Project K from both target context and draft hidden states
        k_ctx = self.k_proj(target_kv[0])  # K from target layer
        k_draft = self.k_proj(hidden_states)  # K from draft layer

        # Project V from both target context and draft hidden states
        v_ctx = self.v_proj(target_kv[0])  # V from target layer
        v_draft = self.v_proj(hidden_states)  # V from draft layer

        # Concatenate context (target) and draft K/V
        k = torch.cat([k_ctx, k_draft], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_draft], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )

        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply rotary positional embeddings
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Update cache if provided
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # Compute attention
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        # Return output and K/V for MSE loss computation
        output_kv = (k.transpose(1, 2), v.transpose(1, 2))
        return attn_output, output_kv


class Qwen3SharedDraftDecoderLayer(GradientCheckpointingLayer):
    """A single transformer decoder layer for the shared backend draft model."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3SharedDraftAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_kv: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Tuple[torch.FloatTensor, torch.FloatTensor]]:
        """Forward pass through the decoder layer.

        Returns:
            Tuple of (hidden_states, (K, V)) where K/V are for MSE loss computation
        """
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self-attention with bi-directional pattern
        hidden_states, kv = self.self_attn(
            hidden_states=hidden_states,
            target_kv=target_kv,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, kv


class Qwen3SharedDraftModel(Qwen3PreTrainedModel):
    """Single-model speculative decoding with shared backend.

    This model uses a frozen pretrained Qwen3 backbone and adds trainable
    draft layers with block bi-directional attention. Each draft layer conditions
    on the key-value pairs from a specific layer in the target model.

    Architecture:
        - Frozen target model (Qwen3-8B or similar)
        - 5 trainable draft layers with bi-directional attention
        - Target layer mapping: [0, 6, 13, 20, 26] for a 32-layer model
        - Shared LM head with the target model

    Args:
        config: Qwen3 configuration
        num_draft_layers: Number of trainable draft layers (default: 5)
        block_size: Block size for parallel training (default: 16)
    """

    config_class = Qwen3Config
    _no_split_modules = ["Qwen3SharedDraftDecoderLayer"]

    def __init__(
        self,
        config: Qwen3Config,
        num_draft_layers: int = 5,
        block_size: int = 16,
    ):
        super().__init__(config)
        self.config = config
        self.num_draft_layers = num_draft_layers
        self.block_size = block_size

        # Target model (set externally via set_target_model)
        self.target_model = None
        self.target_layer_ids = []

        # Trainable draft layers
        self.layers = nn.ModuleList([
            Qwen3SharedDraftDecoderLayer(config, layer_idx=idx)
            for idx in range(num_draft_layers)
        ])

        # Layer norm for the final output
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Rotary embeddings
        self.rotary_emb = Qwen3RotaryEmbedding(config)

        # Post initialization
        self.post_init()

    def set_target_model(self, target_model: nn.Module) -> None:
        """Set and freeze the target model.

        Args:
            target_model: Pretrained Qwen3 model to use as the frozen backbone
        """
        self.target_model = target_model
        self.freeze_target_backbone()

        # Compute target layer IDs for KV extraction
        num_target_layers = len(target_model.model.layers)
        self.target_layer_ids = build_target_layer_ids(
            num_target_layers, self.num_draft_layers
        )

    def freeze_target_backbone(self) -> None:
        """Freeze all target model parameters."""
        if self.target_model is None:
            return
        for param in self.target_model.parameters():
            param.requires_grad = False

    def get_target_layer_ids(self) -> list[int]:
        """Get the target layer IDs used for KV extraction."""
        return self.target_layer_ids

    def extract_target_kv(
        self,
        target_hidden_states: torch.Tensor,
        target_layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract K/V from a target layer.

        This method extracts key and value projections from a specific target layer
        for use as context in the draft layers.

        Args:
            target_hidden_states: Hidden states from the target layer
            target_layer_idx: Index of the target layer

        Returns:
            Tuple of (K, V) tensors from the target layer
        """
        target_layer = self.target_model.model.layers[target_layer_idx]

        # Project to K and V
        batch_size, seq_len, hidden_size = target_hidden_states.shape
        k = target_layer.self_attn.k_proj(target_hidden_states)
        v = target_layer.self_attn.v_proj(target_hidden_states)

        num_key_value_heads = self.config.num_key_value_heads
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        k = k.view(batch_size, seq_len, num_key_value_heads, head_dim)
        v = v.view(batch_size, seq_len, num_key_value_heads, head_dim)

        return k, v

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward pass through the shared backend model.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask
            position_ids: Position IDs
            past_key_values: Past key values for caching
            use_cache: Whether to use caching
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states

        Returns:
            CausalLMOutputWithPast containing:
                - draft_logits: Logits from the draft model
                - target_logits: Logits from the target model (frozen)
                - draft_kvs: List of (K, V) tuples from each draft layer
                - target_kvs: List of (K, V) tuples from corresponding target layers
        """
        if self.target_model is None:
            raise ValueError(
                "Target model not set. Please call set_target_model() first."
            )

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=device
            ).unsqueeze(0).expand(batch_size, -1)

        # Run target model to get all hidden states
        with torch.no_grad():
            target_outputs = self.target_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        # Get embeddings from target model (layer 0)
        hidden_states = target_outputs.hidden_states[0]

        # Extract K/V from target layers for each draft layer
        target_kvs = []
        for layer_id in self.target_layer_ids:
            target_hidden = target_outputs.hidden_states[layer_id + 1]  # +1 for layer output
            kv = self.extract_target_kv(target_hidden, layer_id)
            target_kvs.append(kv)

        # Compute position embeddings for draft layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Run through draft layers
        draft_kvs = []
        for i, layer in enumerate(self.layers):
            hidden_states, kv = layer(
                hidden_states=hidden_states,
                target_kv=target_kvs[i],
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            draft_kvs.append(kv)

        # Apply final norm
        hidden_states = self.norm(hidden_states)

        # Generate draft logits using SHARED LM head from target model
        draft_logits = self.target_model.lm_head(hidden_states)

        return CausalLMOutputWithPast(
            loss=None,
            logits=draft_logits,
            past_key_values=past_key_values,
            hidden_states=target_outputs.hidden_states,
            attentions=None,
        )

    def _extract_kv_from_layer(
        self,
        target_outputs,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract K/V from a specific target layer.

        This is a convenience method for external access to target layer K/V.
        """
        target_hidden = target_outputs.hidden_states[layer_idx + 1]
        return self.extract_target_kv(target_hidden, layer_idx)
