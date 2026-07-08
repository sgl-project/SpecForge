# coding=utf-8
"""Multi-Token Prediction (MTP) draft model for Qwen3.5.

Architecture follows the Qwen3.5 MTP design:
  1. Normalize input embeddings and target last hidden states separately.
  2. Concatenate and project via fc( [norm(emb); norm(hidden)] ).
  3. Run a 1-layer Qwen3 transformer.
  4. Compute logits with a (shared) lm_head.

Weight key layout matches SGLang's Qwen3_5ForCausalLMMTP:
  mtp.pre_fc_norm_embedding.weight
  mtp.pre_fc_norm_hidden.weight
  mtp.fc.weight
  mtp.model.layers.0.self_attn.q_proj.weight
  mtp.model.layers.0.mlp.gate_proj.weight
  mtp.lm_head.weight
"""

import copy
from typing import Optional, Tuple, Unpack

import torch
from torch import nn
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

from specforge.modeling._mask_utils import _expand_mask, _make_causal_mask


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary positional embeddings to q/k tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3MTPAttention(nn.Module):
    """Standard causal self-attention used inside the MTP transformer."""

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
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.config.num_attention_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.config.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.config.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attn_fn = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attn_fn(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3MTPDecoderLayer(GradientCheckpointingLayer):
    """A single Qwen3-style decoder layer for the MTP draft model."""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3MTPAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class Qwen3_5MTPModel(nn.Module):
    """The core MTP module wrapped under the ``mtp.`` prefix.

    Weight key layout (flat, matches the native Qwen3.5 checkpoint and
    SGLang's ``Qwen3_5ForCausalLMMTP.load_weights`` remap):
      mtp.pre_fc_norm_embedding.weight
      mtp.pre_fc_norm_hidden.weight
      mtp.fc.weight
      mtp.layers.0.self_attn.q_proj.weight
      mtp.layers.0.mlp.gate_proj.weight
      mtp.norm.weight
      mtp.lm_head.weight

    SGLang's ``Qwen3_5ForCausalLMMTP`` wraps a *flat* ``Qwen3_5ForCausalLM``
    (``self.layers`` directly on the ForCausalLM, not nested under
    ``self.model``), so after the ``mtp.`` -> ``model.`` remap the flat keys
    ``mtp.layers.0.*`` / ``mtp.norm.weight`` become ``model.layers.0.*`` /
    ``model.norm.weight``, matching ``self.model.layers`` / ``self.model.norm``.
    """

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config

        # Fusion projection: fc( concat( norm(input_embeds), norm(target_hidden) ) )
        self.pre_fc_norm_embedding = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)

        # Single-layer Qwen3 transformer, flat under `mtp.layers.*` / `mtp.norm`
        # to match the native Qwen3.5 checkpoint layout consumed by SGLang.
        mtp_config = copy.deepcopy(config)
        mtp_config.num_hidden_layers = 1
        self.layers = nn.ModuleList(
            [Qwen3MTPDecoderLayer(mtp_config, layer_idx=0)]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(mtp_config)

        # LM head (shared with target model during training)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # Fusion
        normed_emb = self.pre_fc_norm_embedding(inputs_embeds)
        normed_hidden = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = self.fc(torch.cat([normed_emb, normed_hidden], dim=-1))

        bsz, seq_len, _ = hidden_states.size()
        if position_ids is None:
            device = hidden_states.device
            position_ids = (
                torch.arange(seq_len, dtype=torch.long, device=device)
                .unsqueeze(0)
                .expand(bsz, -1)
            )

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Causal mask
        if attention_mask is not None and attention_mask.dim() == 2:
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            combined_mask = _make_causal_mask(
                (bsz, seq_len), hidden_states.dtype, device=hidden_states.device
            )
            expanded_mask = _expand_mask(
                attention_mask, hidden_states.dtype, tgt_len=seq_len
            ).to(hidden_states.device)
            attention_mask = expanded_mask + combined_mask
        elif attention_mask is None and seq_len > 1:
            attention_mask = _make_causal_mask(
                (bsz, seq_len), hidden_states.dtype, device=hidden_states.device
            )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )[0]

        hidden_states = self.norm(hidden_states)
        return hidden_states


class Qwen3_5MTPDraftModel(Qwen3PreTrainedModel):
    """
    Qwen3.5 MTP draft model for SpecForge training.

    The embed_tokens table is loaded from the target model and frozen by default;
    the lm_head is optionally shared with the target model.  All trainable MTP
    parameters live under the `mtp.*` prefix so that checkpoints can be loaded
    directly by SGLang's Qwen3_5ForCausalLMMTP.
    """

    config_class = Qwen3Config
    _no_split_modules = ["Qwen3MTPDecoderLayer"]

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)
        self.config = config

        # Shared embedding with the target model (loaded externally, frozen)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )

        # All trainable MTP weights under `mtp.*`
        self.mtp = Qwen3_5MTPModel(config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.mtp.lm_head

    def set_output_embeddings(self, value):
        self.mtp.lm_head = value

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = self.mtp(
            inputs_embeds=inputs_embeds,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        logits = self.mtp.lm_head(hidden_states)
        return CausalLMOutputWithPast(logits=logits)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: Optional[list[int]] = None,
        temperature: float = 0.0,
    ) -> torch.LongTensor:
        """Sequential MTP speculative generation (single MTP layer)."""
        self.eval()
        device = input_ids.device
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens
        output_ids = input_ids.clone()

        from transformers.cache_utils import DynamicCache

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # Prefill target once to get initial last hidden state
        target_out = target(
            input_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        next_token_logits = target_out.logits[:, -1, :]
        if temperature < 1e-5:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        else:
            next_token = torch.multinomial(
                torch.softmax(next_token_logits / temperature, dim=-1), num_samples=1
            )
        output_ids = torch.cat([output_ids, next_token], dim=1)
        target_hidden = target_out.hidden_states[-1][:, -1:, :]

        while output_ids.shape[1] < max_length:
            draft_input_ids = output_ids[:, -1:]
            draft_position_ids = torch.tensor(
                [[output_ids.shape[1] - 1]], dtype=torch.long, device=device
            )
            draft_embeds = self.embed_tokens(draft_input_ids)
            draft_hidden = self.mtp(
                inputs_embeds=draft_embeds,
                hidden_states=target_hidden,
                position_ids=draft_position_ids,
            )
            draft_logits = self.mtp.lm_head(draft_hidden)
            if temperature < 1e-5:
                draft_token = torch.argmax(draft_logits[:, -1, :], dim=-1, keepdim=True)
            else:
                draft_token = torch.multinomial(
                    torch.softmax(draft_logits[:, -1, :] / temperature, dim=-1),
                    num_samples=1,
                )

            # Verify against target
            verify_input_ids = torch.cat([output_ids[:, -1:], draft_token], dim=1)
            verify_position_ids = torch.arange(
                output_ids.shape[1] - 1,
                output_ids.shape[1] + 1,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0)
            target_out = target(
                verify_input_ids,
                position_ids=verify_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )
            target_hidden = target_out.hidden_states[-1][:, -1:, :]
            target_token = torch.argmax(target_out.logits[:, -1, :], dim=-1, keepdim=True)

            if torch.equal(draft_token, target_token):
                output_ids = torch.cat([output_ids, draft_token], dim=1)
            else:
                output_ids = torch.cat([output_ids, target_token], dim=1)

            if (
                stop_token_ids is not None
                and output_ids[0, -1].item() in stop_token_ids
            ):
                break

        return output_ids


