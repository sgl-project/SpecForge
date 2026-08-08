import copy
from typing import Callable, Optional

import torch
from torch import nn
from transformers import DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack

from .dflash_kernels import DEFAULT_DFLASH_KERNELS, DFlashKernels
from .registry import register_draft

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"
_VALID_DFLASH_LAYER_TYPES = {FULL_ATTENTION, SLIDING_ATTENTION}
_VALID_DFLASH_ATTENTION_MODES = {"gqa", "mha", "mla"}


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def resolve_dflash_attention_layout(
    config: Qwen3Config,
) -> tuple[tuple[str, ...], Optional[int]]:
    """Validate and return the configured per-layer DFlash attention layout."""

    num_hidden_layers = config.num_hidden_layers
    layer_types = tuple(config.layer_types)

    if len(layer_types) != num_hidden_layers:
        raise ValueError(
            "DFlash config.layer_types must contain exactly "
            f"num_hidden_layers={num_hidden_layers} entries, got "
            f"{len(layer_types)}"
        )
    invalid = set(layer_types) - _VALID_DFLASH_LAYER_TYPES
    if invalid:
        raise ValueError(
            "DFlash config.layer_types supports only full_attention and "
            f"sliding_attention, got {sorted(invalid)}"
        )

    if SLIDING_ATTENTION not in layer_types:
        return layer_types, None

    sliding_window = config.sliding_window
    if sliding_window is None or sliding_window <= 0:
        raise ValueError(
            "DFlash sliding_attention layers require use_sliding_window=true "
            "and a positive config.sliding_window"
        )
    return layer_types, sliding_window


def resolve_dflash_attention_mode(config: Qwen3Config) -> str:
    """Return the configured draft attention architecture.

    GQA and MHA share the existing Qwen attention implementation. MLA swaps
    only the attention projections while retaining the DFlash-family decoder,
    target-context injection, masks, and objective.
    """

    dflash_config = dict(getattr(config, "dflash_config", None) or {})
    attention_mode = str(dflash_config.get("attention_mode", "gqa")).lower()
    if attention_mode not in _VALID_DFLASH_ATTENTION_MODES:
        raise ValueError(
            "DFlash dflash_config.attention_mode must be one of "
            f"{sorted(_VALID_DFLASH_ATTENTION_MODES)}, got {attention_mode!r}"
        )
    return attention_mode


def validate_dflash_mla_config(config: Qwen3Config) -> None:
    """Validate the standard MLA dimensions carried by a draft config."""

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

    positive = ("kv_lora_rank", "qk_rope_head_dim", "v_head_dim")
    for name in positive:
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


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _prepare_dflash_eager_mask(
    attention_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Convert a boolean allow-mask to eager's additive representation."""

    if attention_mask is None or attention_mask.dtype != torch.bool:
        return attention_mask, None

    valid_queries = attention_mask.any(dim=-1, keepdim=True)
    # A finite minimum keeps eager softmax stable. Fully masked query rows are
    # explicitly zeroed after attention so they cannot average forbidden values.
    additive_mask = torch.zeros_like(attention_mask, dtype=dtype)
    additive_mask.masked_fill_(~attention_mask, torch.finfo(dtype).min)
    return additive_mask, valid_queries


class Qwen3DFlashAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ):
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
        self.is_causal = False
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
        self.q_norm = kernels.make_rms_norm(self.head_dim, config.rms_norm_eps)
        self.k_norm = kernels.make_rms_norm(self.head_dim, config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == SLIDING_ATTENTION
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, -1, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        valid_queries = None
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation == "eager":
            attention_mask, valid_queries = _prepare_dflash_eager_mask(
                attention_mask,
                q.dtype,
            )
        else:
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        if valid_queries is not None and attn_weights is not None:
            attn_weights = attn_weights.masked_fill(~valid_queries, 0)
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        if valid_queries is not None:
            attn_output = attn_output.masked_fill(
                ~valid_queries.any(dim=1),
                0,
            )
        return attn_output, attn_weights


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive pairs, as used by DeepSeek-style MLA RoPE."""

    paired = x.float().reshape(*x.shape[:-1], -1, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2).to(x.dtype)


def _apply_mla_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    interleaved: bool,
) -> torch.Tensor:
    if interleaved:
        half = cos.shape[-1] // 2
        cos = cos[..., :half].repeat_interleave(2, dim=-1)
        sin = sin[..., :half].repeat_interleave(2, dim=-1)
        rotated = _rotate_half_interleaved(x)
    else:
        rotated = rotate_half(x)
    return x * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


class Qwen3DFlashMLAAttention(nn.Module):
    """Config-selected Multi-head Latent Attention for DFlash-family drafts.

    The module uses the standard MLA parameterization: an optional low-rank Q
    path, a shared compressed KV latent, partial RoPE, and expanded per-head
    K/V projections for training. Expanding K/V makes the implementation work
    with the same eager, SDPA, and FlexAttention interfaces and masks as the
    existing GQA/MHA path while preserving MLA's parameterization.
    """

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ):
        super().__init__()
        validate_dflash_mla_config(config)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = int(config.hidden_size)
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
        self.attention_dropout = float(config.attention_dropout)
        self.is_causal = False

        dflash_config = dict(getattr(config, "dflash_config", None) or {})
        self.rope_interleaved = bool(
            dflash_config.get(
                "mla_rope_interleaved",
                getattr(config, "rope_interleave", True),
            )
        )
        self.use_output_gate = bool(
            dflash_config.get(
                "mla_use_output_gate",
                getattr(config, "mla_use_output_gate", False),
            )
        )

        bias = bool(config.attention_bias)
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.qk_head_dim,
                bias=bias,
            )
        else:
            self.q_a_proj = nn.Linear(
                self.hidden_size,
                self.q_lora_rank,
                bias=bias,
            )
            self.q_a_layernorm = kernels.make_rms_norm(
                self.q_lora_rank,
                config.rms_norm_eps,
            )
            self.q_b_proj = nn.Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=bias,
            )

        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size,
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
            bias=bias,
        )
        if self.use_output_gate:
            self.g_proj = nn.Linear(
                self.hidden_size,
                self.num_heads * self.v_head_dim,
                bias=False,
            )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=bias,
        )

        rope_config = copy.deepcopy(config)
        rope_config.head_dim = self.qk_rope_head_dim
        self.rotary_emb = Qwen3RotaryEmbedding(rope_config)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == SLIDING_ATTENTION
            else None
        )

    def _project_q(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.q_lora_rank is None:
            return self.q_proj(hidden_states)
        return self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))

    def _project_kv(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = hidden_states.shape[:2]
        kv_combined = self.kv_a_proj_with_mqa(hidden_states)
        kv_compressed, k_rope = kv_combined.split(
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        kv = self.kv_b_proj(self.kv_a_layernorm(kv_compressed)).view(
            batch_size,
            seq_len,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, value = kv.split(
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1,
        )
        return k_nope, k_rope, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        del position_embeddings
        if position_ids is None:
            raise ValueError("MLA attention requires position_ids")

        batch_size, query_len = hidden_states.shape[:2]
        context_len = target_hidden.shape[1]
        key_len = context_len + query_len
        if position_ids.shape[-1] != key_len:
            raise ValueError(
                "MLA position_ids must cover target context and draft queries; "
                f"expected {key_len}, got {position_ids.shape[-1]}"
            )

        query = self._project_q(hidden_states).view(
            batch_size,
            query_len,
            self.num_heads,
            self.qk_head_dim,
        )
        q_nope, q_rope = query.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        context_k_nope, context_k_rope, context_value = self._project_kv(target_hidden)
        noise_k_nope, noise_k_rope, noise_value = self._project_kv(hidden_states)
        k_nope = torch.cat((context_k_nope, noise_k_nope), dim=1)
        k_rope = torch.cat((context_k_rope, noise_k_rope), dim=1)
        value = torch.cat((context_value, noise_value), dim=1)

        cos, sin = self.rotary_emb(k_rope, position_ids)
        q_rope = _apply_mla_rope(
            q_rope.transpose(1, 2),
            cos[:, -query_len:],
            sin[:, -query_len:],
            interleaved=self.rope_interleaved,
        )
        k_rope = _apply_mla_rope(
            k_rope.unsqueeze(1),
            cos,
            sin,
            interleaved=self.rope_interleaved,
        ).expand(-1, self.num_heads, -1, -1)
        query = torch.cat((q_nope.transpose(1, 2), q_rope), dim=-1)
        key = torch.cat((k_nope.transpose(1, 2), k_rope), dim=-1)
        value = value.transpose(1, 2)

        if past_key_values is not None:
            cache_kwargs = {"cache_position": cache_position}
            key, value = past_key_values.update(
                key,
                value,
                self.layer_idx,
                cache_kwargs,
            )

        valid_queries = None
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation == "eager":
            attention_mask, valid_queries = _prepare_dflash_eager_mask(
                attention_mask,
                query.dtype,
            )
        else:
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            query,
            key,
            value,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        if valid_queries is not None and attn_weights is not None:
            attn_weights = attn_weights.masked_fill(~valid_queries, 0)

        if self.use_output_gate:
            gate = torch.sigmoid(self.g_proj(hidden_states)).view(
                batch_size,
                query_len,
                self.num_heads,
                self.v_head_dim,
            )
            attn_output = attn_output * gate
        attn_output = self.o_proj(attn_output.reshape(batch_size, query_len, -1))
        if valid_queries is not None:
            attn_output = attn_output.masked_fill(
                ~valid_queries.any(dim=1),
                0,
            )
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        attention_cls = (
            Qwen3DFlashMLAAttention
            if resolve_dflash_attention_mode(config) == "mla"
            else Qwen3DFlashAttention
        )
        self.self_attn = attention_cls(
            config=config,
            layer_idx=layer_idx,
            kernels=kernels,
        )
        self.mlp = kernels.make_mlp(config)
        self.input_layernorm = kernels.make_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.post_attention_layernorm = kernels.make_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    offset = 1
    selected_states = []
    for layer_id in layer_ids:
        selected_states.append(hidden_states[layer_id + offset])
    target_hidden = torch.cat(selected_states, dim=-1)
    return target_hidden


def normalize_draft_head_checkpoint_keys(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    """Map checkpoint-only nested head names onto the direct module layout.

    Early Domino/DSpark checkpoints saved their auxiliary heads beneath a
    ``logit_head`` container. The live architecture no longer owns that wrapper,
    but those tensors remain valid and must not be dropped during warm start or
    full resume.
    """

    del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    checkpoint_prefixes = (
        ("logit_head.prefix_gru.", "prefix_gru."),
        ("logit_head.embed_proj.", "embed_proj."),
        ("logit_head.markov_head.", "markov_head."),
        ("logit_head.confidence_head.", "confidence_head."),
    )
    for key in list(state_dict):
        if not key.startswith(prefix):
            continue
        local_key = key[len(prefix) :]
        for checkpoint_prefix, model_prefix in checkpoint_prefixes:
            if not local_key.startswith(checkpoint_prefix):
                continue
            normalized_key = prefix + model_prefix + local_key[len(checkpoint_prefix) :]
            if normalized_key not in state_dict:
                state_dict[normalized_key] = state_dict[key]
            state_dict.pop(key)
            break


@register_draft
class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(
        self,
        config,
        dflash_kernels: Optional[DFlashKernels] = None,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.layer_types, self.sliding_window = resolve_dflash_attention_layout(config)
        self.attention_mode = resolve_dflash_attention_mode(config)
        kernels = dflash_kernels or DEFAULT_DFLASH_KERNELS
        self.layers = nn.ModuleList(
            [
                Qwen3DFlashDecoderLayer(config, layer_idx, kernels)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        dflash_config = getattr(config, "dflash_config", {}) or {}
        self.target_layer_ids = dflash_config.get(
            "target_layer_ids",
            build_target_layer_ids(config.num_target_layers, config.num_hidden_layers),
        )
        self.norm = kernels.make_rms_norm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = kernels.make_rms_norm(
            config.hidden_size, config.rms_norm_eps
        )
        self.block_size = config.block_size
        self.mask_token_id = dflash_config.get("mask_token_id", None)
        self.projector_type = dflash_config.get("projector_type", None)
        self.pure_draft_prefix_len = dflash_config.get("pure_draft_prefix_len", 0)
        self.shift_label = dflash_config.get("shift_label", False)
        self._init_draft_head(config, dflash_config)
        self.register_load_state_dict_pre_hook(normalize_draft_head_checkpoint_keys)
        self.post_init()

    def _init_draft_head(self, config, dflash_config: dict) -> None:
        del config, dflash_config

    def apply_logits_head(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
        prev_token_embeddings: Optional[torch.Tensor] = None,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del prev_token_ids, prev_token_embeddings, hidden_states
        return base_logits

    def apply_markov_logits(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.apply_logits_head(
            base_logits,
            prev_token_ids=prev_token_ids,
            hidden_states=hidden_states,
        )

    def predict_confidence(
        self,
        hidden_states: torch.Tensor,
        *,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        del hidden_states, prev_token_ids
        return None

    def _sample_draft_tokens(
        self,
        target: nn.Module,
        draft_hidden: torch.Tensor,
        block_output_ids: torch.LongTensor,
    ) -> torch.LongTensor:
        """Sample one speculative block from the draft-model hidden states.

        DFlash predicts the whole suffix in one LM-head call. Draft families
        with an auxiliary logits head can override this boundary without
        duplicating the target-cache and acceptance logic in ``spec_generate``.
        """
        del block_output_ids
        draft_logits = target.lm_head(draft_hidden[:, -self.block_size + 1 :, :])
        return sample(draft_logits)

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[object] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = (
            None
            if self.attention_mode == "mla"
            else self.rotary_emb(hidden_states, position_ids)
        )
        for layer_type, layer in zip(self.layer_types, self.layers):
            layer_attention_mask = (
                attention_mask[layer_type]
                if isinstance(attention_mask, dict)
                else attention_mask
            )
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=layer_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
    ):
        self.eval()
        num_input_tokens = input_ids.shape[1]
        max_length = num_input_tokens + max_new_tokens

        block_size = self.block_size
        output_ids = torch.full(
            (1, max_length + block_size),
            self.mask_token_id,
            dtype=torch.long,
            device=target.device,
        )
        position_ids = torch.arange(
            output_ids.shape[1], device=target.device
        ).unsqueeze(0)

        past_key_values_target = DynamicCache()
        past_key_values_draft = DynamicCache()

        # Prefill stage
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        output_ids[:, :num_input_tokens] = input_ids
        output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(
            output.logits, temperature
        )
        target_hidden = extract_context_feature(
            output.hidden_states, self.target_layer_ids
        )

        # Decode stage
        acceptance_lengths = []
        start = input_ids.shape[1]
        while start < max_length:
            block_output_ids = output_ids[:, start : start + block_size].clone()
            block_position_ids = position_ids[:, start : start + block_size]
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_hidden = self(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[
                    :, past_key_values_draft.get_seq_length() : start + block_size
                ],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = self._sample_draft_tokens(
                target,
                draft_hidden,
                block_output_ids,
            )

            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=past_key_values_target,
                use_cache=True,
                output_hidden_states=True,
            )

            posterior = sample(output.logits, temperature)
            acceptance_length = (
                (block_output_ids[:, 1:] == posterior[:, :-1])
                .cumprod(dim=1)
                .sum(dim=1)[0]
                .item()
            )
            output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
                :, : acceptance_length + 1
            ]
            output_ids[:, start + acceptance_length + 1] = posterior[
                :, acceptance_length
            ]
            start += acceptance_length + 1
            past_key_values_target.crop(start)
            target_hidden = extract_context_feature(
                output.hidden_states, self.target_layer_ids
            )[:, : acceptance_length + 1, :]
            acceptance_lengths.append(acceptance_length + 1)
            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:]
                for stop_token_id in stop_token_ids
            ):
                break
        output_ids = output_ids[:, :max_length]
        output_ids = output_ids[:, output_ids[0] != self.mask_token_id]
        if stop_token_ids is not None:
            stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
            stop_token_indices = torch.isin(
                output_ids[0][num_input_tokens:], stop_token_ids
            ).nonzero(as_tuple=True)[0]
            if stop_token_indices.numel() > 0:
                output_ids = output_ids[
                    :, : num_input_tokens + stop_token_indices[0] + 1
                ]

        return output_ids
