"""Context-then-draft attention execution, shared by projection variants."""

from typing import Callable, Optional

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    Qwen3Config,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Unpack

from ..kernels import DFlashKernels
from .backends.eager import prepare_eager_mask as _prepare_dflash_eager_mask
from .backends.flash import flash_attention_forward
from .backends.flex import compile_friendly_flex_attention
from .backends.flex_options import flex_attention_backend
from .backends.sdpa import sdpa_attention_forward
from .masks import resolve_dflash_is_causal

SLIDING_ATTENTION = "sliding_attention"


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _rotate_half_interleaved(x: torch.Tensor) -> torch.Tensor:
    """Rotate consecutive pairs, the DeepSeek-style MLA RoPE convention."""

    paired = x.reshape(*x.shape[:-1], -1, 2)
    first, second = paired.unbind(dim=-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def apply_mla_rope(
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


class Qwen3DFlashAttentionBase(nn.Module):
    """Shared scaffold for the DFlash-family attention modes.

    Subclasses own only the projection parameterization: ``_init_projections``
    builds the weights and must define ``scaling``, ``num_key_value_groups``,
    and ``o_proj`` (the shared forward relies on them); ``_compute_qkv``
    returns rotated ``(q, k, v)`` in ``(batch, heads, seq, dim)`` layout with
    keys ordered context-then-draft. Everything the modes must agree on —
    KV-cache updates, backend dispatch, fully-masked-query zeroing, and the
    output projection — lives here so it is maintained in exactly one place.
    """

    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        kernels: DFlashKernels,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        if config._attn_implementation == "flex_attention":
            assert (
                config.attention_dropout == 0.0
            ), "DFlash FlexAttention requires attention_dropout=0.0"
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == SLIDING_ATTENTION
            else None
        )
        # Read by mask-free backends (flash_attention_2) and SDPA without a mask.
        self.is_causal = resolve_dflash_is_causal(
            getattr(config, "is_causal", None), config.layer_types[layer_idx]
        )
        self._init_projections(config, kernels)
        for attribute in ("scaling", "num_key_value_groups", "o_proj"):
            assert hasattr(
                self, attribute
            ), f"_init_projections must define {attribute}"

    def _init_projections(self, config: Qwen3Config, kernels: DFlashKernels) -> None:
        raise NotImplementedError

    def _compute_qkv(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

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
        q, k, v = self._compute_qkv(hidden_states, target_hidden, position_embeddings)
        if past_key_values is not None:
            cos, sin = position_embeddings
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        valid_queries = None
        if self.config._attn_implementation == "flex_attention":
            kernel_options = dict(kwargs.pop("kernel_options", None) or {})
            backend = flex_attention_backend()
            if backend is not None:
                kernel_options["BACKEND"] = backend

            attn_output = compile_friendly_flex_attention(
                q,
                k,
                v,
                block_mask=attention_mask,
                enable_gqa=True,
                scale=self.scaling,
                kernel_options=kernel_options or None,
            ).transpose(1, 2)
            attn_weights = None
        else:
            attn_fn: Callable = eager_attention_forward
            if self.config._attn_implementation == "eager":
                attention_mask, valid_queries = _prepare_dflash_eager_mask(
                    attention_mask,
                    q.dtype,
                )
            elif self.config._attn_implementation == "sdpa":
                attn_fn = sdpa_attention_forward
            elif self.config._attn_implementation.startswith("flash_attention"):
                attn_fn = flash_attention_forward
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
