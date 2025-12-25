import math
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from transformers import LlamaConfig
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.models.llama.configuration_llama import LlamaConfig
from yunchang import EXTRACT_FUNC_DICT
from yunchang.comm import SeqAllToAll4D

from specforge.modeling.draft.flex_attention import (
    compile_friendly_create_block_mask,
    compile_friendly_flex_attention,
    generate_eagle3_mask,
)
from specforge.utils import print_with_rank

from ...distributed import get_sp_ring_group, get_sp_ulysses_group
from .base import Eagle3DraftModel


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@torch.compile(dynamic=True)
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat(
        [m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)
    sin = torch.cat(
        [m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1
    ).unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def prepare_decoder_attention_mask(
    attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape,
            inputs_embeds.dtype,
            device=inputs_embeds.device,
            past_key_values_length=past_key_values_length,
        )

    if attention_mask is not None:
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        expanded_attn_mask = _expand_mask(
            attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
        ).to(inputs_embeds.device)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask + combined_attention_mask
        )

    return combined_attention_mask


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=None,
        low_freq_factor=None,
        high_freq_factor=None,
        orig_max_position=None,
    ):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        # Llama3 style rotary embedding frequency scaling
        if all(
            v is not None
            for v in [
                scaling_factor,
                low_freq_factor,
                high_freq_factor,
                orig_max_position,
            ]
        ):
            print_with_rank(
                f"Using Llama3 style rotary embedding with scaling_factor={scaling_factor}, low_freq_factor={low_freq_factor}, high_freq_factor={high_freq_factor}, orig_max_position={orig_max_position}"
            )
            self.scaling_factor = scaling_factor
            self.low_freq_factor = low_freq_factor
            self.high_freq_factor = high_freq_factor
            self.orig_max_position = orig_max_position

            low_freq_wavelen = orig_max_position / low_freq_factor
            high_freq_wavelen = orig_max_position / high_freq_factor
            wave_len = 2 * math.pi / inv_freq

            if low_freq_factor != high_freq_factor:
                smooth = (orig_max_position / wave_len - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
            else:
                smooth = 0

            new_freqs = torch.where(
                wave_len < high_freq_wavelen,
                inv_freq,
                torch.where(
                    wave_len > low_freq_wavelen,
                    inv_freq / self.scaling_factor,
                    (1 - smooth) * inv_freq / self.scaling_factor + smooth * inv_freq,
                ),
            )
            inv_freq = new_freqs

        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings + 20,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    @torch.compile(dynamic=True)
    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaMutiRotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__(dim, max_position_embeddings, base, device)
        self.scaling_factor = scaling_factor

    def forward(self, x, position_ids):
        # In contrast to other models, Qwen2_5_VL has different position ids for the grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[
            :, :, None, :
        ].float()  # shape (3, bs, 1, positions)

        device_type = (
            x.device.type
            if isinstance(x.device.type, str) and x.device.type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.scaling_factor
            sin = emb.sin() * self.scaling_factor

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
    num_rotations, dim, base=10000, max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def yarn_find_correction_range(
    low_rot, high_rot, dim, base=10000, max_position_embeddings=2048
):
    low = math.floor(
        yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min_val, max_val, dim):
    if min_val == max_val:
        max_val += 0.001  # Prevent singularity
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (
        max_val - min_val
    )
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


class LlamaYarnRotaryEmbedding(LlamaRotaryEmbedding):

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=1,
        mscale_all_dim=0,
    ):
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        dim = self.dim

        freq_extra = 1.0 / (
            self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        freq_inter = 1.0 / (
            self.scaling_factor
            * self.base
            ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            self.base,
            self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2).to(
            device=device, dtype=torch.float32
        )
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached",
            (emb.cos() * _mscale)[None, None, :, :].to(dtype),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            (emb.sin() * _mscale)[None, None, :, :].to(dtype),
            persistent=False,
        )


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

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

            if scaling_type == "linear":
                if scaling_factor is None:
                    raise ValueError(
                        "Linear RoPE scaling requires 'factor' in rope_scaling config."
                    )
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                if scaling_factor is None:
                    raise ValueError(
                        "Dynamic RoPE scaling requires 'factor' in rope_scaling config."
                    )
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "llama3":
                # for nv type
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
                    self.head_dim, max_position_embeddings=self.max_position_embeddings
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

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

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
                    query_states, key_states, cos, sin, position_ids
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

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)

        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaUSPAttention(LlamaAttention):
    def __init__(self, config):
        super().__init__(config)
        assert (
            dist.is_initialized()
        ), f"LlamaUSPAttention requires torch.distributed; call init_distributed first."
        self.rank = torch.distributed.get_rank()
        self.ring_pg = get_sp_ring_group()
        self.ulysses_pg = get_sp_ulysses_group()
        self.sp_ring_degree = torch.distributed.get_world_size(self.ring_pg)
        self.sp_ulysses_degree = torch.distributed.get_world_size(self.ulysses_pg)
        self.world_size = torch.distributed.get_world_size()
        self.ring_group_ranks = dist.get_process_group_ranks(self.ring_pg)
        self.ring_rank = torch.distributed.get_rank(self.ring_pg)

        if self.sp_ring_degree == 1:
            self.attention_impl = self.attention_with_cache
        else:
            self.attention_impl = self.ring_attention_hybrid_masked
        self.extract_func = EXTRACT_FUNC_DICT["basic"]
        self.scatter_idx = 2
        self.gather_idx = 1
        self.use_sync = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()  # bs, seq_len, hidden_size
        local_q_len = q_len
        query_states = self.q_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        query_states = SeqAllToAll4D.apply(
            self.ulysses_pg,
            query_states,
            self.scatter_idx,
            self.gather_idx,
            self.use_sync,
        ).transpose(1, 2)

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
        ).transpose(1, 2)

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
        ).transpose(1, 2)
        q_len = q_len * self.sp_ring_degree * self.sp_ulysses_degree

        if self.sp_ring_degree > 1:
            if isinstance(self.rotary_emb, LlamaMutiRotaryEmbedding):
                # mRoPE: [3, bs, seq_len] -> split dim 2
                position_ids = position_ids.chunk(self.sp_ring_degree, dim=2)[
                    self.ring_rank
                ].clone()
            else:
                # Standard RoPE: [bs, seq_len] -> split dim 1
                position_ids = position_ids.chunk(self.sp_ring_degree, dim=1)[
                    self.ring_rank
                ].clone()

        # bs, shard_seqlen, hc, hs
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

            cache_k = [key_states]
            cache_v = [value_states]
            attn_output = self.attention_impl(
                query_states, attention_mask, cache_k, cache_v, q_len
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
                    query_states, key_states, cos, sin, position_ids
                )

            cache_hidden[0] = cache_hidden[0] + [key_states]
            cache_hidden[1] = cache_hidden[1] + [value_states]

            cache_k = cache_hidden[0]  # bs, head_num, seq_len, dim
            cache_v = cache_hidden[1]  # bs, head_num, seq_len, dim
            attn_output = self.attention_impl(
                query_states, attention_mask, cache_k, cache_v, q_len
            )

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

    def ring_attention_hybrid_masked(
        self,
        local_q: torch.Tensor,  # [1, H, Local_S, D]
        attention_mask: torch.Tensor,  # [1, 1, Global_S, Global_S] (global Mask)
        cache_k: List[torch.Tensor],
        cache_v: List[torch.Tensor],
        q_len: int,  # Global_S
    ):
        group = self.ring_pg
        rank = self.ring_rank
        world_size = self.sp_ring_degree

        # bs=1
        _, H, local_seq_len, head_dim = local_q.shape
        scale = 1.0 / math.sqrt(head_dim)

        # global Q index
        start_q_idx = rank * local_seq_len
        end_q_idx = start_q_idx + local_seq_len

        # =============================================================
        # [FP32] 1. init Online Softmax static to float32
        # =============================================================
        m = torch.full(
            (1, H, local_seq_len),
            float("-inf"),
            device=local_q.device,
            dtype=torch.float32,
        )
        l = torch.zeros(
            (1, H, local_seq_len), device=local_q.device, dtype=torch.float32
        )
        acc = torch.zeros(
            (1, H, local_seq_len, head_dim), device=local_q.device, dtype=torch.float32
        )

        # =============================================================
        # Phase 1: Extras (cache_k[1:]) - local compute
        # =============================================================
        num_extras = len(cache_k) - 1
        if num_extras > 0:
            for i in range(1, len(cache_k)):
                ki = repeat_kv(cache_k[i], self.num_key_value_groups)  # bf16
                vi = repeat_kv(cache_v[i], self.num_key_value_groups)  # bf16

                score_i = (local_q * ki).sum(dim=-1).to(torch.float32) * scale

                # [FP32] Online Softmax Update
                m_new = torch.maximum(m, score_i)
                alpha = torch.exp(m - m_new)
                p_i = torch.exp(score_i - m_new)  # fp32

                l_new = l * alpha + p_i

                # [FP32] Accumulation
                # acc(fp32) = acc * alpha + p_i * vi(bf16->fp32)
                acc = acc * alpha.unsqueeze(-1)
                acc += p_i.unsqueeze(-1) * vi.to(torch.float32)

                m, l = m_new, l_new

        # =============================================================
        # Phase 2: Main Sequence (cache_k[0]) - Ring Attention
        # =============================================================
        local_k0 = repeat_kv(cache_k[0], self.num_key_value_groups)
        local_v0 = repeat_kv(cache_v[0], self.num_key_value_groups)

        curr_k, curr_v = local_k0, local_v0
        next_k, next_v = torch.empty_like(local_k0), torch.empty_like(local_v0)

        for step in range(world_size):
            # 2.1 communicate
            if step < world_size - 1:
                # ring group rank to global rank for isend and irecv
                send_dst = self.ring_group_ranks[(rank + 1) % world_size]
                recv_src = self.ring_group_ranks[(rank - 1 + world_size) % world_size]
                reqs = []

                if rank % 2 == 0:
                    reqs.append(dist.isend(curr_k, dst=send_dst, group=group))
                    reqs.append(dist.isend(curr_v, dst=send_dst, group=group))
                    reqs.append(dist.irecv(next_k, src=recv_src, group=group))
                    reqs.append(dist.irecv(next_v, src=recv_src, group=group))
                else:
                    reqs.append(dist.irecv(next_k, src=recv_src, group=group))
                    reqs.append(dist.irecv(next_v, src=recv_src, group=group))
                    reqs.append(dist.isend(curr_k, dst=send_dst, group=group))
                    reqs.append(dist.isend(curr_v, dst=send_dst, group=group))

            # 2.2 get index
            block_rank = (rank - step + world_size) % world_size
            start_k_idx = block_rank * local_seq_len
            end_k_idx = start_k_idx + local_seq_len

            # 2.3 [FP32] MatMul Score
            attn_block = torch.matmul(local_q, curr_k.transpose(2, 3)) * scale

            # 2.4 [FP32] Mask slice
            mask_slice = attention_mask[
                :, :, start_q_idx:end_q_idx, start_k_idx:end_k_idx
            ]
            if mask_slice.dtype != torch.float32:
                mask_slice = mask_slice.to(torch.float32)
            if mask_slice.device != attn_block.device:
                mask_slice = mask_slice.to(attn_block.device)

            attn_block = attn_block + mask_slice

            # 2.5 [FP32] Online Softmax Update
            m_block = attn_block.max(dim=-1).values  # fp32
            m_new = torch.maximum(m, m_block)

            alpha = torch.exp(m - m_new)
            p_block = torch.exp(attn_block - m_new.unsqueeze(-1))  # fp32

            l_new = l * alpha + p_block.sum(dim=-1)

            acc = acc * alpha.unsqueeze(-1)
            acc += torch.matmul(p_block, curr_v.to(torch.float32))

            m, l = m_new, l_new

            if step < world_size - 1:
                for req in reqs:
                    req.wait()
                curr_k, curr_v = next_k.clone(), next_v.clone()

        # =============================================================
        # 3. Finalize
        # =============================================================
        final_output = acc / (l.unsqueeze(-1) + 1e-6)

        return final_output.to(local_q.dtype).transpose(1, 2).contiguous()

    def attention_with_cache(
        self,
        query_states: torch.Tensor,
        attention_mask,
        cache_k,
        cache_v,
        q_len,
    ):
        k0 = repeat_kv(cache_k[0], self.num_key_value_groups)
        v0 = repeat_kv(cache_v[0], self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, k0.transpose(2, 3)) / math.sqrt(
            self.head_dim
        )
        lck = len(cache_k)

        attn_weights = attn_weights + attention_mask

        for i in range(1, lck):
            ki = repeat_kv(cache_k[i], self.num_key_value_groups)
            qi = query_states
            kiq = ki

            attn_weightsi = (qi * kiq).sum(-1) / math.sqrt(self.head_dim)
            attn_weights = torch.cat((attn_weights, attn_weightsi[..., None]), dim=-1)

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_weights0 = attn_weights[..., :q_len]

        attn_output = torch.matmul(attn_weights0, v0)

        for i in range(1, lck):
            vi = repeat_kv(cache_v[i], self.num_key_value_groups)
            attn_weightsi = attn_weights[..., q_len + i - 1]
            attn_outputi = attn_weightsi[..., None] * vi
            attn_output = attn_output + attn_outputi

        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output


class LlamaFlexAttention(LlamaAttention):
    """
    Attention layer implemented with flex attention. We keep the parameters consistent with LlamaAttention.
    The used parameters are:
        - hidden_states: input hidden states
        - attention_mask: attention mask not expanded, straight from data loader.
        - position_ids: position ids
        - past_key_values: dynamic cache used for storing past key and value states.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_hidden: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )

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
            # Keep positions ids aligned when padding so the KV cache is unaffected.
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
            layer_idx=0,  # TODO: support multiple layers
            cache_kwargs=cache_kwargs,
        )

        seq_lengths = attention_mask.sum(dim=-1)
        # Shrink the attention mask to align with the padding to the right.
        # This is equivalent to the shrinking logic in eagle3.py
        seq_lengths -= lck
        # TODO: Remove the usage of uncompiled create_block_mask after
        # https://github.com/pytorch/pytorch/issues/160018
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
            H=1,  # Rely on broadcast
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
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.head_dim * self.num_heads)
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [
                    F.linear(x, gate_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            up_proj = torch.cat(
                [
                    F.linear(x, up_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    @torch.compile(dynamic=True)
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, attention_backend: str = "sdpa"):
        super().__init__()
        self.hidden_size = config.hidden_size

        if attention_backend == "sdpa":
            self.self_attn = LlamaAttention(config=config)
        elif attention_backend == "usp":
            self.self_attn = LlamaUSPAttention(config=config)
        elif attention_backend == "flex_attention":
            print_with_rank("Using flex attention on draft model training!")
            self.self_attn = LlamaFlexAttention(config=config)
        else:
            raise ValueError(f"Unknown attention backend {attention_backend}")

        self.attention_backend = attention_backend
        self.mlp = LlamaMLP(config)
        # self.fc = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # if self.index!=0:

        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: List[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_values (`Cache`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
        # Self Attention
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

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # outputs = (hidden_states, return_hidden)
        return hidden_states


class LlamaForCausalLMEagle3(Eagle3DraftModel):

    config_class = LlamaConfig

    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        super().__init__(config)
        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.midlayer = LlamaDecoderLayer(config, attention_backend=attention_backend)

        if hasattr(config, "target_hidden_size"):
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

        # create vocab buffers
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
        """
        Arguments:
            hidden_states (`torch.FloatTensor`): input to the layer, cat low, mid high hidden_states of shape `(batch, seq_len, hidden_states * 3)`
            input_ids (`torch.LongTensor`): input ids of shape `(batch, seq_len)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor`, *optional*): position ids of shape `(batch, seq_len)`
        """
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

        # fc
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

        # norm
        hidden_states = self.norm(hidden_states)

        return hidden_states

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def project_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # eagle 3 requires hidden states from 3 layers
        assert hidden_states.size(-1) == self.config.hidden_size * 3
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
        past_key_values: Optional[Cache] = None,
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
