import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange
from transformers import GenerationMixin, LlamaConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.models.llama.configuration_llama import LlamaConfig

try:
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward, _flash_attn_varlen_backward
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False

from ..utils import padding
from .base import Eagle3DraftModel

logger = logging.getLogger(__name__)


class FlashAttnVarlenFunc(torch.autograd.Function):
    """
    Flash Attention with KV Cache Support
    """
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        kv_cache,
        dropout_p,
        causal,
        softmax_scale,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        batch_size = q.shape[0]
        
        # Flash Attention only supports fp16 and bf16
        original_dtype = q.dtype
        if original_dtype not in [torch.float16, torch.bfloat16]:
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
        
        if 'k_cache' in kv_cache:
            k_cache, v_cache = kv_cache['k_cache'], kv_cache['v_cache']
            # Ensure cache tensors have the same dtype as current tensors
            k_cache = k_cache.to(k.dtype)
            v_cache = v_cache.to(v.dtype)
            offset = k_cache.shape[1]
            
            k_whole = torch.cat([k_cache, k], dim=1).contiguous()
            v_whole = torch.cat([v_cache, v], dim=1).contiguous()
        else:
            offset = 0
            k_whole = k
            v_whole = v
        
        kv_cache['k_cache'], kv_cache['v_cache'] = k_whole, v_whole 
        
        seqlen_k = k_whole.shape[1]
        seqlen_q = q.shape[1]
        ctx._seqlen_k = seqlen_k
        ctx._seqlen_q = seqlen_q
        ctx._offset = offset
        
        # Convert to Flash Attention format
        q, k_whole, v_whole = [rearrange(x, 'b s ... -> (b s) ...').contiguous() for x in [q, k_whole, v_whole]]
        
        # Save tensors for backward pass BEFORE Flash Attention modifies them
        # Flash Attention modifies input tensors in-place, so we need copies
        # detach() prevents these tensors from participating in autograd (we handle gradients manually)
        q_for_backward = q.detach().clone()
        k_for_backward = k_whole.detach().clone() 
        v_for_backward = v_whole.detach().clone()

        # q_for_backward = q
        # k_for_backward = k
        # v_for_backward = v
            
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device="cuda")
        cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device="cuda")

        q = q.contiguous()
        k_whole = k_whole.contiguous()
        v_whole = v_whole.contiguous()
        
        # Flash attention API call
        try:
            # Try with newer API that includes window_size parameters
            flash_result = _flash_attn_varlen_forward(
                q,
                k_whole,
                v_whole,
                cu_seqlens_q,
                cu_seqlens_k,
                seqlen_q,
                seqlen_k,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size_left=-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                return_softmax=False,
            )
        except TypeError:
            # Fallback to older API without window_size parameters
            flash_result = _flash_attn_varlen_forward(
                q,
                k_whole,
                v_whole,
                cu_seqlens_q,
                cu_seqlens_k,
                seqlen_q,
                seqlen_k,
                dropout_p,
                softmax_scale,
                causal=causal,
                return_softmax=False,
            )
        
        # Handle different API versions
        if len(flash_result) == 4:
            out, softmax_lse, q_modified, k_modified = flash_result
            out_padded = out
            rng_state = None
        elif len(flash_result) == 8:
            out, q_modified, k_modified, v_modified, out_padded, softmax_lse, S_dmask, rng_state = flash_result
        else:
            raise ValueError(f"Unexpected number of return values from _flash_attn_varlen_forward: {len(flash_result)}")
        
        ctx._kv_cache = kv_cache
        # Save the ORIGINAL tensors for backward pass, NOT the modified ones
        saved_tensors = [q_for_backward, k_for_backward, v_for_backward, out_padded if out_padded is not None else out, softmax_lse, cu_seqlens_q, cu_seqlens_k]
        if rng_state is not None:
            saved_tensors.append(rng_state)
        ctx.save_for_backward(*saved_tensors)
        ctx._has_rng_state = rng_state is not None
        ctx._batch_size = batch_size
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx._original_dtype = original_dtype
        
        # Convert output back to original dtype
        if original_dtype != out.dtype:
            out = out.to(original_dtype)
        return out 

    @staticmethod
    def backward(ctx, dout, *args):
        saved_tensors = ctx.saved_tensors
        if ctx._has_rng_state:
            q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = saved_tensors
        else:
            q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = saved_tensors
            rng_state = None
            
        # Convert to fp16 for Flash Attention backward if needed
        original_dtype = ctx._original_dtype
        if original_dtype not in [torch.float16, torch.bfloat16]:
            dout = dout.to(torch.float16)
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
            out = out.to(torch.float16)
            
        k_whole = ctx._kv_cache['k_cache']
        v_whole = ctx._kv_cache['v_cache']
        
        # Ensure cached tensors have correct dtype for backward
        if original_dtype not in [torch.float16, torch.bfloat16]:
            k_whole = k_whole.to(torch.float16)
            v_whole = v_whole.to(torch.float16)
        
        # Calculate batch_size
        if ctx._seqlen_q > 0:
            batch_size = q.size(0) // ctx._seqlen_q
        else:
            batch_size = ctx._batch_size
        
        # Update KV cache: remove the current sequence
        if ctx._seqlen_k > 0 and k_whole.shape[1] >= ctx._seqlen_k:
            pk = k_whole[:, :ctx._seqlen_k]
            pv = v_whole[:, :ctx._seqlen_k].contiguous()
            
            if ctx._seqlen_q > 0 and pk.shape[1] >= ctx._seqlen_q:
                ctx._kv_cache['k_cache'] = pk[:, :-ctx._seqlen_q]
                ctx._kv_cache['v_cache'] = pv[:, :-ctx._seqlen_q]
            else:
                ctx._kv_cache['k_cache'] = pk
                ctx._kv_cache['v_cache'] = pv
        else:
            ctx._kv_cache['k_cache'] = k_whole
            ctx._kv_cache['v_cache'] = v_whole
        
        # Extract relevant portions for gradient computation
        pk = k.contiguous()
        pv = v.contiguous()
        
        # Ensure all tensors are contiguous
        q = q.contiguous()
        dout = dout.contiguous()
        out = out.contiguous()
        
        # Initialize gradient tensors
        dq = torch.empty_like(q)
        dk = torch.empty_like(pk)  
        dv = torch.empty_like(pv)
        
        # Call Flash Attention backward
        _flash_attn_varlen_backward(
            dout,
            q,
            pk,
            pv,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx._seqlen_q,
            ctx._seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            -1,  # window_size_left
            -1,  # window_size_right
            0.0,  # softcap
            None,  # alibi_slopes
            False,  # deterministic
            rng_state,
            False,  # zero_tensors
        )
        
        # Ensure head dimension consistency
        dq = dq[..., : dout.shape[-1]]
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        
        # Only return gradients for the CURRENT INPUT, not the entire KV cache
        # In varlen format, dk and dv have shape [batch_size * seqlen_k, num_heads, head_dim]
        # But we need to return gradients only for the current input k,v with shape [batch_size * seqlen_q, num_heads, head_dim]
        
        current_input_len = ctx._seqlen_q  # Length of current input sequence
        offset = ctx._offset  # Length of cached sequence
        
        # Extract gradients for current input from each batch sample
        dk_current_list = []
        dv_current_list = []
        
        for batch_idx in range(batch_size):
            # For each batch sample, extract gradients for the current input portion
            start_idx = batch_idx * ctx._seqlen_k + offset  # Start of current input in this batch
            end_idx = start_idx + current_input_len  # End of current input in this batch
            
            dk_current_list.append(dk[start_idx:end_idx])
            dv_current_list.append(dv[start_idx:end_idx])
        
        # Concatenate all batch samples
        dk_current = torch.cat(dk_current_list, dim=0)
        dv_current = torch.cat(dv_current_list, dim=0)
        
        # Convert back to original tensor format [batch, seq, heads, dim]
        dq_final = rearrange(dq, '(b s) h d -> b s h d', b=batch_size, s=current_input_len).contiguous()
        dk_final = rearrange(dk_current, '(b s) h d -> b s h d', b=batch_size, s=current_input_len).contiguous()
        dv_final = rearrange(dv_current, '(b s) h d -> b s h d', b=batch_size, s=current_input_len).contiguous()
        
        # Convert gradients back to original dtype
        if original_dtype not in [torch.float16, torch.bfloat16]:
            dq_final = dq_final.to(original_dtype)
            dk_final = dk_final.to(original_dtype)
            dv_final = dv_final.to(original_dtype)
        
        return dq_final, dk_final, dv_final, None, None, None, None, None, None, None


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


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
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

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
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
        
        # Initialize flash attention kv cache
        self.flash_kv_cache = {}

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
        cache_hidden: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
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
            # Check if Flash Attention should be used
            use_flash_attention = (
                FLASH_ATTENTION_AVAILABLE and 
                q_len >= 128 and  # sequence length threshold
                bsz * q_len >= 512  # total token count threshold
            )
            
            if use_flash_attention:
                lck = len(cache_hidden[0]) if cache_hidden[0] else 0
                
                # Handle empty sequence case
                if q_len == 0:
                    attn_output = torch.zeros(
                        bsz, self.num_heads, 0, self.head_dim,
                        dtype=query_states.dtype,
                        device=query_states.device
                    )
                else:
                    cos, sin = self.rotary_emb(query_states, seq_len=q_len + lck)
                    cos, sin = cos.to(query_states.device), sin.to(query_states.device)
                    query_states, key_states = apply_rotary_pos_emb(
                        query_states, key_states, cos, sin, position_ids + lck
                    )

                    key_states = repeat_kv(key_states, self.num_key_value_groups)
                    value_states = repeat_kv(value_states, self.num_key_value_groups)

                    # Prepare Flash Attention input format
                    # query_states: [bsz, num_heads, q_len, head_dim] -> [bsz, q_len, num_heads, head_dim]
                    q = query_states.transpose(1, 2).contiguous()
                    k = key_states.transpose(1, 2).contiguous()
                    v = value_states.transpose(1, 2).contiguous()

                    # Use Flash Attention
                    attn_output = FlashAttnVarlenFunc.apply(
                        q,
                        k,
                        v,
                        self.flash_kv_cache,
                        0.0,  # dropout_p
                        True,  # causal
                        None,  # softmax_scale (auto-computed)
                    )
                    
                    # Convert output format [bsz * q_len, num_heads, head_dim] -> [bsz, num_heads, q_len, head_dim]
                    attn_output = rearrange(attn_output, '(b s) h d -> b h s d', b=bsz).contiguous()
                
            else:
                # Fallback to original implementation
                lck = len(cache_hidden[0]) if cache_hidden[0] else 0

                # Handle empty sequence case
                if q_len == 0:
                    attn_output = torch.zeros(
                        bsz, self.num_heads, 0, self.head_dim,
                        dtype=query_states.dtype,
                        device=query_states.device
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

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaMLP(nn.Module):
    def __init__(self, config, last=True):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # if last:
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # else:
        #     self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size * 2, bias=False)
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

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, last=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config, last=last)
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
        cache_hidden: List[List[torch.Tensor]] = [],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
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
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
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
            past_key_value=past_key_value,
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

    def __init__(self, config, quant_config=None) -> None:
        super().__init__(config)
        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.midlayer = LlamaDecoderLayer(config)

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
        t2d = torch.zeros(self.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
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
            logger.info("using ttt_length 1, no need to cache hidden states")
            cache_hidden = None
        else:
            logger.info(f"using ttt_length {ttt_length}, caching hidden states")
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
        attention_mask = self._prepare_decoder_attention_mask(
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
            past_key_value=None,
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
        use_cache: bool = True,
    ) -> torch.Tensor:
        return self.midlayer(
            input_emb=input_embeds,
            hidden_states=hidden_states,
            cache_hidden=cache_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
        )
