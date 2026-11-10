"""FlashAttention-2 over padded sequences, with preparation shared by layers."""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..masks import PaddingMask
from .base import AttentionBackend


def _unpad_metadata(valid):
    lengths = valid.sum(dim=-1, dtype=torch.int32)
    if (lengths == 0).any():
        raise ValueError(
            "FlashAttention requires at least one query and key per sequence"
        )
    indices = valid.flatten().nonzero().flatten()
    cumulative = torch.cat((lengths.new_zeros(1), lengths.cumsum(0, dtype=torch.int32)))
    return indices, cumulative, int(lengths.max().item())


def _reject_lse_gradient(grad):
    raise NotImplementedError(
        "FlashAttention does not backpropagate through LSE; use eager or Flex"
    )


@dataclass(frozen=True, eq=False)
class FlashMask:
    query_indices: Tensor
    key_indices: Tensor
    cu_query: Tensor
    cu_key: Tensor
    max_query: int
    max_key: int
    causal: bool
    window: tuple[int, int]
    active_queries: Tensor


class FlashAttentionBackend(AttentionBackend):
    def validate(self, *, head_dim, value_head_dim, dropout_p):
        if head_dim > 256 or value_head_dim != head_dim:
            raise ValueError(
                "FlashAttention requires equal Q/K/V head dimensions no greater than 256"
            )

    def prepare(self, layout, *, batch_size, num_heads, is_causal, sliding_window):
        padding = layout.visibility
        if not isinstance(padding, PaddingMask):
            raise ValueError(
                "FlashAttention requires PaddingMask visibility; use Flex for arbitrary masks"
            )
        layout.validate_dimensions(batch_size, num_heads)
        qvalid = padding.query_valid.expand(batch_size, -1)
        kvalid = padding.key_valid.expand(batch_size, -1)
        if layout.active_queries is not None:
            qvalid = qvalid & layout.active_queries
        qi, cq, mq = _unpad_metadata(qvalid)
        ki, ck, mk = _unpad_metadata(kvalid)
        if is_causal or sliding_window is not None:
            # FA2 aligns query/key sequence ends. Check logical positions after
            # unpadding so causal/window semantics are identical to other backends.
            ends = []
            for valid, positions in (
                (qvalid, layout.query_positions),
                (kvalid, layout.key_positions),
            ):
                positions = positions.expand(batch_size, -1)
                ranks = valid.long().cumsum(-1)
                offsets = positions - ranks
                first = valid.long().argmax(-1, keepdim=True)
                start = offsets.gather(1, first)
                if ((offsets != start) & valid).any():
                    raise ValueError(
                        "FlashAttention requires consecutive unpadded positions"
                    )
                ends.append(start.squeeze(1) + valid.sum(-1))
            if not torch.equal(*ends):
                raise ValueError(
                    "FlashAttention requires bottom-right aligned query/key positions"
                )
        window = (
            (-1, -1)
            if sliding_window is None
            else (sliding_window - 1, 0 if is_causal else sliding_window - 1)
        )
        active = qvalid
        # Q can be longer than KV. Preserve packing/alignment while marking
        # leading rows with no visible key inactive (including projection bias).
        aligned = (
            qvalid.long().cumsum(-1)
            - qvalid.sum(-1, keepdim=True)
            + kvalid.sum(-1, keepdim=True)
            - 1
        )
        if is_causal:
            active = active & (aligned >= 0)
        elif sliding_window is not None:
            active = active & (aligned >= 1 - sliding_window)
        return FlashMask(qi, ki, cq, ck, mq, mk, is_causal, window, active), active

    def attend(
        self, query, key, value, *, mask, scale, dropout_p, training, return_lse
    ):
        device_type = query.device.type
        if query.dtype == torch.float32 and torch.is_autocast_enabled(device_type):
            # weighted QK norm promotes Q/K to FP32 under autocast; cast like SDPA
            dtype = torch.get_autocast_dtype(device_type)
            query, key, value = query.to(dtype), key.to(dtype), value.to(dtype)
        if device_type != "cuda" or query.dtype not in (
            torch.float16,
            torch.bfloat16,
        ):
            raise ValueError("FlashAttention requires CUDA fp16/bf16 query/key/value")
        self.validate(
            head_dim=query.shape[-1],
            value_head_dim=value.shape[-1],
            dropout_p=dropout_p if training else 0.0,
        )
        from flash_attn import flash_attn_varlen_func

        def unpad(tensor, indices):
            return tensor.transpose(1, 2).flatten(0, 1)[indices]

        result = flash_attn_varlen_func(
            unpad(query, mask.query_indices),
            unpad(key, mask.key_indices),
            unpad(value, mask.key_indices),
            mask.cu_query,
            mask.cu_key,
            mask.max_query,
            mask.max_key,
            dropout_p=dropout_p if training else 0.0,
            softmax_scale=scale,
            causal=mask.causal,
            window_size=mask.window,
            return_attn_probs=return_lse,
        )
        output, lse = (result[0], result[1]) if return_lse else (result, None)
        batch, heads, length, width = query.shape
        padded = output.new_zeros(batch * length, heads, width).index_copy(
            0, mask.query_indices, output
        )
        if lse is not None:
            lse = lse.new_full((heads, batch * length), -torch.inf).index_copy(
                1, mask.query_indices, lse
            )
            lse = lse.view(heads, batch, length).transpose(0, 1)
            lse = lse.masked_fill(~mask.active_queries[:, None, :], -torch.inf)
            if lse.requires_grad:
                # flash_attn's varlen backward silently drops the LSE gradient
                lse.register_hook(_reject_lse_gradient)
        return padded.view(batch, length, heads, width).transpose(1, 2), lse
