"""BlockMask preparation and compiled FlexAttention execution."""

import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import AuxRequest, BlockMask, create_block_mask
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.attention.flex_attention import flex_attention as torch_flex_attention

from .base import AttentionBackend
from .policy import flex_attention_backend

_compiled_attention = torch.compile(flex_attention)
_compiled_block_mask = torch.compile(create_block_mask)


class FlexAttentionBackend(AttentionBackend):
    def __init__(self, *, kernel_backend="TRITON"):
        if kernel_backend not in {"TRITON", "FLASH"}:
            raise ValueError("Flex kernel_backend must be TRITON or FLASH")
        # AUTO can select flex-decoding, which rejects sparse short-query batches.
        self.kernel_backend = kernel_backend

    def validate(self, *, head_dim, value_head_dim, dropout_p):
        if dropout_p:
            raise ValueError("FlexAttention does not support attention dropout")

    def prepare(self, layout, *, batch_size, num_heads, is_causal, sliding_window):
        qlen, klen = layout.query_positions.shape[1], layout.key_positions.shape[1]
        if not qlen or not klen:
            raise ValueError("Flex requires nonempty query/key sequences")
        if layout.active_queries is None:
            raise ValueError("Flex preparation requires explicit active_queries")
        prepare = (
            create_block_mask if torch.compiler.is_compiling() else _compiled_block_mask
        )
        mask = prepare(
            layout.mask_mod(is_causal=is_causal, sliding_window=sliding_window),
            batch_size,
            layout.mask_heads(num_heads),
            qlen,
            klen,
            device=layout.query_positions.device,
            BLOCK_SIZE=(256, 128) if self.kernel_backend == "FLASH" else 128,
        )
        return mask, layout.active_queries.expand(batch_size, -1)

    def attend(
        self, query, key, value, *, mask, scale, dropout_p, training, return_lse
    ):
        self.validate(
            head_dim=query.shape[-1],
            value_head_dim=value.shape[-1],
            dropout_p=dropout_p if training else 0.0,
        )
        compute = (
            flex_attention if torch.compiler.is_compiling() else _compiled_attention
        )
        output, aux = compute(
            query,
            key,
            value,
            block_mask=mask,
            scale=scale,
            enable_gqa=query.shape[1] != key.shape[1],
            return_aux=AuxRequest(lse=True),
            kernel_options={"BACKEND": self.kernel_backend},
        )
        return output, aux.lse


_attention = torch.compile(torch_flex_attention, dynamic=True)


def flex_attention(
    query, key, value, *, attention_mask, scale, dropout_p, training, return_valid=False
):
    if training and dropout_p:
        raise ValueError("FlexAttention does not support attention dropout")
    batch, heads, queries, _ = query.shape
    keys = key.shape[-2]
    value_width = value.shape[-1]

    def for_width(width):
        return max(16, 1 << (width - 1).bit_length())

    query = F.pad(query, (0, for_width(query.shape[-1]) - query.shape[-1]))
    key = F.pad(key, (0, for_width(key.shape[-1]) - key.shape[-1]))
    value = F.pad(value, (0, for_width(value_width) - value_width))

    def visible(b, h, q, k):
        return (
            (q < queries)
            & (k < keys)
            & attention_mask[
                b % attention_mask.shape[0],
                h % attention_mask.shape[1],
                q.clamp(max=queries - 1),
                k.clamp(max=keys - 1),
            ]
        )

    backend = flex_attention_backend() or "TRITON"
    block_mask = (
        attention_mask
        if isinstance(attention_mask, BlockMask)
        else create_block_mask(
            visible,
            B=batch,
            H=None if attention_mask.shape[1] == 1 else heads,
            Q_LEN=queries,
            KV_LEN=keys,
            device=query.device,
            BLOCK_SIZE=(256, 128) if backend == "FLASH" else 128,
        )
    )
    options = (
        {"BACKEND": backend}
        if tuple(map(int, torch.__version__.split(".")[:2])) >= (2, 11)
        else {"FORCE_USE_FLEX_ATTENTION": True}
    )
    result = _attention(
        query,
        key,
        value,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=query.shape[1] != key.shape[1],
        kernel_options=options,
        return_lse=return_valid,
    )
    if return_valid:
        output, lse = result
        return output[..., :value_width], torch.isfinite(lse).any(dim=1)
    return result[..., :value_width]
