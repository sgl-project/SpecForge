"""BlockMask preparation and compiled FlexAttention execution."""

import torch
from torch.nn.attention.flex_attention import (
    AuxRequest,
    create_block_mask,
    flex_attention,
)

from .base import AttentionBackend

_compiled_attention = torch.compile(flex_attention)
_compiled_block_mask = torch.compile(create_block_mask)


class FlexAttentionBackend(AttentionBackend):
    def __init__(self, *, kernel_backend="TRITON"):
        if kernel_backend not in {"TRITON", "FLASH"}:
            raise ValueError("Flex kernel_backend must be TRITON or FLASH")
        # AUTO can select flex-decoding, which rejects sparse short-query batches.
        self.kernel_backend = kernel_backend

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
        if training and dropout_p:
            raise ValueError("FlexAttention does not support attention dropout")
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
