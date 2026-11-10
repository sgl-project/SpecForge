"""Dense mask preparation shared by eager and SDPA execution."""

from torch import Tensor
from torch.nn import functional as F

from ..masks import dense_attention_mask
from .base import AttentionBackend
from .eager import eager_attention


class DenseAttentionBackend(AttentionBackend):
    def prepare(self, layout, *, batch_size, num_heads, is_causal, sliding_window):
        mask = dense_attention_mask(
            layout,
            batch_size=batch_size,
            num_heads=num_heads,
            is_causal=is_causal,
            sliding_window=sliding_window,
        )
        return mask, mask.any(dim=-1).any(dim=1).expand(batch_size, -1)


class EagerAttentionBackend(DenseAttentionBackend):
    def attend(
        self, query, key, value, *, mask: Tensor, scale, dropout_p, training, return_lse
    ):
        result = eager_attention(
            query,
            key,
            value,
            attention_mask=mask,
            scale=scale,
            dropout_p=dropout_p,
            training=training,
            return_lse=return_lse,
        )
        return result if return_lse else (result, None)


class SDPAAttentionBackend(DenseAttentionBackend):
    def attend(
        self, query, key, value, *, mask: Tensor, scale, dropout_p, training, return_lse
    ):
        if return_lse:
            raise NotImplementedError(
                "SDPA does not expose attention LSE; use eager or Flex"
            )
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            scale=scale,
            dropout_p=dropout_p if training else 0.0,
            enable_gqa=query.shape[1] != key.shape[1],
        )
        return output, None
