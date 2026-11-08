"""SDPA over the same explicit visibility mask as eager attention."""

from torch.nn import functional as F

from .eager import _repeat_kv


def sdpa_attention(query, key, value, *, attention_mask, scale, dropout_p, training):
    # Repeat explicitly, as the old HF SDPA path did. This also supports devices
    # whose SDPA kernel does not implement enable_gqa.
    groups = query.shape[1] // key.shape[1]
    return F.scaled_dot_product_attention(
        query,
        _repeat_kv(key, groups),
        _repeat_kv(value, groups),
        attn_mask=attention_mask,
        dropout_p=dropout_p if training else 0.0,
        is_causal=False,
        scale=scale,
    )
