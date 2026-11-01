"""Use Transformers' SDPA adapter, retaining its GQA and mask handling."""

from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS


def sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs):
    return ALL_ATTENTION_FUNCTIONS["sdpa"](
        module, query, key, value, attention_mask, **kwargs
    )
