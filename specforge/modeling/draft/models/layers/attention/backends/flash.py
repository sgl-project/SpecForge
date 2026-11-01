"""Dispatch explicitly selected FlashAttention implementations through HF."""

from transformers.models.qwen3.modeling_qwen3 import ALL_ATTENTION_FUNCTIONS


def flash_attention_forward(module, query, key, value, attention_mask, **kwargs):
    implementation = module.config._attn_implementation
    return ALL_ATTENTION_FUNCTIONS[implementation](
        module, query, key, value, attention_mask, **kwargs
    )
