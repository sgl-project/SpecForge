"""Reference softmax attention and boolean-mask conversion."""

from typing import Optional

import torch
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward


def prepare_eager_mask(attention_mask: Optional[torch.Tensor], dtype: torch.dtype):
    if attention_mask is None or attention_mask.dtype != torch.bool:
        return attention_mask, None
    valid_queries = attention_mask.any(dim=-1, keepdim=True)
    additive_mask = torch.zeros_like(attention_mask, dtype=dtype)
    additive_mask.masked_fill_(~attention_mask, torch.finfo(dtype).min)
    return additive_mask, valid_queries


__all__ = ["eager_attention_forward", "prepare_eager_mask"]
