"""Align only legacy RoPE precision when testing complete model composition."""

from contextlib import contextmanager
from unittest.mock import patch

from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from specforge.modeling.draft.layers.rope import apply_rotary


@contextmanager
def legacy_fp32_rope():
    from specforge.modeling.draft.llama3_eagle import LlamaRotaryEmbedding

    qwen_forward = Qwen3RotaryEmbedding.forward
    llama_forward = LlamaRotaryEmbedding.forward

    def qwen_tables(module, hidden, *args, **kwargs):
        return qwen_forward(module, hidden.float(), *args, **kwargs)

    def llama_tables(module, hidden, *args, **kwargs):
        return llama_forward(module, hidden.float(), *args, **kwargs)

    def context_rope(query, key, cos, sin, position_ids=None, unsqueeze_dim=1):
        length = query.shape[-2]
        return (
            apply_rotary(query, (cos[:, -length:].float(), sin[:, -length:].float())),
            apply_rotary(key, (cos.float(), sin.float())),
        )

    def indexed_rope(query, key, cos, sin, position_ids, unsqueeze_dim=1):
        cos, sin = cos.squeeze(1).squeeze(0), sin.squeeze(1).squeeze(0)
        tables = (cos[position_ids].float(), sin[position_ids].float())
        return apply_rotary(query, tables), apply_rotary(key, tables)

    with (
        patch.object(Qwen3RotaryEmbedding, "forward", qwen_tables),
        patch.object(LlamaRotaryEmbedding, "forward", llama_tables),
        patch("specforge.modeling.draft.dflash.apply_rotary_pos_emb", context_rope),
        patch(
            "specforge.modeling.draft.llama3_eagle.apply_rotary_pos_emb", indexed_rope
        ),
    ):
        yield
