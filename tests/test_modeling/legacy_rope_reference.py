"""Run legacy RoPE in FP32 with one final cast, the canonical/SGLang precision rule.

Legacy keeps its own tables and rotation math, so model parity stays independent
of the canonical `apply_rotary`.
"""

from contextlib import contextmanager
from unittest.mock import patch

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding


def _fp32_tables(forward):
    def tables(module, hidden, *args, **kwargs):
        return forward(module, hidden.float(), *args, **kwargs)

    return tables


def _fp32_query_key(rotate):
    def rotate_in_fp32(query, key, cos, sin, *args, **kwargs):
        rotated_query, rotated_key = rotate(
            query.float(), key.float(), cos.float(), sin.float(), *args, **kwargs
        )
        return rotated_query.to(query.dtype), rotated_key.to(key.dtype)

    return rotate_in_fp32


def _fp32_single(rotate):
    def rotate_in_fp32(hidden, cos, sin, **kwargs):
        return rotate(hidden.float(), cos.float(), sin.float(), **kwargs).to(
            hidden.dtype
        )

    return rotate_in_fp32


@contextmanager
def legacy_fp32_rope():
    from specforge.modeling.draft import dflash, llama3_eagle

    with (
        patch.object(
            Qwen3RotaryEmbedding, "forward", _fp32_tables(Qwen3RotaryEmbedding.forward)
        ),
        patch.object(
            llama3_eagle.LlamaRotaryEmbedding,
            "forward",
            _fp32_tables(llama3_eagle.LlamaRotaryEmbedding.forward),
        ),
        patch.object(
            dflash,
            "apply_rotary_pos_emb",
            _fp32_query_key(dflash.apply_rotary_pos_emb),
        ),
        patch.object(dflash, "apply_mla_rope", _fp32_single(dflash.apply_mla_rope)),
        patch.object(
            llama3_eagle,
            "apply_rotary_pos_emb",
            _fp32_query_key(llama3_eagle.apply_rotary_pos_emb),
        ),
    ):
        yield


@contextmanager
def eager_legacy_reference():
    """Discard compiled legacy helpers before comparing eager model math."""
    torch._dynamo.reset()
    with patch("torch._dynamo.config.disable", True):
        yield
