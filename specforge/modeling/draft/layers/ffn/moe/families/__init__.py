# coding=utf-8
"""MoE families, keyed by the draft config's ``moe_preset``."""

from specforge.modeling.draft.layers.ffn.moe.families import deepseek_v4, qwen3_5_moe

FAMILIES = {
    "deepseek_v4": deepseek_v4.build_moe,
    "qwen3_5_moe": qwen3_5_moe.build_moe,
}
