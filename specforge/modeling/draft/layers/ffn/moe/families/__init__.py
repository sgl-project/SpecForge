# coding=utf-8
"""MoE families, keyed by the draft config's ``moe_preset``."""

from specforge.modeling.draft.layers.ffn.moe.families import qwen3_5_moe

FAMILIES = {"qwen3_5_moe": qwen3_5_moe.build_moe}
