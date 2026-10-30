# coding=utf-8
"""Qwen3.5-MoE family (Qwen3-Next, Qwen3.5, Qwen3.8): the MoE layer from its config."""

from torch import nn
from transformers import PretrainedConfig

from specforge.modeling.draft.moe.experts import RoutedExperts, SharedExpert
from specforge.modeling.draft.moe.layer import MoELayer
from specforge.modeling.draft.moe.router import TopKRouter


def build_moe(config: PretrainedConfig) -> MoELayer:
    """Softmax top-k router, stacked SwiGLU experts, sigmoid-gated shared expert."""
    if config.hidden_act != "silu":
        raise ValueError(
            f"qwen3_5_moe experts use silu, got hidden_act={config.hidden_act!r}"
        )
    hidden_size, std = config.hidden_size, config.initializer_range
    shared_size = getattr(config, "shared_expert_intermediate_size", 0)
    dflash_config = getattr(config, "dflash_config", None) or {}
    return MoELayer(
        gate=TopKRouter(
            config.num_experts,
            hidden_size,
            config.num_experts_per_tok,
            # Qwen3.8's config omits it; transformers always renormalizes.
            norm_topk_prob=getattr(config, "norm_topk_prob", True),
            initializer_range=std,
            # torchtitan's units: 1 when balanced, one coefficient per layer.
            aux_loss_coef=dflash_config.get("moe_router_aux_loss_coef", 0.0),
            # torchtitan's aux_loss_type: batch_wise (the default here) or microbatch_wise.
            aux_loss_type=dflash_config.get("moe_router_aux_loss_type", "batch_wise"),
            # DeepSeek-style aux-loss-free balancing, moved once per optimizer step.
            selection_bias=bool(dflash_config.get("moe_router_selection_bias", False)),
            bias_update_rate=dflash_config.get("moe_router_bias_update_rate", 0.001),
        ),
        experts=RoutedExperts(
            config.num_experts,
            hidden_size,
            config.moe_intermediate_size,
            initializer_range=std,
        ),
        shared_expert=SharedExpert(hidden_size, shared_size) if shared_size else None,
        shared_expert_gate=(
            nn.Linear(hidden_size, 1, bias=False) if shared_size else None
        ),
    )
