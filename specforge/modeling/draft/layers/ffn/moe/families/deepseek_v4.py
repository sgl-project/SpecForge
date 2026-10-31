# coding=utf-8
"""DeepSeek-V4 family: the MoE layer from its config (sqrtsoftplus router, clamped SwiGLU)."""

import torch
import torch.nn.functional as F
from transformers import PretrainedConfig

from specforge.modeling.draft.layers.ffn.moe.experts import (
    Activation,
    RoutedExperts,
    SharedExpert,
)
from specforge.modeling.draft.layers.ffn.moe.layer import MoELayer
from specforge.modeling.draft.layers.ffn.moe.router import TopKRouter


def sqrtsoftplus_scores(logits: torch.Tensor) -> torch.Tensor:
    """``sqrt(softplus(logits))``: DeepSeek-V4's router scores."""
    return F.softplus(logits).sqrt()


def clamped_swiglu(limit: float) -> Activation:
    """DeepSeek-V4's SwiGLU: ``gate`` capped at ``limit``, ``up`` clamped to it both ways."""

    def activation(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.silu(gate.clamp(max=limit)) * up.clamp(min=-limit, max=limit)

    return activation


def build_moe(config: PretrainedConfig) -> MoELayer:
    """Sqrtsoftplus top-k router, clamped SwiGLU experts, one ungated shared expert."""
    if config.hidden_act != "silu":
        raise ValueError(
            f"deepseek_v4 experts use silu, got hidden_act={config.hidden_act!r}"
        )
    scoring_func = getattr(config, "scoring_func", "sqrtsoftplus")
    if scoring_func != "sqrtsoftplus":
        raise ValueError(
            f"deepseek_v4 routes with sqrtsoftplus, got scoring_func={scoring_func!r}"
        )
    hidden_size, std = config.hidden_size, config.initializer_range
    intermediate_size = config.moe_intermediate_size
    activation = clamped_swiglu(getattr(config, "swiglu_limit", 10.0))
    shared_size = getattr(config, "n_shared_experts", 1) * intermediate_size
    dflash_config = getattr(config, "dflash_config", None) or {}
    return MoELayer(
        gate=TopKRouter(
            config.n_routed_experts,
            hidden_size,
            config.num_experts_per_tok,
            # DeepSeek-V4 always renormalizes the top-k weights; the flag is not read.
            norm_topk_prob=True,
            initializer_range=std,
            score_fn=sqrtsoftplus_scores,
            route_scale=config.routed_scaling_factor,
            # DeepSeek-style aux-loss-free balancing, moved once per optimizer step.
            expert_bias=bool(dflash_config.get("moe_router_enable_expert_bias", False)),
            bias_update_rate=dflash_config.get("moe_router_bias_update_rate", 0.001),
        ),
        experts=RoutedExperts(
            config.n_routed_experts,
            hidden_size,
            intermediate_size,
            initializer_range=std,
            activation=activation,
        ),
        # V4's shared expert has no gate; n_shared_experts widens it (1 in V4).
        shared_expert=(
            SharedExpert(hidden_size, shared_size, activation=activation)
            if shared_size
            else None
        ),
        # The load-balance loss: 1 when balanced, one coefficient per layer; 0 skips it.
        aux_loss_coef=dflash_config.get("moe_router_aux_loss_coef", 0.0),
        # batch_wise (the default) balances the step's pooled load, microbatch_wise this batch's.
        aux_loss_type=dflash_config.get("moe_router_aux_loss_type", "batch_wise"),
    )
