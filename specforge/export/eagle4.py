"""Derive EAGLE-4 retrieval factors without changing dense checkpoint weights."""

from __future__ import annotations

import torch

from specforge.modeling.draft.eagle4_structure import (
    EAGLE4_ARCH,
    efficient_head_export_config,
)


@torch.no_grad()
def add_eagle4_svd_factors(config, state_dict, *, device="cpu"):
    """Return inference weights with W ~= B @ A, retaining W for exact rescoring.

    Compute an unweighted truncated SVD in FP32. Store the factors in W's dtype.
    Runtime training checkpoints never call this function or contain these factors.
    """
    if EAGLE4_ARCH not in (getattr(config, "architectures", None) or ()):
        return state_dict
    head = efficient_head_export_config(config)
    if head is None:
        return state_dict
    factor_keys = ("efficient_lm_head.weight", "efficient_lm_head_down_proj.weight")
    if any(key in state_dict for key in factor_keys):
        raise ValueError(
            "SVD export requires a dense stage-1 checkpoint without factors"
        )
    weight = state_dict["lm_head.weight"]
    if tuple(weight.shape) != (config.vocab_size, config.hidden_size):
        raise ValueError("Dense lm_head shape does not match the EAGLE-4 config")
    work = weight.detach().to(device=device, dtype=torch.float32)
    if not torch.isfinite(work).all():
        raise ValueError("Cannot export SVD factors from a non-finite dense head")
    u, singular, vh = torch.linalg.svd(work, full_matrices=False)
    rank = head["rank"]
    result = dict(state_dict)
    result[factor_keys[0]] = (
        (u[:, :rank] * singular[:rank])
        .to(device=weight.device, dtype=weight.dtype)
        .contiguous()
    )
    result[factor_keys[1]] = (
        vh[:rank].to(device=weight.device, dtype=weight.dtype).contiguous()
    )
    return result
