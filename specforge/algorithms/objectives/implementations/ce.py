"""Shared applicability checks for projection-fused cross entropy."""

import torch


def projection_ce_support(request, group):
    """Return None or a rejection reason for the requested fused CE group."""
    p = request.predictions
    if not p.hidden.is_cuda or p.hidden.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        return "requires CUDA FP16/BF16/FP32"
    if p.hidden.numel() == 0:
        return "empty projection uses the reference path"
    if not p.projection.chunk_safe or not p.projection.recompute_safe:
        return "projection does not permit chunked/recomputed fusion"
    if request.materialized:
        return "projection is already materialized"
    if p.auxiliary is not None:
        return "auxiliary computation consumes full projection outputs"
    if p.vocab_mask is not None:
        return "vocabulary mapping requires a different projection implementation"
    if any(
        request.terms[i].loss.kind != "ce" or request.terms[i].loss.options
        for i in group
    ):
        return "requires ordinary hard-label CE"
    return None
