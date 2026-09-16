# coding=utf-8
"""Expert parallelism for the routed experts: the group, and the autograd seam.

Tokens stay replicated across the expert-parallel group and the experts are
partitioned: each rank owns a disjoint slice and computes only the tokens routed
into it. The combined routed output is the globally reduced sum, while autograd
follows the local expert graph, so expert weight gradients stay rank-local and
the input gradient is summed exactly once.

This is the MoE half of the runtime support in ``specforge/distributed.py`` and
``specforge/optimizer.py``; a module that shards its experts also marks them
``_specforge_rank_local_parameters`` so the optimizer counts replicated
parameters once in the global gradient norm.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist


def draft_ep_layout() -> Tuple[Optional[object], int, int]:
    """``(group, rank, size)`` of the draft expert-parallel group, or a size of 1."""
    if not dist.is_available() or not dist.is_initialized():
        return None, 0, 1
    try:
        from specforge.distributed import get_draft_ep_group

        group = get_draft_ep_group()
    except (ImportError, AttributeError):
        return None, 0, 1
    if group is None:
        return None, 0, 1
    return group, dist.get_rank(group), dist.get_world_size(group)


class CopyToExpertParallel(torch.autograd.Function):
    """Identity in forward; sums the local input gradients in backward.

    The routed input is replicated, so every rank's experts contribute a term to
    the same ``dL/dx``. Without this seam each rank would keep only its own term.
    """

    @staticmethod
    def forward(ctx, value: torch.Tensor, group):
        ctx.group = group
        # Returning the input object lets autograd elide the boundary in some
        # versions; the clone makes the backward collective explicit.
        return value.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        gradient = grad_output.clone()
        dist.all_reduce(gradient, group=ctx.group)
        return gradient, None


def reduce_routed_output(local: torch.Tensor, group) -> torch.Tensor:
    """Numerically the group-wide routed sum; structurally the local graph.

    The all-reduced value is attached as a detached correction so backward walks
    only this rank's experts — their gradients are already the right ones, and
    reducing them again would multiply them by the group size.
    """
    total = local.detach().clone()
    dist.all_reduce(total, group=group)
    return local + (total - local).detach()


def reduce_replicated_gradients(module: torch.nn.Module, group) -> None:
    """All-reduce a replicated module's parameter gradients over the EP group.

    The router is replicated but only sees gradient from the experts this rank
    owns, so its local gradient is a partial sum of the real one.
    """

    def _hook(gradient: torch.Tensor) -> torch.Tensor:
        reduced = gradient.clone()
        dist.all_reduce(reduced, group=group)
        return reduced

    for parameter in module.parameters():
        if parameter.requires_grad:
            parameter.register_hook(_hook)


__all__ = [
    "CopyToExpertParallel",
    "draft_ep_layout",
    "reduce_replicated_gradients",
    "reduce_routed_output",
]
