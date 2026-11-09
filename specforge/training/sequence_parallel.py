"""Supply process groups and differentiable acceptance reductions to objectives."""

from functools import partial

import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn


def reduce_counts(*, group, local_correct, local_denom):
    return (
        dist_nn.all_reduce(local_correct, group=group),
        dist_nn.all_reduce(local_denom, group=group),
    )


def configure_sequence_parallel(runtime):
    from specforge.distributed import (
        get_draft_sp_group,
        get_sp_ring_group,
        get_sp_ulysses_group,
    )

    if not dist.is_initialized():
        raise RuntimeError("USP requires initialized sequence-parallel process groups")
    runtime.ring_group = get_sp_ring_group()
    runtime.ulysses_group = get_sp_ulysses_group()
    runtime.reduce_sequence_metrics = partial(reduce_counts, group=get_draft_sp_group())
