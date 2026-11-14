"""Sequence-parallel process groups and detached loss statistics."""

from functools import partial

import torch
import torch.distributed as dist


def reduce_statistics(accepted, count, denominator, *, group):
    """Sum detached statistics over one sequence, leaving DP samples independent."""
    values = torch.stack(
        (accepted, count, count if denominator is None else denominator)
    )
    dist.all_reduce(values, group=group)
    return values[0], values[1], None if denominator is None else values[2]


def configure_sequence_parallel(runtime):
    from specforge.distributed import (
        get_draft_sp_group,
        get_sp_ring_group,
        get_sp_ulysses_group,
    )

    runtime.ring_group = get_sp_ring_group()
    runtime.ulysses_group = get_sp_ulysses_group()
    runtime.reduce_sequence_statistics = partial(
        reduce_statistics, group=get_draft_sp_group()
    )
