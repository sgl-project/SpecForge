"""Sequence-parallel process groups and detached loss statistics."""

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import torch
import torch.distributed as dist


def reduce_statistics(accepted, count, denominator, *, group):
    """Sum detached statistics over one sequence, leaving DP samples independent."""
    values = torch.stack(
        (accepted, count, count if denominator is None else denominator)
    )
    dist.all_reduce(values, group=group)
    return values[0], values[1], None if denominator is None else values[2]


@dataclass(frozen=True)
class SequenceParallel:
    """The sequence-parallel axis as an objective model takes it at construction.

    The one handle that crosses into a program: the process groups of a
    model-parallel axis. Ranks of one group share a sequence, not samples.
    """

    #: Group of the ring-attention exchange.
    ring_group: Any
    #: Group of the Ulysses all-to-all.
    ulysses_group: Any
    #: ``(accepted, count, denominator)`` summed over the ranks of one sequence.
    reduce_statistics: Callable


def sequence_parallel_groups() -> SequenceParallel:
    """This process's sequence-parallel groups, as ``init_distributed`` built them."""
    from specforge.distributed import (
        get_draft_sp_group,
        get_sp_ring_group,
        get_sp_ulysses_group,
    )

    return SequenceParallel(
        ring_group=get_sp_ring_group(),
        ulysses_group=get_sp_ulysses_group(),
        reduce_statistics=partial(reduce_statistics, group=get_draft_sp_group()),
    )
