"""Recurrent state after each block's visible context prefix, for all blocks at once.

A block reading ``L`` context rows starts from the KDA state after rows
``[0, L)``. Within a row, the sorted distinct lengths cut the context into
segments; ``group_size`` consecutive segments form a group. A two-level scan
reaches every length exactly in about ``2 * sqrt(segments)`` launches instead
of one sequential launch per segment: span launches chain whole groups (all
rows together, one launch per group index), then segment launches advance
every in-group offset of every group of every row (one launch per offset).

The plan is computed on the host from one device-to-host copy per forward and
holds no parameters, so the model prepares it once and every KDA layer reuses
it outside its FSDP unit.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from ..masks import ContextBlocks
from .kernels import FLA_CHUNK_SIZE, round_up


@dataclass(frozen=True, eq=False)
class ScanLaunch:
    """One variable-length launch, described on the host."""

    num_tokens: int  # padded to a multiple of FLA_CHUNK_SIZE
    num_sequences: int  # real sequences; padding sequences read the zero row
    cu_seqlens_cpu: Tensor
    cu_seqlens: Tensor
    initial_slots: Tensor  # state-pool slot per padded sequence
    final_slots: Tensor  # state-pool slot per real sequence


@dataclass(frozen=True, eq=False)
class ContextScanPlan:
    """Launch plan of the two-level scan for one batch of context blocks."""

    batch_size: int
    num_context_rows: int
    num_blocks: int
    group_size: int
    num_state_slots: int
    span_launches: tuple[ScanLaunch, ...]
    segment_launches: tuple[ScanLaunch, ...]
    span_token_index: Tensor  # flattened-context rows of all span launches
    segment_token_index: Tensor  # flattened-context rows of all segment launches
    group_input_slots: Tensor  # copied into running_slots before segment launches
    running_slots: Tensor
    final_row_by_block: Tensor  # [batch * blocks] row of the collected final states
    num_finals: int
    # [batch, blocks] context rows each block reads; inactive blocks read none
    context_lengths: Tensor


def _pad_launch_sequences(
    sequences: list[tuple[int, int, int]], zero_index: int
) -> tuple[list[int], list[int], int, int]:
    """Token index and cu_seqlens of (start, end, row_base) sequences, padded.

    Same policy as the FLA variable-length wrapper: one-token padding
    sequences up to a power-of-two count and a chunk-aligned token count, the
    last one absorbing the remainder. Padding tokens read the zero row.
    """
    lengths = [end - start for start, end, _ in sequences]
    total = sum(lengths)
    num_sequences = len(sequences)
    padded_sequences = 1 << max(num_sequences - 1, 0).bit_length()
    extra = padded_sequences - num_sequences
    padded_total = round_up(total + extra, FLA_CHUNK_SIZE)
    if padded_total > total and extra == 0:
        padded_sequences *= 2
        extra = padded_sequences - num_sequences
        padded_total = round_up(total + extra, FLA_CHUNK_SIZE)
    token_index: list[int] = []
    for start, end, base in sequences:
        token_index.extend(range(base + start, base + end))
    token_index.extend([zero_index] * (padded_total - total))
    cu_seqlens = [0]
    for length in lengths:
        cu_seqlens.append(cu_seqlens[-1] + length)
    for j in range(extra):
        cu_seqlens.append(total + j + 1 if j < extra - 1 else padded_total)
    return token_index, cu_seqlens, padded_sequences, padded_total


def _host_context_lengths(blocks: ContextBlocks) -> list[list[int]]:
    """Copy the lengths each block scans to the host and check them, in one sync."""
    lengths, active = blocks.context_lengths.long(), blocks.active_blocks
    batch, count = lengths.shape
    valid_prefix = blocks.context_valid.long().cumprod(-1).sum(-1).expand(batch)
    host = torch.cat((lengths, active.long(), valid_prefix[:, None]), dim=1).cpu()
    lengths, active, valid_prefix = host.split((count, count, 1), dim=1)
    if bool(((lengths < 0) | (lengths > blocks.num_context_rows)).any()):
        raise ValueError("context_lengths must lie in [0, context rows]")
    # A scan reads every row of the prefix; padding inside it would be read
    # by KDA while attention skips it.
    if bool((active.bool() & (lengths > valid_prefix)).any()):
        raise ValueError(
            "KDA requires each active block's visible context to be unpadded"
        )
    return (lengths * active).tolist()


def build_context_scan_plan(
    blocks: ContextBlocks, *, group_size: int | None = None
) -> ContextScanPlan:
    """Plan the scan for every block; inactive blocks start from the zero state."""
    rows = blocks.num_context_rows
    device = blocks.context_lengths.device
    lengths = _host_context_lengths(blocks)
    batch_size = len(lengths)
    num_blocks = blocks.context_lengths.shape[1]
    # Distinct lengths cut each row into segments; a zero length is an empty one.
    boundaries_by_row = [sorted(set(row)) for row in lengths]
    num_segments = [len(b) for b in boundaries_by_row]
    group = (
        int(group_size)
        if group_size
        else max(1, math.isqrt(max(max(num_segments, default=0), 1)))
    )
    num_groups = [-(-count // group) for count in num_segments]
    max_groups = max(num_groups, default=0)

    # Pool slots: 0 is the zero state; group inputs, then running slots per row.
    next_slot = 1
    group_inputs: list[list[int]] = []
    running: list[list[int]] = []
    for row in range(batch_size):
        later_groups = max(num_groups[row] - 1, 0)
        group_inputs.append([0] + list(range(next_slot, next_slot + later_groups)))
        next_slot += later_groups
        running.append(list(range(next_slot, next_slot + num_groups[row])))
        next_slot += num_groups[row]
    zero_index = batch_size * rows  # the zero row appended to the flat context

    final_row_by_segment: dict[tuple[int, int], int] = {}
    num_finals = 0

    def plan_launch(sequences, initial_slots, final_slots, owners, token_index):
        nonlocal num_finals
        index, cu_seqlens, padded_sequences, padded_total = _pad_launch_sequences(
            sequences, zero_index
        )
        initial_slots = initial_slots + [0] * (padded_sequences - len(sequences))
        for owner in owners:
            final_row_by_segment[owner] = num_finals
            num_finals += 1
        token_index.extend(index)
        return ScanLaunch(
            num_tokens=padded_total,
            num_sequences=len(sequences),
            cu_seqlens_cpu=torch.tensor(cu_seqlens, dtype=torch.long),
            cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.long, device=device),
            initial_slots=torch.tensor(initial_slots, dtype=torch.long, device=device),
            final_slots=torch.tensor(final_slots, dtype=torch.long, device=device),
        )

    span_launches, span_token_index = [], []
    for g in range(max_groups):
        sequences, initial_slots, final_slots, owners = [], [], [], []
        for row in range(batch_size):
            if g >= num_groups[row]:
                continue
            bounds = [0] + boundaries_by_row[row]
            start = bounds[g * group]
            last = min((g + 1) * group, num_segments[row]) - 1
            end = bounds[last + 1]
            if end <= start:
                continue  # only a lone zero length: the zero state
            sequences.append((start, end, row * rows))
            initial_slots.append(group_inputs[row][g])
            # A group writes the next group's input; a row's last group writes
            # its own running slot, which only feeds the collected finals.
            final_slots.append(
                group_inputs[row][g + 1] if g + 1 < num_groups[row] else running[row][g]
            )
            owners.append((row, last))
        if sequences:
            span_launches.append(
                plan_launch(
                    sequences, initial_slots, final_slots, owners, span_token_index
                )
            )

    segment_launches, segment_token_index = [], []
    for offset in range(group):
        sequences, initial_slots, final_slots, owners = [], [], [], []
        for row in range(batch_size):
            bounds = [0] + boundaries_by_row[row]
            for g in range(num_groups[row]):
                segment = g * group + offset
                # A group's last segment ends at its span and reuses that final.
                if segment >= min((g + 1) * group, num_segments[row]) - 1:
                    continue
                start, end = bounds[segment], bounds[segment + 1]
                if end <= start:
                    continue  # a zero length: the zero state
                sequences.append((start, end, row * rows))
                initial_slots.append(running[row][g])
                final_slots.append(running[row][g])
                owners.append((row, segment))
        if sequences:
            segment_launches.append(
                plan_launch(
                    sequences, initial_slots, final_slots, owners, segment_token_index
                )
            )

    # An empty segment has no final; it maps to the zero row after the finals.
    final_row_by_block = []
    for row in range(batch_size):
        segment_by_length = {
            length: index for index, length in enumerate(boundaries_by_row[row])
        }
        for length in lengths[row]:
            final_row_by_block.append(
                final_row_by_segment.get((row, segment_by_length[length]), num_finals)
            )

    def as_index(values):
        return torch.tensor(values, dtype=torch.long, device=device)

    return ContextScanPlan(
        batch_size=batch_size,
        num_context_rows=rows,
        num_blocks=num_blocks,
        group_size=group,
        num_state_slots=next_slot,
        span_launches=tuple(span_launches),
        segment_launches=tuple(segment_launches),
        span_token_index=as_index(span_token_index),
        segment_token_index=as_index(segment_token_index),
        group_input_slots=as_index([s for slots in group_inputs for s in slots]),
        running_slots=as_index([s for slots in running for s in slots]),
        final_row_by_block=as_index(final_row_by_block),
        num_finals=num_finals,
        context_lengths=torch.where(
            blocks.active_blocks, blocks.context_lengths.long(), 0
        ),
    )


def scan_context_states(
    kda_fn: Callable,
    k: Tensor,
    v: Tensor,
    gate: Tensor,
    beta: Tensor,
    A_log: Tensor,
    dt_bias: Tensor,
    lower_bound: float | None,
    plan: ContextScanPlan,
) -> Tensor:
    """FP32 states ``[batch, blocks, H, K, V]`` from context ``[batch, rows, ...]``.

    All rows share one launch chain; token gathers are one ``index_select`` and
    ``split`` per level, and states live in one pool updated out of place.
    """
    batch_size, rows = k.shape[:2]
    if (batch_size, rows) != (plan.batch_size, plan.num_context_rows):
        raise ValueError("context projections must match the scan plan")
    num_heads, key_dim, value_dim = k.shape[2], k.shape[3], v.shape[3]

    def flatten(tensor: Tensor) -> Tensor:
        flat = tensor.reshape(batch_size * rows, *tensor.shape[2:])
        return torch.cat((flat, flat.new_zeros((1, *flat.shape[1:]))), dim=0)

    k_flat, v_flat, gate_flat, beta_flat = (flatten(t) for t in (k, v, gate, beta))
    pool = k.new_zeros(
        (plan.num_state_slots, num_heads, key_dim, value_dim), dtype=torch.float32
    )
    max_tokens = max(
        (launch.num_tokens for launch in plan.span_launches + plan.segment_launches),
        default=0,
    )
    # The state does not depend on queries; one zero buffer serves every launch.
    zero_q = k.new_zeros((1, max_tokens, num_heads, key_dim))
    finals: list[Tensor] = []
    for is_segment_level, token_index, launches in (
        (False, plan.span_token_index, plan.span_launches),
        (True, plan.segment_token_index, plan.segment_launches),
    ):
        if not launches:
            continue
        if is_segment_level:
            pool = pool.index_copy(
                0, plan.running_slots, pool.index_select(0, plan.group_input_slots)
            )
        sizes = [launch.num_tokens for launch in launches]
        ks, vs, gates, betas = (
            torch.split(t.index_select(0, token_index), sizes, dim=0)
            for t in (k_flat, v_flat, gate_flat, beta_flat)
        )
        for i, launch in enumerate(launches):
            _, final_state = kda_fn(
                zero_q[:, : launch.num_tokens],
                ks[i].unsqueeze(0),
                vs[i].unsqueeze(0),
                gates[i].unsqueeze(0),
                betas[i].unsqueeze(0),
                A_log,
                dt_bias,
                lower_bound,
                initial_state=pool.index_select(0, launch.initial_slots),
                output_final_state=True,
                cu_seqlens=launch.cu_seqlens,
                cu_seqlens_cpu=launch.cu_seqlens_cpu,
            )
            real = final_state[: launch.num_sequences]
            pool = pool.index_copy(0, launch.final_slots, real)
            finals.append(real)
    collected = torch.cat(finals + [pool[:1]], dim=0)
    states = collected.index_select(0, plan.final_row_by_block)
    return states.view(batch_size, plan.num_blocks, num_heads, key_dim, value_dim)


def gather_left_context_rows(context: Tensor, lengths: Tensor, window: int) -> Tensor:
    """The ``window`` context rows before each block, zero where none exist."""
    batch_size, _, width = context.shape
    num_blocks = lengths.shape[1]
    if window == 0:
        return context.new_zeros((batch_size, num_blocks, 0, width))
    offsets = torch.arange(window, 0, -1, device=lengths.device)
    positions = lengths.unsqueeze(-1) - offsets
    valid = positions >= 0
    flat = positions.clamp(min=0).reshape(batch_size, -1)
    rows = context.gather(1, flat.unsqueeze(-1).expand(-1, -1, width))
    rows = rows.reshape(batch_size, num_blocks, window, width)
    return rows * valid.unsqueeze(-1).to(rows.dtype)
