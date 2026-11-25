# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Read the optimizer state the reference runtime wrote.

The reference runtime (``FSDPTrainingBackend`` with ``BF16Optimizer``) keeps
fp32 masters and Adam moments beside bf16 parameters. Under DDP every rank
saved them whole. Under FSDP1 each rank saved the 1-D slices of its own flat
shard, in FSDP1's padded flat layout. This module reads either, one parameter
at a time, and hands each whole parameter to the caller, which keeps its own
part of it; so a run can resume on a runtime with another layout.

The FSDP1 layout is not recomputed here: it is replayed by wrapping an empty
copy of the program with FSDP1 exactly as the reference wrapped it, and read
from FSDP1's own record of where each parameter's slice lies
(``FlatParameter._shard_param_infos``). Each parameter's slices are then
gathered from every rank and placed; a slice that lands where another did, or
a position no slice covers, fails the conversion.
"""

from __future__ import annotations

import copy
import functools
from dataclasses import dataclass
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import nn

#: Adam's per-parameter tensors that share a parameter's layout.
_MOMENTS = ("exp_avg", "exp_avg_sq")


@dataclass(frozen=True)
class ReferenceState:
    """The reference optimizer's state, per trainable parameter in order.

    Each master and moment is what the caller kept of the whole tensor.
    ``states[i]`` is ``{"step", "exp_avg", "exp_avg_sq"}``, or empty for a
    parameter the reference never stepped.
    """

    masters: Tuple[torch.Tensor, ...]
    states: Tuple[Mapping[str, torch.Tensor], ...]
    param_groups: Sequence[Mapping]
    scheduler_state_dict: Mapping
    lr_scheduler_type: str
    max_grad_norm: Optional[float]


def empty_structure(module: nn.Module) -> nn.Module:
    """A copy of ``module`` whose parameters and buffers are on the meta device.

    Holds no tensor data of ``module``'s parameters and buffers.
    """
    return _copy_on(module, torch.device("meta"))


def read_reference_state(
    saved: Mapping,
    *,
    shapes: Sequence[torch.Size],
    structure: nn.Module,
    trainable: str,
    frozen: Sequence[str],
    blocks: Sequence[str],
    group=None,
    keep: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    state_indices: Optional[Sequence[int]] = None,
) -> ReferenceState:
    """This rank's saved ``BF16Optimizer`` state, read whole, kept in part.

    A collective over ``group`` when ``torch.distributed`` is initialized:
    every rank calls it with its own ``saved``. The ranks first agree on
    whether the state is whole (DDP) or FSDP1 slices, which one rank's shapes
    cannot always tell.

    Args:
        saved: ``BF16Optimizer.state_dict()`` as this rank wrote it (or the
            replicated state every rank read).
        shapes: The full shape of each trainable parameter, in order.
        structure: An ``empty_structure`` of the program as the reference
            built it: same modules, same parameter dtypes.
        trainable: Name of the module the reference optimizer trained, in
            ``structure`` (``"draft_model"``).
        frozen: Names, in ``structure``, of the modules the reference kept out
            of sharding.
        blocks: Class names the reference wrapped as one FSDP1 unit each.
        group: The group the reference sharded over, gloo, in its rank order.
        keep: ``keep(index, full)``: what this rank keeps of ``full``, the
            whole master or moment of trainable parameter ``index``, on the
            host. Whole when omitted. One parameter is whole at a time.
        state_indices: For each trainable parameter ``index`` in
            ``named_parameters()`` order, its key in
            ``saved["optimizer_state_dict"]["state"]`` (which uses flattened
            ``param_groups`` order). Defaults to ``range(len(shapes))``.

    Raises:
        ValueError: The saved state does not fit the parameters, or the
            replayed layout does not place every slice exactly once.
    """
    keep = keep or (lambda index, full: full)
    masters = list(saved["fp32_params"])
    if len(masters) != len(shapes):
        raise ValueError(
            f"saved optimizer has {len(masters)} masters for {len(shapes)} "
            "trainable parameters"
        )
    index_map = range(len(masters)) if state_indices is None else state_indices
    if len(index_map) != len(masters):
        raise ValueError(
            f"state_indices has {len(index_map)} entries for {len(masters)} "
            "trainable parameters"
        )
    optimizer = saved["optimizer_state_dict"]
    states = [
        _checked_state(optimizer["state"].get(state_index, {}), index)
        for index, state_index in enumerate(index_map)
    ]
    whole = _all_agree(all(m.shape == s for m, s in zip(masters, shapes)), group)
    spans = (
        None if whole else _fsdp1_slices(structure, trainable, frozen, blocks, group)
    )
    kept_masters, kept_states = [], []
    failure = None
    for index, shape in enumerate(shapes):
        if whole:
            master, state = masters[index].float(), states[index]
        else:
            master, state = _gathered(
                index, shape, spans[index], masters[index], states[index], group
            )
        try:
            kept_masters.append(keep(index, master))
            kept_states.append(
                {
                    name: keep(index, value) if name in _MOMENTS else value
                    for name, value in state.items()
                }
            )
        except Exception as error:
            # This rank's own failure: it keeps taking part in the remaining
            # gathers and fails once they are over, as its peers do not wait.
            failure = failure or error
    if failure is not None:
        raise failure
    return ReferenceState(
        masters=tuple(kept_masters),
        states=tuple(kept_states),
        param_groups=optimizer["param_groups"],
        scheduler_state_dict=saved["scheduler_state_dict"],
        lr_scheduler_type=saved.get("lr_scheduler_type", "cosine"),
        max_grad_norm=saved.get("max_grad_norm"),
    )


def _checked_state(state: Mapping, index: int) -> dict:
    """One parameter's saved AdamW state: empty, or its step and both moments."""
    unknown = sorted(set(state) - {"step", *_MOMENTS})
    if unknown:
        raise ValueError(
            f"parameter {index}'s saved optimizer state holds {unknown}; only "
            "AdamW's step and moments are read"
        )
    if not state:
        return {}
    missing = [name for name in ("step", *_MOMENTS) if name not in state]
    if missing:
        raise ValueError(f"parameter {index}'s saved optimizer state lacks {missing}")
    return {
        "step": torch.as_tensor(state["step"]).detach().cpu(),
        **{name: state[name].float() for name in _MOMENTS},
    }


def _all_agree(value: bool, group) -> bool:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    votes = [None] * dist.get_world_size(group)
    dist.all_gather_object(votes, value, group=group)
    return all(votes)


def _gathered(index, shape, span, master, state, group):
    """One parameter's master and state, whole, from every rank's slice.

    Only the ranks whose shard holds part of the parameter have its slices and,
    once the reference stepped it, its moments.
    """
    parts = [None] * dist.get_world_size(group)
    dist.all_gather_object(parts, (span, master, state), group=group)
    full_master = _place(shape, [(span, values) for span, values, _ in parts])
    holders = [part for part in parts if part[0] is not None]
    stepped = [part for part in holders if part[2]]
    if not stepped:
        return full_master, {}
    if len(stepped) != len(holders):
        raise ValueError(
            f"the ranks holding parameter {index} disagree on whether the "
            "reference stepped it"
        )
    steps = sorted({float(state["step"]) for _, _, state in stepped})
    if len(steps) != 1:
        raise ValueError(f"the ranks holding parameter {index} saved steps {steps}")
    full = {"step": stepped[0][2]["step"]}
    for name in _MOMENTS:
        full[name] = _place(
            shape,
            [
                (span, state[name] if state else torch.empty(0))
                for span, _, state in parts
            ],
        )
    return full_master, full


def _place(shape: torch.Size, parts) -> torch.Tensor:
    """Assemble one full tensor from ``(range, slice)`` pairs, each rank's."""
    numel = int(torch.Size(shape).numel())
    flat = torch.zeros(numel, dtype=torch.float32)
    covered = torch.zeros(numel, dtype=torch.int32)
    for span, values in parts:
        if span is None:
            if values.numel():
                raise ValueError("a rank holds values outside its shard")
            continue
        start, end = span
        if values.numel() != end - start + 1:
            raise ValueError(
                f"a slice of {values.numel()} values for positions {start}..{end}"
            )
        flat[start : end + 1] = values.reshape(-1).float()
        covered[start : end + 1] += 1
    if not bool((covered == 1).all()):
        raise ValueError(
            "the replayed FSDP1 layout does not cover the parameter exactly once; "
            "the checkpoint was written with another layout"
        )
    return flat.view(shape)


def _copy_on(module: nn.Module, device: torch.device) -> nn.Module:
    """A copy of ``module`` with empty parameters and buffers on ``device``.

    A tensor registered under several names (tied weights) stays one tensor,
    and no module's ``_apply`` runs on the empty storage.
    """
    memo = {}
    for tensor in [*module.parameters(), *module.buffers()]:
        if id(tensor) in memo:
            continue
        empty = torch.empty_like(tensor, device=device)
        memo[id(tensor)] = (
            nn.Parameter(empty, requires_grad=tensor.requires_grad)
            if isinstance(tensor, nn.Parameter)
            else empty
        )
    return copy.deepcopy(module, memo)


def _fsdp1_slices(structure, trainable, frozen, blocks, group) -> List:
    """Each trainable parameter's span in this rank's FSDP1 shard, or None."""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    program = _copy_on(structure, torch.device("cpu"))
    if not all(p.is_contiguous() for p in program.parameters()):
        raise ValueError(
            "the program holds a non-contiguous parameter, whose FSDP1 flat "
            "layout is not replayed"
        )
    ignored = [program.get_submodule(name) for name in frozen]
    classes = {type(m) for m in program.modules() if type(m).__name__ in blocks}
    options = dict(
        use_orig_params=True,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        process_group=group,
        device_id=torch.device("cpu"),
    )
    if ignored:
        options["ignored_modules"] = ignored
    if classes:
        options["auto_wrap_policy"] = functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=classes
        )
    wrapped = FSDP(program, **options)
    spans = {}
    for unit in wrapped.modules():
        if isinstance(unit, FSDP) and unit._handle is not None:
            flat = unit._handle.flat_param
            for param, info in zip(flat._params, flat._shard_param_infos):
                spans[id(param)] = (
                    (info.intra_param_start_idx, info.intra_param_end_idx)
                    if info.in_shard
                    else None
                )
    return [
        spans[id(param)]
        for param in program.get_submodule(trainable).parameters()
        if param.requires_grad
    ]


__all__ = ["ReferenceState", "empty_structure", "read_reference_state"]
