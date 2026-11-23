# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The FSDP2 runtime: fp32 masters, low-precision compute, fp32 gradients."""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.contracts import PrecisionSpec, RankFailure

#: Sharding strategy names (``training.fsdp_sharding``) and what they mean here.
#: ``NO_SHARD`` replicates (hybrid sharding with one rank per shard group).
_RESHARD_AFTER_FORWARD = {
    "FULL_SHARD": True,
    "SHARD_GRAD_OP": False,
    "NO_SHARD": False,
}


class FSDP2TrainingBackend(FSDPTrainingBackend):
    """Data-parallel runtime on FSDP2 with fp32 masters.

    The trainable draft is stored in fp32: its parameters are the optimizer's
    masters. Forward and backward run in the spec's compute dtype; gradients are upcast
    to fp32 as they are produced, accumulate across micro-batches in fp32 and
    are reduced across ranks in fp32. Frozen modules stay as built, replicated
    and outside the optimizer. The wrapper never casts the program's inputs.

    The runner sees the reference backend's contract: one call per
    micro-batch, the window's gradient collective at its boundary, gradients
    averaged over the ranks (FSDP2's default), and a step that clips, checks
    and clears.

    Limit (torch 2.14): a parameter that receives no gradient on an earlier
    micro-batch of a window but one on its last fails FSDP2's reduce-scatter,
    which then holds that gradient in the compute dtype beside the others'
    fp32 accumulators ("FSDP reduce-scatter expects uniform gradient dtype").
    """

    name = "fsdp2"

    def __init__(
        self,
        parallel: ParallelConfig,
        *,
        optimizer_factory: Optional[Callable[[nn.Module], Any]] = None,
        precision: PrecisionSpec,
    ) -> None:
        """
        Raises:
            ValueError: An unknown sharding strategy, or a precision this runtime
                does not hold: fp32 parameters that are the masters and fp32
                gradients.
        """
        if parallel.sharding_strategy not in _RESHARD_AFTER_FORWARD:
            raise ValueError(
                f"unknown sharding strategy {parallel.sharding_strategy!r}; "
                f"expected one of {sorted(_RESHARD_AFTER_FORWARD)}"
            )
        if not (
            precision.parameter_dtype == torch.float32
            and precision.gradient_dtype == torch.float32
            and precision.master_dtype is None
        ):
            raise ValueError(
                "the fsdp2 runtime holds fp32 parameters as the masters and fp32 "
                f"gradients; {precision} asks otherwise"
            )
        super().__init__(parallel, optimizer_factory=optimizer_factory)
        self.precision = precision
        self.compute_dtype = precision.compute_dtype
        self._trainable: Tuple[nn.Parameter, ...] = ()
        self._artifact_dtypes: Dict[str, torch.dtype] = {}
        self._reference_layout: dict = {}

    @property
    def optimizer_state_is_replicated(self) -> bool:
        # NO_SHARD replicates every parameter, so every rank holds the same
        # masters and moments; sharded states are each rank's own.
        return self.parallel_config.sharding_strategy == "NO_SHARD"

    def prepare_model(
        self,
        model: nn.Module,
        *,
        wrap: bool = True,
        optimizer_target: Optional[nn.Module] = None,
        frozen: Tuple[nn.Module, ...] = (),
        blocks: Tuple[str, ...] = (),
    ) -> nn.Module:
        """Store the trainable draft in fp32 and shard it with FSDP2.

        Args:
            model: The program.
            wrap: Must be True: one process trains through the plain runner.
            optimizer_target: The module whose parameters the optimizer takes;
                the program when omitted.
            frozen: Frozen modules of ``model``: kept as built, replicated.
            blocks: Class names of the modules sharded as one unit each.

        Raises:
            ValueError: A frozen module holds a trainable parameter, or
                ``wrap`` is False.
        """
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        if not wrap:
            raise ValueError("the FSDP2 runtime wraps the program; use PlainRunner")
        frozen = tuple(frozen)
        if any(p.requires_grad for module in frozen for p in module.parameters()):
            raise ValueError("frozen modules must contain only frozen parameters")
        target = model if optimizer_target is None else optimizer_target
        from .reference_state import empty_structure

        # What the reference runtime would have wrapped, kept without data so
        # a resume can read the state it wrote.
        names = {id(module): name for name, module in model.named_modules()}
        self._reference_layout = dict(
            structure=empty_structure(model),
            trainable=names[id(target)],
            frozen=tuple(names[id(module)] for module in frozen),
            blocks=tuple(blocks),
        )
        self._artifact_dtypes = _store_in_fp32(
            model, frozen=frozen, artifact_dtype=self.precision.artifact_dtype
        )
        mesh = self._mesh(next(model.parameters()).device)
        policy = MixedPrecisionPolicy(
            param_dtype=self.compute_dtype,
            reduce_dtype=torch.float32,
            cast_forward_inputs=False,
        )
        options = dict(
            mesh=mesh,
            mp_policy=policy,
            reshard_after_forward=_RESHARD_AFTER_FORWARD[
                self.parallel_config.sharding_strategy
            ],
            # Every unit, not only the root: a frozen table inside a block is
            # otherwise taken by that block's unit before the root sees it.
            ignored_params={p for module in frozen for p in module.parameters()},
        )
        block_classes = {
            type(module)
            for module in model.modules()
            if type(module).__name__ in blocks
        }
        for module in list(model.modules()):
            if type(module) in block_classes:
                fully_shard(module, **options)
        fully_shard(model, **options)
        if self.parallel_config.sharding_strategy == "NO_SHARD":
            _broadcast_replicas(model, mesh.get_group("replicate"))
        self.module = model
        self._wrapped = True
        self._wrapper_kind = "fsdp2"
        self.auto_wrap_block_classes = block_classes
        self.ignored_frozen_modules = frozen
        self._trainable = tuple(p for p in target.parameters() if p.requires_grad)
        if self._optimizer_factory is not None:
            from specforge.optimizer import FP32Optimizer

            self.optimizer = self._optimizer_factory(target)
            if not isinstance(self.optimizer, FP32Optimizer):
                raise TypeError(
                    "the FSDP2 runtime keeps the fp32 masters in the parameters; "
                    f"it steps with FP32Optimizer, not {type(self.optimizer).__name__}"
                )
        return self.module

    def _mesh(self, device: torch.device):
        from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

        group = self.parallel_config.fsdp_process_group
        world = dist.get_world_size(group) if dist.is_initialized() else 1
        if self.parallel_config.sharding_strategy == "NO_SHARD":
            if dist.is_initialized() and world != dist.get_world_size():
                raise ValueError(
                    "the fsdp2 runtime replicates (NO_SHARD) over every rank of the "
                    f"job; its group has {world} of {dist.get_world_size()}"
                )
            # Replicated: hybrid sharding with one rank per shard group. The
            # meshes create their groups here, on every rank in one order.
            return init_device_mesh(
                device.type, (world, 1), mesh_dim_names=("replicate", "shard")
            )
        return DeviceMesh.from_group(group or dist.group.WORLD, device.type)

    def backward(self, loss: torch.Tensor, *, is_boundary: bool = True) -> None:
        """Backward one micro-batch; the gradient collective runs at the boundary.

        Between boundaries FSDP2 keeps the unsharded gradient in fp32 and the
        parameters gathered after backward, so a window reduces once; under
        ``SHARD_GRAD_OP`` and ``NO_SHARD`` it also gathers once.
        """
        self.module.set_requires_gradient_sync(is_boundary)
        self.module.set_reshard_after_backward(is_boundary)
        loss.backward()

    def accumulation_context(self, *, is_boundary: bool = True):
        return contextlib.nullcontext()

    def scale_gradients(self, factor: torch.Tensor) -> None:
        if self.module is None:
            raise RuntimeError("scale_gradients called before prepare_model")
        from specforge.optimizer import _local

        grads = [_local(p.grad) for p in self._trainable if p.grad is not None]
        if grads:
            with torch.no_grad():
                torch._foreach_mul_(grads, factor.to(grads[0].device))

    def state_dict(self) -> dict:
        """``{"model", "optimizer", "rng"}``: the program's full state on rank 0.

        The model's entries the runtime stored in fp32 leave in their artifact
        dtype; the fp32 masters travel in the optimizer's state, this rank's
        shards (the whole state under ``NO_SHARD``).
        """
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        if self.module is None:
            raise RuntimeError("state_dict called before prepare_model")
        # A collective: a rank that fails inside it is already apart.
        model = get_model_state_dict(
            self.module,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )
        if dist.is_initialized() and dist.get_rank() != 0:
            model = {}
        try:
            # Rounded as a cast rounds. This rank's own work, past the gather.
            model = {
                name: (
                    value.to(self._artifact_dtypes[name])
                    if name in self._artifact_dtypes
                    else value
                )
                for name, value in model.items()
            }
            optimizer = (
                self.optimizer.state_dict() if self.optimizer is not None else None
            )
            rng = self._rng_state()
        except Exception as error:
            raise RankFailure(f"{type(error).__name__}: {error}") from error
        return {"model": model, "optimizer": optimizer, "rng": rng}

    def load_state_dict(self, state: dict) -> None:
        """Restore optimizer and RNG state; reads the reference runtime's too.

        An optimizer state ``FP32Optimizer`` did not write is the reference
        runtime's (``BF16Optimizer`` under DDP or FSDP1): every rank reads it,
        one parameter at a time, and keeps its own shard.
        """
        from specforge.optimizer import FP32Optimizer

        saved = state.get("optimizer")
        if saved is not None and FP32Optimizer.FORMAT_KEY not in saved:
            try:
                # Past the conversion's collectives: a failure is this rank's.
                converted = self._from_reference(saved)
            except ValueError as error:
                raise RankFailure(f"{type(error).__name__}: {error}") from error
            state = {**state, "optimizer": converted}
        super().load_state_dict(state)

    def _from_reference(self, saved) -> dict:
        from specforge.optimizer import FP32Optimizer

        from .reference_state import read_reference_state

        group = self.parallel_config.fsdp_process_group or dist.group.WORLD
        replay_group = None
        if dist.is_initialized():
            if dist.get_world_size(group) != dist.get_world_size():
                raise ValueError(
                    "reading the fsdp runtime's optimizer state needs this runtime's "
                    "group to be the whole job, as the reference's was"
                )
            if dist.get_backend(group) != "gloo":
                # The replay runs FSDP1 on the host. Over the whole job, in its
                # rank order: the order the reference sharded in.
                replay_group = group = dist.new_group(backend="gloo")
        try:
            state = read_reference_state(
                saved,
                shapes=[p.shape for p in self._trainable],
                group=group,
                keep=lambda index, full: _local_part(full, self._trainable[index]),
                **self._reference_layout,
            )
        finally:
            if replay_group is not None:
                dist.destroy_process_group(replay_group)
        return {
            "optimizer_state_dict": {
                # A parameter the reference never stepped starts afresh.
                "state": {index: dict(s) for index, s in enumerate(state.states) if s},
                "param_groups": state.param_groups,
            },
            "scheduler_state_dict": state.scheduler_state_dict,
            "lr_scheduler_type": state.lr_scheduler_type,
            "max_grad_norm": state.max_grad_norm,
            "fp32_params": list(state.masters),
            FP32Optimizer.FORMAT_KEY: True,
        }

    def _load_module_state_dict(self, model_state: dict) -> None:
        raise ValueError(
            "the fsdp2 runtime restores optimizer and RNG state; the program's "
            "weights load into it before the wrap"
        )


def _store_in_fp32(
    program: nn.Module, *, frozen, artifact_dtype: torch.dtype
) -> Dict[str, torch.dtype]:
    """Store the trainable parameters and floating buffers of ``program`` in fp32.

    Each trainable parameter is replaced by an exact fp32 copy, in place, and
    each floating buffer by an fp32 copy, as FSDP1's ``buffer_dtype`` casts
    them; a tensor registered under several names (tied, aliased) stays one
    tensor. Frozen modules, and everything inside them, are left as built.

    Returns:
        For each state-dict name now stored in fp32, the dtype it is written
        in: ``artifact_dtype`` for parameters, the built dtype for buffers
        (FSDP1's full state dict restores buffers to it).
    """
    inside_frozen = {id(m) for module in frozen for m in module.modules()}
    artifact: Dict[str, torch.dtype] = {}
    replaced: Dict[int, torch.Tensor] = {}
    for prefix, owner in program.named_modules(remove_duplicate=False):
        if id(owner) in inside_frozen:
            continue
        # ``_parameters`` / ``_buffers`` directly: ``named_*`` drops a second
        # name for the same tensor, which would then keep the old one.
        for name, param in list(owner._parameters.items()):
            if param is None or not param.requires_grad:
                continue
            if id(param) not in replaced:
                replaced[id(param)] = nn.Parameter(
                    param.detach().to(torch.float32), requires_grad=True
                )
            owner._parameters[name] = replaced[id(param)]
            artifact[f"{prefix}.{name}" if prefix else name] = artifact_dtype
        for name, buffer in list(owner._buffers.items()):
            if buffer is None or not buffer.is_floating_point():
                continue
            built = buffer.dtype
            if id(buffer) not in replaced:
                replaced[id(buffer)] = buffer.to(torch.float32)
            owner._buffers[name] = replaced[id(buffer)]
            if name not in owner._non_persistent_buffers_set:
                artifact[f"{prefix}.{name}" if prefix else name] = built
    return artifact


def _local_part(full: torch.Tensor, param) -> torch.Tensor:
    """This rank's part of the whole tensor ``full``, cut on the host as
    ``param`` is sharded (``torch.chunk`` per sharded mesh dimension, empty
    parts last, as ``Shard`` splits)."""
    from torch.distributed.tensor import DTensor, Shard

    if not isinstance(param, DTensor):
        return full
    part = full
    coordinate = param.device_mesh.get_coordinate()
    for mesh_dim, placement in enumerate(param.placements):
        if isinstance(placement, Shard):
            chunks = torch.chunk(
                part, param.device_mesh.size(mesh_dim), dim=placement.dim
            )
            index = coordinate[mesh_dim]
            part = (
                chunks[index]
                if index < len(chunks)
                else part.narrow(placement.dim, 0, 0)
            )
    if part.shape != param.to_local().shape:
        raise ValueError(
            f"a part of shape {tuple(part.shape)} for this rank's shard of shape "
            f"{tuple(param.to_local().shape)}"
        )
    return part.contiguous()


def _broadcast_replicas(program: nn.Module, group) -> None:
    """Give every replica rank 0's parameters and buffers, as DDP does at wrap."""
    from specforge.optimizer import _local

    source = dist.get_global_rank(group, 0)
    with torch.no_grad():
        for tensor in [*program.parameters(), *program.buffers()]:
            dist.broadcast(_local(tensor), src=source, group=group)


__all__ = ["FSDP2TrainingBackend"]
