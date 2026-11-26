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
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn

from specforge.optimizer import FSDP2_OPTIMIZER_FORMAT_KEY, local_tensor
from specforge.training.backend import (
    FSDPTrainingBackend,
    ParallelConfig,
    _optimizer_params,
)
from specforge.training.contracts import (
    OptimizerSpec,
    PrecisionSpec,
    RankFailure,
    SchedulerSpec,
)

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
    masters, so any optimizer from the shared registry (``torch.optim.AdamW``,
    ``Muon``, custom optimizers) updates the sharded DTensor parameters
    directly without a separate master-copy wrapper. Forward and backward run
    in the spec's compute dtype; gradients are upcast to fp32 as they are
    produced, accumulate across micro-batches in fp32 and are reduced across
    ranks in fp32. Frozen modules stay as built, replicated and outside the
    optimizer. The wrapper never casts the program's inputs.

    The runner sees the reference backend's contract: one call per
    micro-batch, the window's gradient collective at its boundary, gradients
    averaged over the ranks (FSDP2's default), and a runtime-owned step that
    clips, validates scalars in one host read, steps, clears and schedules.

    Limit (torch 2.14, with ``gradient_dtype=fp32``): a parameter that receives
    no gradient on an earlier micro-batch of a window but one on its last fails
    FSDP2's reduce-scatter, which then holds that gradient in the compute dtype
    beside the others' fp32 accumulators ("FSDP reduce-scatter expects uniform
    gradient dtype"). Opting into ``gradient_dtype=compute_dtype`` avoids this
    by accumulating and reducing uniformly in the compute dtype (with sharded
    grads still cast to the fp32 masters after reduction).
    """

    name = "fsdp2"
    FORMAT_KEY = FSDP2_OPTIMIZER_FORMAT_KEY

    def __init__(
        self,
        parallel: ParallelConfig,
        *,
        optimizer_factory: Optional[Callable[[nn.Module], Any]] = None,
        optimizer_spec: Optional[OptimizerSpec] = None,
        scheduler_spec: Optional[SchedulerSpec] = None,
        max_grad_norm: Optional[float] = None,
        precision: PrecisionSpec,
    ) -> None:
        """
        Raises:
            ValueError: An unknown sharding strategy, or a precision this runtime
                does not hold: fp32 parameters as masters, fp32 or compute-dtype
                gradients and fp32 floating buffers.
        """
        if parallel.sharding_strategy not in _RESHARD_AFTER_FORWARD:
            raise ValueError(
                f"unknown sharding strategy {parallel.sharding_strategy!r}; "
                f"expected one of {sorted(_RESHARD_AFTER_FORWARD)}"
            )
        super().__init__(
            parallel,
            optimizer_factory=optimizer_factory,
            optimizer_spec=optimizer_spec,
            scheduler_spec=scheduler_spec,
            max_grad_norm=max_grad_norm,
            precision=precision,
        )
        self.compute_dtype = precision.compute_dtype
        self._trainable: Tuple[nn.Parameter, ...] = ()
        self._artifact_dtypes: Dict[str, torch.dtype] = {}
        self._owned_groups: Tuple[Any, ...] = ()
        self._reference_layout: Optional[dict] = None

    def _validate_precision(self, precision: PrecisionSpec) -> None:
        if not (
            precision.parameter_dtype == torch.float32
            and precision.gradient_dtype in (torch.float32, precision.compute_dtype)
            and precision.master_dtype is None
            and precision.buffer_dtype == torch.float32
        ):
            raise ValueError(
                "the fsdp2 runtime holds fp32 parameters as the masters, fp32 or "
                "compute-dtype gradients and fp32 floating buffers; "
                f"{precision} asks otherwise"
            )

    @property
    def optimizer_state_is_replicated(self) -> bool:
        # NO_SHARD replicates every parameter, so every rank holds the same
        # masters and moments; sharded states are each rank's own.
        return self.parallel_config.sharding_strategy == "NO_SHARD"

    def close(self) -> None:
        """Destroy any auxiliary mesh subgroups created for ``NO_SHARD``."""
        groups, self._owned_groups = self._owned_groups, ()
        for group in groups:
            dist.destroy_process_group(group)

    def prepare_model(
        self,
        model: nn.Module,
        *,
        wrap: bool = True,
        optimizer_target: Optional[nn.Module] = None,
        frozen_modules: Tuple[nn.Module, ...] = (),
        blocks: Tuple[str, ...] = (),
        retain_reference_layout: bool = False,
    ) -> nn.Module:
        """Store the trainable draft in fp32 and shard it with FSDP2.

        Args:
            model: The program.
            wrap: Must be True; FSDP2 always wraps the program.
            optimizer_target: The module whose parameters the optimizer takes;
                the program when omitted.
            frozen_modules: Frozen modules of ``model``: kept as built,
                replicated.
            blocks: Class names of the modules sharded as one unit each.
            retain_reference_layout: Capture a meta-device copy of the
                unwrapped module tree so ``load_state_dict`` can translate an
                ``fsdp`` (FSDP1/DDP) checkpoint. Kept False on fresh runs so no
                extra module graph is retained, and freed as soon as
                ``load_state_dict`` finishes.

        Raises:
            ValueError: A frozen module holds a trainable parameter, or
                ``wrap`` is False.
        """
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

        if not wrap:
            raise ValueError("the FSDP2 runtime requires wrap=True")
        if any(
            p.requires_grad for module in frozen_modules for p in module.parameters()
        ):
            raise ValueError("frozen modules must contain only frozen parameters")
        block_classes = {
            type(module)
            for module in model.modules()
            if type(module).__name__ in blocks
        }
        # As the reference runtime: names that match nothing would train the
        # program as one root unit. Validate before mutating model dtypes or
        # creating mesh subgroups so failure is side-effect-free.
        if blocks and not block_classes:
            raise ValueError(
                f"blocks {sorted(blocks)} name no module class of the model"
            )
        target = model if optimizer_target is None else optimizer_target
        if retain_reference_layout:
            from .reference_state import empty_structure

            names = {id(module): name for name, module in model.named_modules()}
            self._reference_layout = dict(
                structure=empty_structure(model),
                trainable=names[id(target)],
                frozen=tuple(names[id(module)] for module in frozen_modules),
                blocks=tuple(blocks),
            )
        else:
            self._reference_layout = None
        self._artifact_dtypes = _store_in_fp32(
            model,
            frozen_modules=frozen_modules,
            artifact_dtype=self.precision.artifact_dtype,
        )
        mesh = self._mesh(next(model.parameters()).device)
        policy = MixedPrecisionPolicy(
            param_dtype=self.compute_dtype,
            reduce_dtype=self.precision.gradient_dtype,
            cast_forward_inputs=False,
        )
        reshard = _RESHARD_AFTER_FORWARD[self.parallel_config.sharding_strategy]
        common_options = dict(
            mesh=mesh,
            mp_policy=policy,
            # Every unit, not only the root: a frozen table inside a block is
            # otherwise taken by that block's unit before the root sees it.
            ignored_params={
                p for module in frozen_modules for p in module.parameters()
            },
        )
        # Wrap blocks bottom-up so nested _no_split_modules claim inner blocks
        # before their enclosing modules, and keep root reshard_after_forward=False
        # so FULL_SHARD does not immediately re-gather the root group for backward.
        block_modules = [
            module
            for module in model.modules()
            if module is not model and type(module) in block_classes
        ]
        for module in reversed(block_modules):
            fully_shard(module, reshard_after_forward=reshard, **common_options)
        fully_shard(model, reshard_after_forward=False, **common_options)
        if self.parallel_config.sharding_strategy == "NO_SHARD":
            _broadcast_replicas(model, mesh.get_group("replicate"))
        self.module = model
        self._wrapped = True
        self._wrapper_kind = "fsdp2"
        self.auto_wrap_block_classes = block_classes
        self.ignored_frozen_modules = frozen_modules
        self._trainable = tuple(p for p in target.parameters() if p.requires_grad)
        if self._optimizer_spec is not None:
            self._build_configured_optimizer(target)
        elif self._optimizer_factory is not None:
            self.set_optimizer(self._optimizer_factory(target))
        return self.module

    def _build_configured_optimizer(self, target: nn.Module) -> None:
        from specforge.lr_scheduler import build_lr_scheduler
        from specforge.optimizer import build_optimizer

        if self._max_grad_norm is None:
            raise ValueError("optimizer_spec requires max_grad_norm")
        if self._scheduler_spec is None:
            raise ValueError("optimizer_spec requires scheduler_spec")
        optimizer = build_optimizer(target, self._optimizer_spec)
        self.set_optimizer(
            optimizer, scheduler=build_lr_scheduler(optimizer, self._scheduler_spec)
        )

    def _mesh(self, device: torch.device):
        from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

        group = self.parallel_config.fsdp_process_group
        world = dist.get_world_size(group)
        if self.parallel_config.sharding_strategy == "NO_SHARD":
            if world != dist.get_world_size():
                raise ValueError(
                    "the fsdp2 runtime replicates (NO_SHARD) over every rank of the "
                    f"job; its group has {world} of {dist.get_world_size()}"
                )
            # Replicated: hybrid sharding with one rank per shard group. The
            # meshes create their groups here, on every rank in one order.
            mesh = init_device_mesh(
                device.type, (world, 1), mesh_dim_names=("replicate", "shard")
            )
            owned = []
            seen = {id(dist.group.WORLD)}
            for dim in ("replicate", "shard"):
                sub = mesh.get_group(dim)
                if id(sub) not in seen:
                    seen.add(id(sub))
                    owned.append(sub)
            self._owned_groups = tuple(owned)
            return mesh
        return DeviceMesh.from_group(group or dist.group.WORLD, device.type)

    def backward(self, loss: torch.Tensor, *, is_boundary: bool = True) -> None:
        """Backward one micro-batch; the gradient collective runs at the boundary.

        Between boundaries FSDP2 skips the gradient collective so a window
        reduces once. ``SHARD_GRAD_OP`` and ``NO_SHARD`` keep parameters
        gathered across micro-batches, whereas ``FULL_SHARD`` reshards after
        each backward to preserve its lower parameter-memory peak. Sync and
        reshard flags are restored in ``finally``.
        """
        if is_boundary:
            loss.backward()
            return
        reshard = _RESHARD_AFTER_FORWARD[self.parallel_config.sharding_strategy]
        self.module.set_requires_gradient_sync(False)
        self.module.set_reshard_after_backward(reshard)
        try:
            loss.backward()
        finally:
            self.module.set_requires_gradient_sync(True)
            self.module.set_reshard_after_backward(True)

    def accumulation_context(self, *, is_boundary: bool = True):
        return contextlib.nullcontext()

    def scale_gradients(self, factor: torch.Tensor) -> None:
        if self.module is None:
            raise RuntimeError("scale_gradients called before prepare_model")
        grads = [local_tensor(p.grad) for p in self._trainable if p.grad is not None]
        if grads:
            with torch.no_grad():
                torch._foreach_mul_(grads, factor.to(grads[0].device))

    def _optimizer_state_dict(self) -> Any:
        if self.optimizer is None:
            return None
        return self._torch_optimizer_state_dict(
            include_params=True, format_key=self.FORMAT_KEY
        )

    def _load_optimizer_state_dict(self, state: Mapping[str, Any]) -> None:
        self._load_torch_optimizer_state_dict(
            state, restore_params=True, format_key=self.FORMAT_KEY
        )

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
        if dist.get_rank() != 0:
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
            optimizer = self._optimizer_state_dict()
            rng = self._rng_state()
        except Exception as error:
            raise RankFailure(f"{type(error).__name__}: {error}") from error
        return {"model": model, "optimizer": optimizer, "rng": rng}

    def load_state_dict(self, state: dict) -> None:
        """Restore optimizer and RNG state; reads the reference runtime's too.

        An optimizer state without :attr:`FORMAT_KEY` is the reference
        runtime's (``BF16Optimizer`` under DDP or FSDP1): every rank reads it,
        one parameter at a time, and keeps its own shard. Any retained
        reference layout is released as soon as loading completes.
        """
        try:
            saved = state.get("optimizer")
            if saved is not None and self.FORMAT_KEY not in saved:
                try:
                    # Past the conversion's collectives: a failure is this rank's.
                    converted = self._from_reference(saved)
                except ValueError as error:
                    raise RankFailure(f"{type(error).__name__}: {error}") from error
                state = {**state, "optimizer": converted}
            super().load_state_dict(state)
        finally:
            self._reference_layout = None

    def _from_reference(self, saved: Mapping[str, Any]) -> dict:
        from .reference_state import read_reference_state

        if self._reference_layout is None:
            raise ValueError(
                "resuming from the fsdp runtime's optimizer state requires "
                "retain_reference_layout=True in prepare_model"
            )
        # Align named_parameters() order (used by BF16Optimizer.fp32_params and
        # FSDP1 layout replay) with flattened param_groups order (used by
        # optimizer_state_dict["state"] and _load_torch_optimizer_state_dict).
        by_id = {id(p): i for i, p in enumerate(self._trainable)}
        if self.optimizer is None:
            raise ValueError(
                "reading the fsdp runtime's optimizer state needs this runtime's "
                "optimizer to read it into"
            )
        opt_params = _optimizer_params(self.optimizer)
        if len(opt_params) != len(self._trainable) or any(
            id(p) not in by_id for p in opt_params
        ):
            raise ValueError(
                "optimizer parameters do not match trainable draft parameters"
            )
        opt_to_named = [by_id[id(p)] for p in opt_params]
        saved_groups = saved["optimizer_state_dict"]["param_groups"]
        saved_sizes = [len(g["params"]) for g in saved_groups]
        current_sizes = [len(g["params"]) for g in self.optimizer.param_groups]
        if saved_sizes != current_sizes:
            raise ValueError(
                f"checkpoint param_groups sizes {saved_sizes} do not match "
                f"current optimizer param_groups {current_sizes}"
            )
        named_to_opt = {ni: oi for oi, ni in enumerate(opt_to_named)}
        state_indices = [named_to_opt[i] for i in range(len(self._trainable))]

        group = self.parallel_config.fsdp_process_group or dist.group.WORLD
        replay_group = None
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
                state_indices=state_indices,
                **self._reference_layout,
            )
        finally:
            if replay_group is not None:
                dist.destroy_process_group(replay_group)
        return {
            "optimizer_state_dict": {
                # A parameter the reference never stepped starts afresh.
                "state": {
                    oi: dict(state.states[ni])
                    for oi, ni in enumerate(opt_to_named)
                    if state.states[ni]
                },
                "param_groups": state.param_groups,
            },
            "scheduler_state_dict": state.scheduler_state_dict,
            "lr_scheduler_type": state.lr_scheduler_type,
            "max_grad_norm": state.max_grad_norm,
            "fp32_params": [state.masters[ni] for ni in opt_to_named],
            self.FORMAT_KEY: True,
        }

    def _load_module_state_dict(self, model_state: dict) -> None:
        raise ValueError(
            "the fsdp2 runtime restores optimizer and RNG state; the program's "
            "weights load into it before the wrap"
        )


def _store_in_fp32(
    program: nn.Module, *, frozen_modules, artifact_dtype: torch.dtype
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
    inside_frozen = {id(m) for module in frozen_modules for m in module.modules()}
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
    from torch.distributed.tensor import Shard

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
    source = dist.get_global_rank(group, 0)
    with torch.no_grad():
        for tensor in [*program.parameters(), *program.buffers()]:
            dist.broadcast(local_tensor(tensor), src=source, group=group)


__all__ = ["FSDP2TrainingBackend"]
