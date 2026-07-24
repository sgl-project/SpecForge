from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn

ADAMW_OPTIMIZER = "adamw"
MUON_OPTIMIZER = "muon"
SUPPORTED_OPTIMIZERS = (ADAMW_OPTIMIZER, MUON_OPTIMIZER)

# Muon is intended for hidden-layer matrices. These path components identify
# embedding projections and output-like heads that stay on AdamW even when their
# weights are two-dimensional.
DEFAULT_MUON_EXCLUDED_MODULES = (
    "classifier",
    "confidence_head",
    "embed_proj",
    "embed_tokens",
    "lm_head",
    "markov_head",
    "output",
    "score",
)


@dataclass(frozen=True)
class NamedTrainableParameter:
    name: str
    parameter: nn.Parameter
    logical_shape: torch.Size


@dataclass(frozen=True)
class MuonParameterMetadata:
    """Pre-FSDP parameter classification and logical matrix shapes."""

    logical_shapes_by_id: Mapping[int, torch.Size]
    muon_parameter_ids: frozenset[int]


@dataclass(frozen=True)
class MuonParameterPartition:
    """Named trainable parameters assigned to Muon or auxiliary AdamW."""

    muon: tuple[NamedTrainableParameter, ...]
    adamw: tuple[NamedTrainableParameter, ...]

    def names(self) -> dict[str, tuple[str, ...]]:
        return {
            MUON_OPTIMIZER: tuple(item.name for item in self.muon),
            ADAMW_OPTIMIZER: tuple(item.name for item in self.adamw),
        }


def partition_parameters_for_muon(
    model: nn.Module,
    *,
    excluded_module_names: Sequence[str] = DEFAULT_MUON_EXCLUDED_MODULES,
    metadata: MuonParameterMetadata | None = None,
) -> MuonParameterPartition:
    """Split trainable parameters into hidden matrices and auxiliary tensors.

    Module ownership is checked rather than tensor rank alone because
    embeddings and output heads are also commonly two-dimensional.
    """

    modules = dict(model.named_modules())
    excluded_names = frozenset(excluded_module_names)
    muon_parameters: list[NamedTrainableParameter] = []
    adamw_parameters: list[NamedTrainableParameter] = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if metadata is None:
            module_name, separator, parameter_name = name.rpartition(".")
            if not separator:
                module_name = ""
                parameter_name = name
            owner = modules.get(module_name)
            module_path = frozenset(part for part in module_name.split(".") if part)
            use_muon = (
                isinstance(owner, nn.Linear)
                and parameter_name == "weight"
                and parameter.ndim == 2
                and module_path.isdisjoint(excluded_names)
            )
            logical_shape = parameter.shape
        else:
            parameter_id = id(parameter)
            try:
                logical_shape = metadata.logical_shapes_by_id[parameter_id]
            except KeyError as error:
                raise ValueError(
                    f"Parameter {name!r} was not present when Muon metadata "
                    "was captured before distributed wrapping"
                ) from error
            use_muon = parameter_id in metadata.muon_parameter_ids

        item = NamedTrainableParameter(
            name=name,
            parameter=parameter,
            logical_shape=torch.Size(logical_shape),
        )
        if use_muon:
            muon_parameters.append(item)
        else:
            adamw_parameters.append(item)

    return MuonParameterPartition(
        muon=tuple(muon_parameters),
        adamw=tuple(adamw_parameters),
    )


def capture_muon_parameter_metadata(
    model: nn.Module,
    *,
    excluded_module_names: Sequence[str] = DEFAULT_MUON_EXCLUDED_MODULES,
) -> MuonParameterMetadata:
    """Capture Muon eligibility before FSDP exposes flattened local shards."""

    partition = partition_parameters_for_muon(
        model, excluded_module_names=excluded_module_names
    )
    trainable_parameters = partition.muon + partition.adamw
    return MuonParameterMetadata(
        logical_shapes_by_id={
            id(item.parameter): item.logical_shape for item in trainable_parameters
        },
        muon_parameter_ids=frozenset(id(item.parameter) for item in partition.muon),
    )


def zeropower_via_newton_schulz(
    update: torch.Tensor,
    *,
    ns_steps: int,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Match the Newton--Schulz transform used by ``torch.optim.Muon``."""

    if update.ndim != 2:
        raise ValueError(f"Muon update must be 2-D, got shape {tuple(update.shape)}")

    orthogonalized = update.to(torch.bfloat16)
    transposed = orthogonalized.size(0) > orthogonalized.size(1)
    if transposed:
        orthogonalized = orthogonalized.T
    orthogonalized.div_(orthogonalized.norm().clamp(min=eps))

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(ns_steps):
        gram_matrix = orthogonalized @ orthogonalized.T
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
        )
        orthogonalized = torch.addmm(
            orthogonalized, gram_update, orthogonalized, beta=a
        )
    return orthogonalized.T if transposed else orthogonalized


def adjust_muon_learning_rate(
    learning_rate: float,
    adjust_lr_fn: str,
    logical_shape: torch.Size,
) -> float:
    rows, columns = logical_shape
    if adjust_lr_fn == "original":
        ratio = math.sqrt(max(1, rows / columns))
    elif adjust_lr_fn == "match_rms_adamw":
        ratio = 0.2 * math.sqrt(max(rows, columns))
    else:
        raise ValueError(f"Unsupported Muon learning-rate adjustment: {adjust_lr_fn}")
    return learning_rate * ratio


@dataclass(frozen=True)
class _ShardLayout:
    sizes: tuple[int, ...]
    offset: int


class FSDPShardedMuon(torch.optim.Optimizer):
    """Muon over FSDP1 local shards with sharded persistent state.

    Newton--Schulz needs a logical 2-D update. This optimizer keeps FP32
    momentum sharded, gathers one BF16 update at a time, applies the same
    transform as native Muon, and writes only the local slice back.
    """

    def __init__(
        self,
        params: Sequence[torch.Tensor],
        logical_shapes: Sequence[torch.Size],
        *,
        lr: float,
        weight_decay: float,
        momentum: float,
        nesterov: bool,
        ns_steps: int,
        adjust_lr_fn: str,
    ) -> None:
        parameters = list(params)
        if len(parameters) != len(logical_shapes):
            raise ValueError("Every sharded Muon parameter needs a logical shape")
        if lr < 0:
            raise ValueError(f"Muon learning rate must be non-negative, got {lr}")
        if weight_decay < 0:
            raise ValueError(
                f"Muon weight decay must be non-negative, got {weight_decay}"
            )
        if not 0 <= momentum < 1:
            raise ValueError(f"Muon momentum must be in [0, 1), got {momentum}")
        if not 0 < ns_steps < 100:
            raise ValueError(f"Muon ns_steps must be in [1, 99], got {ns_steps}")
        if adjust_lr_fn not in ("original", "match_rms_adamw"):
            raise ValueError(
                f"Unsupported Muon learning-rate adjustment: {adjust_lr_fn}"
            )

        defaults = {
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "adjust_lr_fn": adjust_lr_fn,
        }
        super().__init__(parameters, defaults)
        self._logical_shapes = {
            id(parameter): torch.Size(shape)
            for parameter, shape in zip(parameters, logical_shapes)
        }
        self._shard_layouts: dict[int, _ShardLayout] = {}
        self._process_group = None
        self._process_group_configured = False

    def configure_process_group(self, process_group=None) -> None:
        """Bind the FSDP group and discover each rank-local shard layout."""

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Sharded Muon requires an initialized process group")
        if dist.get_world_size(group=process_group) <= 1:
            raise RuntimeError("Sharded Muon requires more than one process")

        self._process_group = process_group
        self._process_group_configured = True
        self._shard_layouts = {}
        for group in self.param_groups:
            for parameter in group["params"]:
                logical_shape = self._logical_shapes[id(parameter)]
                self._shard_layouts[id(parameter)] = self._build_shard_layout(
                    parameter, logical_shape
                )

    def _build_shard_layout(
        self, parameter: torch.Tensor, logical_shape: torch.Size
    ) -> _ShardLayout:
        local_size = torch.tensor(
            parameter.numel(), dtype=torch.int64, device=parameter.device
        )
        world_size = dist.get_world_size(group=self._process_group)
        gathered_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(gathered_sizes, local_size, group=self._process_group)
        sizes = tuple(int(size.item()) for size in gathered_sizes)
        expected_numel = math.prod(logical_shape)
        if sum(sizes) != expected_numel:
            raise RuntimeError(
                "FSDP Muon shards do not reconstruct the logical matrix: "
                f"shape={tuple(logical_shape)}, shard_sizes={sizes}"
            )
        group_rank = dist.get_rank(group=self._process_group)
        return _ShardLayout(sizes=sizes, offset=sum(sizes[:group_rank]))

    def _all_gather_flat(
        self, local_tensor: torch.Tensor, layout: _ShardLayout
    ) -> torch.Tensor:
        max_size = max(layout.sizes)
        padded = local_tensor.new_zeros(max_size)
        padded[: local_tensor.numel()].copy_(local_tensor.reshape(-1))
        gathered = [torch.empty_like(padded) for _ in layout.sizes]
        dist.all_gather(gathered, padded, group=self._process_group)
        return torch.cat(
            [tensor[:size] for tensor, size in zip(gathered, layout.sizes)]
        )

    @torch.no_grad()
    def step(self, closure=None):
        if not self._process_group_configured:
            raise RuntimeError(
                "Sharded Muon process group was not configured by the training backend"
            )

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            learning_rate = float(group["lr"])
            for parameter in group["params"]:
                has_local_grad = torch.tensor(
                    parameter.grad is not None,
                    dtype=torch.uint8,
                    device=parameter.device,
                )
                dist.all_reduce(
                    has_local_grad,
                    op=dist.ReduceOp.MAX,
                    group=self._process_group,
                )
                if not bool(has_local_grad.item()):
                    continue

                local_gradient = (
                    torch.zeros_like(parameter)
                    if parameter.grad is None
                    else parameter.grad
                )
                if local_gradient.shape != parameter.shape:
                    raise RuntimeError(
                        "FSDP Muon gradient and local parameter shard differ: "
                        f"gradient={tuple(local_gradient.shape)}, "
                        f"parameter={tuple(parameter.shape)}"
                    )

                state = self.state[parameter]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(local_gradient)
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.lerp_(local_gradient, 1 - group["momentum"])
                local_update = (
                    local_gradient.lerp(momentum_buffer, group["momentum"])
                    if group["nesterov"]
                    else momentum_buffer
                )

                logical_shape = self._logical_shapes[id(parameter)]
                layout = self._shard_layouts[id(parameter)]
                full_update = self._all_gather_flat(
                    local_update.to(torch.bfloat16), layout
                ).reshape(logical_shape)
                orthogonalized = zeropower_via_newton_schulz(
                    full_update, ns_steps=group["ns_steps"]
                ).reshape(-1)
                local_orthogonalized = orthogonalized.narrow(
                    0, layout.offset, parameter.numel()
                )
                adjusted_lr = adjust_muon_learning_rate(
                    learning_rate, group["adjust_lr_fn"], logical_shape
                )

                parameter.mul_(1 - learning_rate * group["weight_decay"])
                parameter.add_(local_orthogonalized, alpha=-adjusted_lr)
        return loss


__all__ = [
    "ADAMW_OPTIMIZER",
    "DEFAULT_MUON_EXCLUDED_MODULES",
    "FSDPShardedMuon",
    "MUON_OPTIMIZER",
    "MuonParameterMetadata",
    "MuonParameterPartition",
    "NamedTrainableParameter",
    "SUPPORTED_OPTIMIZERS",
    "capture_muon_parameter_metadata",
    "partition_parameters_for_muon",
]
