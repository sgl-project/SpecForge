# coding=utf-8
"""Expert parallelism (EP) for the routed experts: layout and autograd seams.

The EP axis is carved out of data parallelism, so every rank keeps its own
micro-batch and the dense part of the draft is never computed twice. Inside an
MoE layer the ``ep`` ranks of a group

1. all-gather their tokens (:func:`gather_tokens`),
2. run the replicated router on the gathered tokens,
3. compute only the experts they own, for all gathered tokens, and
4. reduce-scatter the partial outputs (:func:`scatter_outputs`) so each rank
   gets its own tokens back, summed over every expert owner.

Each rank owns a disjoint slice of the stacked expert weights: the ``[E, ...]``
parameters become ``DTensor`` ``Shard(0)`` over the 1-D ``ep`` mesh
(:func:`shard_expert_parameter`). FSDP2 then shards that slice again over the
``efsdp`` mesh through a per-parameter ``shard_placement_fn`` (see
``specforge/training/fsdp2.py``), so FSDP and EP compose and
``get_model_state_dict(full_state_dict=True)`` still yields the full ``[E, ...]``
tensors the checkpoint converters expect.

Both collectives have fixed sizes, so EP adds no device-to-host sync of its
own; the only host sync is reading this rank's slot bounds out of the routing
counts, which the ``sorted_loop`` dispatch already pays.

Gradient semantics: a rank's expert gradient covers the tokens of its whole EP
group. FSDP2 averages it over the ``efsdp`` ranks only, while dense gradients
are averaged over the full data-parallel world, so the FSDP2 backend rescales
expert gradients by ``1 / ep`` at the optimizer boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor


@dataclass(frozen=True)
class ExpertParallelLayout:
    """Which experts this rank owns, and the group it shares tokens with."""

    mesh: DeviceMesh
    rank: int
    size: int
    n_experts: int

    @property
    def n_local_experts(self) -> int:
        return self.n_experts // self.size

    @property
    def expert_start(self) -> int:
        return self.rank * self.n_local_experts

    @property
    def expert_end(self) -> int:
        return self.expert_start + self.n_local_experts

    @property
    def group(self):
        return self.mesh.get_group()


def expert_parallel_layout(ep_mesh: DeviceMesh, n_experts: int) -> ExpertParallelLayout:
    """Resolve the 1-D ``ep`` mesh into this rank's expert slice."""
    if ep_mesh.ndim != 1:
        raise ValueError(f"expert parallelism needs a 1-D mesh, got {ep_mesh.ndim}-D")
    size = ep_mesh.size()
    if size < 1 or n_experts % size:
        raise ValueError(
            f"n_routed_experts={n_experts} is not divisible by "
            f"expert_parallel_size={size}"
        )
    return ExpertParallelLayout(
        mesh=ep_mesh, rank=ep_mesh.get_local_rank(), size=size, n_experts=n_experts
    )


def shard_expert_parameter(param: nn.Parameter, ep_mesh: DeviceMesh) -> nn.Parameter:
    """Replace a full stacked ``[E, ...]`` parameter by this rank's EP shard.

    The result is a ``DTensor`` with ``Shard(0)`` over ``ep_mesh``; the slice is
    scattered from rank 0 of the mesh so every EP group starts from the same
    weights even if the ranks' initializations drifted. Frozen parameters stay
    frozen.
    """
    if isinstance(param, DTensor):
        raise ValueError(
            "parameter is already a DTensor; expert parallelism was applied twice"
        )
    sharded = distribute_tensor(param.detach(), ep_mesh, [Shard(0)])
    return nn.Parameter(sharded, requires_grad=param.requires_grad)


def local_expert_weight(param: torch.Tensor) -> torch.Tensor:
    """This rank's ``[E/ep, ...]`` slice of a (possibly FSDP-unsharded) expert weight."""
    return param.to_local() if isinstance(param, DTensor) else param


class _GatherTokens(torch.autograd.Function):
    """All-gather ``[T, D]`` -> ``[ep * T, D]``; backward reduce-scatters (sum)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group):
        ctx.group = group
        world = dist.get_world_size(group)
        x = x.contiguous()
        out = x.new_empty((world * x.shape[0],) + tuple(x.shape[1:]))
        dist.all_gather_into_tensor(out, x, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        world = dist.get_world_size(ctx.group)
        grad_output = grad_output.contiguous()
        grad = grad_output.new_empty(
            (grad_output.shape[0] // world,) + tuple(grad_output.shape[1:])
        )
        dist.reduce_scatter_tensor(
            grad, grad_output, op=dist.ReduceOp.SUM, group=ctx.group
        )
        return grad, None


class _ScatterOutputs(torch.autograd.Function):
    """Reduce-scatter (sum) ``[ep * T, D]`` -> ``[T, D]``; backward all-gathers."""

    @staticmethod
    def forward(ctx, partial: torch.Tensor, group):
        ctx.group = group
        world = dist.get_world_size(group)
        partial = partial.contiguous()
        out = partial.new_empty((partial.shape[0] // world,) + tuple(partial.shape[1:]))
        dist.reduce_scatter_tensor(out, partial, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        world = dist.get_world_size(ctx.group)
        grad_output = grad_output.contiguous()
        grad = grad_output.new_empty(
            (world * grad_output.shape[0],) + tuple(grad_output.shape[1:])
        )
        dist.all_gather_into_tensor(grad, grad_output, group=ctx.group)
        return grad, None


def gather_tokens(x: torch.Tensor, layout: ExpertParallelLayout) -> torch.Tensor:
    """Every rank's tokens, in EP-rank order; differentiable."""
    return _GatherTokens.apply(x, layout.group)


def scatter_outputs(
    partial: torch.Tensor, layout: ExpertParallelLayout
) -> torch.Tensor:
    """Sum the per-owner partial outputs and return this rank's tokens; differentiable."""
    return _ScatterOutputs.apply(partial, layout.group)


__all__ = [
    "ExpertParallelLayout",
    "expert_parallel_layout",
    "gather_tokens",
    "local_expert_weight",
    "scatter_outputs",
    "shard_expert_parameter",
]
