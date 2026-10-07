"""Composable FSDP2 backend with the same training contract as FSDP1.

Expert parallelism (``training.expert_parallel_size > 1``) lives here as well:
each MoE layer's experts are sliced over the ``ep`` mesh before wrapping, and a
per-parameter ``shard_placement_fn`` lets FSDP2 shard the slices over the
``efsdp`` ranks while every other parameter stays on the full data-parallel
mesh. Gradient accumulation, grad-norm reduction over local shards and the
full-state-dict checkpoint path are unchanged.
"""

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import Shard

from specforge.training.backend import DistributedTrainingBackend
from specforge.training.params import local_tensor


class FSDP2TrainingBackend(DistributedTrainingBackend):
    name = "fsdp2"

    def __init__(self, parallel_config, *, optimizer_factory=None) -> None:
        super().__init__(parallel_config, optimizer_factory=optimizer_factory)
        self._expert_params: list = []
        self._expert_grad_scale = 1.0

    def _shard_model(self, model, block_classes, ignored_frozen_modules):
        pc = self.parallel_config
        if pc.sharding_strategy not in ("FULL_SHARD", "SHARD_GRAD_OP"):
            raise ValueError(f"unsupported FSDP2 sharding: {pc.sharding_strategy!r}")
        device = next(model.parameters()).device
        # The existing FSDP group spans WORLD, including sequence-parallel ranks.
        # A draft-DP-only mesh would silently change the training reduction.
        mesh = DeviceMesh.from_group(
            pc.fsdp_process_group or torch.distributed.group.WORLD,
            device_type=device.type,
            mesh_dim_names=("fsdp",),
        )
        kwargs = dict(mesh=mesh)
        if pc.expert_parallel_size > 1:
            # Slice the experts first: the frozen/ignored parameter sets below
            # must refer to the sliced parameters.
            kwargs["shard_placement_fn"] = self._apply_expert_parallel(model)
        kwargs["ignored_params"] = {
            p for module in ignored_frozen_modules for p in module.parameters()
        }
        # FSDP1 keeps floating buffers (e.g. RoPE) in FP32. FSDP2 does not
        # manage buffer precision, so preserve that policy explicitly.
        for module in model.modules():
            if module in ignored_frozen_modules:
                continue
            for name, buffer in module.named_buffers(recurse=False):
                if buffer.is_floating_point():
                    setattr(module, name, buffer.float())
        # Bottom-up application preserves the FSDP1 block/root boundaries.
        # Leave non-block draft parameters on the composite root: some losses
        # call custom draft head methods after draft_model.forward has returned.
        for module in reversed(list(model.modules())):
            if module is not model and type(module) in block_classes:
                fully_shard(
                    module,
                    reshard_after_forward=pc.sharding_strategy == "FULL_SHARD",
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=pc.param_dtype, cast_forward_inputs=False
                    ),
                    **kwargs,
                )
        fully_shard(
            model,
            # Like FSDP1, reuse the root's full parameters in backward even
            # under FULL_SHARD. Child blocks still reshard after forward.
            reshard_after_forward=False,
            mp_policy=MixedPrecisionPolicy(param_dtype=pc.param_dtype),
            **kwargs,
        )
        if pc.expert_parallel_size > 1:
            # fully_shard replaced every parameter with its FSDP-sharded DTensor;
            # gradient rescaling must address those, not the pre-wrap slices.
            from specforge.modeling.draft.moe import iter_moe_layers

            self._expert_params = [
                parameter
                for layer in iter_moe_layers(model)
                for parameter in layer.experts.parameters()
                if parameter.requires_grad
            ]
        return model

    def _apply_expert_parallel(self, model):
        """Slice MoE experts over the ``ep`` mesh; return FSDP2's placement fn.

        Routed experts end up ``Shard(0)`` over ``ep`` (their own slice) and
        ``Shard(0)`` over ``efsdp`` (FSDP across the ranks that own the same
        slice); dense parameters keep the default placement on the full mesh.
        """
        from torch.distributed.fsdp._fully_shard._fsdp_common import (
            ShardPlacementResult,
        )
        from torch.distributed.fsdp._fully_shard._fsdp_init import _get_mesh_info

        from specforge.modeling.draft.moe import apply_expert_parallel, iter_moe_layers

        pc = self.parallel_config
        ep_mesh = pc.draft_ep_mesh
        if ep_mesh is None:
            raise RuntimeError(
                "training.expert_parallel_size > 1 but init_distributed built no "
                "expert-parallel mesh"
            )
        if apply_expert_parallel(model, ep_mesh["ep"]) == 0:
            raise ValueError(
                "training.expert_parallel_size > 1 requires a MoE draft "
                "(n_routed_experts > 0 in the draft JSON)"
            )
        expert_params = {
            parameter
            for layer in iter_moe_layers(model)
            for parameter in layer.experts.parameters()
        }
        expert_mesh_info = _get_mesh_info(ep_mesh["efsdp"])

        def shard_placement_fn(param):
            if param in expert_params:
                return ShardPlacementResult(
                    placement=Shard(0), mesh_info=expert_mesh_info
                )
            return None

        # FSDP2 averages every parameter's gradient over its own reduce-scatter
        # group: ``efsdp`` for the experts, the full mesh for dense parameters.
        # An expert gradient already sums its whole EP group's tokens, so bring
        # it to the dense per-token scale before clipping and the optimizer.
        self._expert_grad_scale = 1.0 / pc.expert_parallel_size
        return shard_placement_fn

    def backward(self, loss: torch.Tensor, *, is_boundary: bool = True) -> None:
        if self._wrapper_kind != "fsdp2":
            return super().backward(loss, is_boundary=is_boundary)
        self.module.set_requires_gradient_sync(is_boundary)
        # FSDP1 SHARD_GRAD_OP retains parameters across no_sync micro-steps.
        # Retain them until the optimizer boundary to avoid re-gathering on
        # every forward; FULL_SHARD still releases them after each backward.
        self.module.set_reshard_after_backward(
            is_boundary or self.parallel_config.sharding_strategy != "SHARD_GRAD_OP"
        )
        try:
            loss.backward()
        finally:
            self.module.set_requires_gradient_sync(True)
            self.module.set_reshard_after_backward(True)
        if is_boundary and self._expert_params:
            # Reduced expert gradients exist only after the boundary backward.
            with torch.no_grad():
                for parameter in self._expert_params:
                    if parameter.grad is not None:
                        local_tensor(parameter.grad).mul_(self._expert_grad_scale)

    def _sharded_model_state_dict(self) -> dict:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        # All ranks participate; full_state_dict + cpu_offload returns the
        # gathered ordinary tensors on rank zero and an empty dict elsewhere.
        # Expert-parallel DTensor slices gather to the full [E, ...] tensors.
        return get_model_state_dict(
            self.module,
            options=StateDictOptions(full_state_dict=True, cpu_offload=True),
        )

    def _load_sharded_model_state_dict(self, model_state: dict) -> None:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_model_state_dict,
        )

        set_model_state_dict(
            self.module,
            model_state,
            options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True),
        )
