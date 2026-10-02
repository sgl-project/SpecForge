"""Composable FSDP2 backend with the same training contract as FSDP1."""

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from specforge.training.backend import DistributedTrainingBackend


class FSDP2TrainingBackend(DistributedTrainingBackend):
    name = "fsdp2"

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
        )
        ignored_params = {
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
        kwargs = dict(
            mesh=mesh,
            reshard_after_forward=pc.sharding_strategy == "FULL_SHARD",
            ignored_params=ignored_params,
        )
        # Bottom-up application preserves the FSDP1 block/root boundaries.
        # Leave non-block draft parameters on the composite root: some losses
        # call custom draft head methods after draft_model.forward has returned.
        for module in reversed(list(model.modules())):
            if module is not model and type(module) in block_classes:
                fully_shard(
                    module,
                    mp_policy=MixedPrecisionPolicy(
                        param_dtype=pc.param_dtype, cast_forward_inputs=False
                    ),
                    **kwargs,
                )
        fully_shard(
            model,
            mp_policy=MixedPrecisionPolicy(param_dtype=pc.param_dtype),
            **kwargs,
        )
        return model

    def backward(self, loss: torch.Tensor, *, is_boundary: bool = True) -> None:
        if self._wrapper_kind != "fsdp2":
            return super().backward(loss, is_boundary=is_boundary)
        self.module.set_requires_gradient_sync(is_boundary)
        try:
            loss.backward()
        finally:
            self.module.set_requires_gradient_sync(True)

    def _sharded_model_state_dict(self) -> dict:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        # All ranks participate; full_state_dict + cpu_offload returns the
        # gathered ordinary tensors on rank zero and an empty dict elsewhere.
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
