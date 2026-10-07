"""Composable FSDP2 backend with the same training contract as FSDP1."""

import logging

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from specforge.training.backend import DistributedTrainingBackend


class FSDP2TrainingBackend(DistributedTrainingBackend):
    name = "fsdp2"
    compiled_blocks: int = 0
    fp8_linear_modules: int = 0
    fp8_all_gather: bool = False

    def _prepare_blocks(self, model, block_classes, optimizer_target) -> None:
        if not (self.options.compile_blocks or self.options.fp8_linear):
            return
        targets = self._block_targets(model, block_classes, optimizer_target)
        if not targets:
            raise ValueError(
                "BackendOptions.compile_blocks / fp8_linear found no draft blocks: "
                "the draft advertises no _no_split_modules and has no midlayer"
            )
        if self.options.fp8_linear:
            # Float8 swap first, before compilation and before sharding
            # (torchtitan's order): the compiled graph then holds the float8
            # GEMMs, and Float8Linear's fsdp_pre_all_gather hook only exists
            # under FSDP2.
            self.fp8_linear_modules, self.fp8_all_gather = _convert_blocks_to_float8(
                targets, dp_size=_data_parallel_size(self.parallel_config)
            )
        if self.options.compile_blocks:
            # ``nn.Module.compile`` compiles in place, so the block keeps its class
            # (the ``fully_shard`` boundary below still matches ``block_classes``)
            # and its parameter names (no ``_orig_mod.`` checkpoint prefix). The
            # FSDP2 hooks registered afterwards run inside the compiled call, but
            # Dynamo skips them (``torch._dynamo.config.skip_fsdp_hooks``), so they
            # execute eagerly around the compiled block body. Dynamo starts static
            # and marks shapes dynamic only after a recompilation.
            for module in targets:
                module.compile()
            self.compiled_blocks = len(targets)

    def _after_optimizer_step(self) -> None:
        if self.fp8_linear_modules and self.fp8_all_gather and self._wrapper_kind == "fsdp2":
            from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

            # One all-reduce computes next step's float8 weight scales for
            # the fp8 all-gather instead of a per-parameter amax reduction.
            precompute_float8_dynamic_scale_for_fsdp(self.module)

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
            ignored_params=ignored_params,
        )
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
        return model

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


def _float8_linear_filter(module: nn.Module, fqn: str) -> bool:
    """Trainable linears whose shapes satisfy the float8 GEMM constraints."""
    return (
        type(module) is nn.Linear
        and module.weight.requires_grad
        and module.in_features % 16 == 0
        and module.out_features % 16 == 0
    )


def _data_parallel_size(parallel_config) -> int:
    world = int(getattr(parallel_config, "world_size", 1) or 1)
    inner = 1
    for name in ("tp_size", "sp_ulysses_size", "sp_ring_size"):
        inner *= int(getattr(parallel_config, name, 1) or 1)
    return max(1, world // max(1, inner))


def _convert_blocks_to_float8(blocks, dp_size: int = 1):
    """Swap the draft blocks' trainable linears for ``Float8Linear``.

    Returns ``(converted_count, float8_all_gather)``. FSDP2 shards dim 0 of every
    parameter and torchao's float8 all-gather builds the unsharded view for the
    unpadded size, so a DP size that does not divide a converted weight's rows
    breaks it; then the weights are all-gathered in bf16 and only the GEMMs run
    in float8.
    """
    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "BackendOptions.fp8_linear requires torchao (pip install torchao)"
        ) from exc

    uneven = sorted(
        {
            int(module.out_features)
            for block in blocks
            for name, module in block.named_modules()
            if _float8_linear_filter(module, name) and module.out_features % max(1, dp_size)
        }
    )
    float8_all_gather = not uneven
    if uneven:
        logging.getLogger(__name__).warning(
            "fp8_linear: weight rows %s are not divisible by the data-parallel size %d; "
            "keeping float8 GEMMs but all-gathering the weights in bf16",
            uneven,
            dp_size,
        )
    config = Float8LinearConfig(enable_fsdp_float8_all_gather=float8_all_gather)
    converted = 0
    for block in blocks:
        convert_to_float8_training(
            block, config=config, module_filter_fn=_float8_linear_filter
        )
        converted += sum(
            type(module).__name__ == "Float8Linear" for module in block.modules()
        )
    if converted == 0:
        raise ValueError(
            "BackendOptions.fp8_linear converted no nn.Linear: every trainable "
            "linear in the draft blocks has a dimension not divisible by 16"
        )
    return converted, float8_all_gather
