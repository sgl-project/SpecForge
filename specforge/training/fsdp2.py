"""Composable FSDP2 backend with the same training contract as FSDP1."""

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from specforge.training.backend import DistributedTrainingBackend


def _configure_bucketed_compile(buckets: int) -> None:
    """Let Dynamo hold one static graph per length bucket.

    Each block is traced twice per bucket (the first block's input does not
    require grad, the others' does), so the per-frame recompile limit must be at
    least ``2 * buckets``; past the limit Dynamo would silently fall back to
    eager for new shapes. Both the current and the pre-2.7 config names are set.
    """
    import torch._dynamo

    cfg = torch._dynamo.config
    needed = 2 * int(buckets) + 4
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(cfg, name) and int(getattr(cfg, name)) < needed:
            setattr(cfg, name, needed)
    for name in ("accumulated_recompile_limit", "accumulated_cache_size_limit"):
        if hasattr(cfg, name) and int(getattr(cfg, name)) < 32 * needed:
            setattr(cfg, name, 32 * needed)


class FSDP2TrainingBackend(DistributedTrainingBackend):
    name = "fsdp2"
    compiled_blocks: int = 0

    def _prepare_blocks(self, model, block_classes, optimizer_target) -> None:
        if not self.options.compile_blocks:
            return
        targets = self._block_targets(model, block_classes, optimizer_target)
        if not targets:
            raise ValueError(
                "BackendOptions.compile_blocks found no draft blocks to compile: "
                "the draft advertises no _no_split_modules and has no midlayer"
            )
        # ``nn.Module.compile`` compiles in place, so the block keeps its class
        # (the ``fully_shard`` boundary below still matches ``block_classes``)
        # and its parameter names (no ``_orig_mod.`` checkpoint prefix). The
        # FSDP2 hooks registered afterwards run inside the compiled call, but
        # Dynamo skips them (``torch._dynamo.config.skip_fsdp_hooks``), so they
        # execute eagerly around the compiled block body. Dynamo starts static
        # and marks shapes dynamic only after a recompilation; with length
        # buckets every bucket must stay a static graph instead.
        buckets = int(getattr(self.options, "compile_shape_buckets", 0) or 0)
        if buckets > 1:
            _configure_bucketed_compile(buckets)
        for module in targets:
            if buckets > 1:
                module.compile(dynamic=False)
            else:
                module.compile()
        self.compiled_blocks = len(targets)

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
