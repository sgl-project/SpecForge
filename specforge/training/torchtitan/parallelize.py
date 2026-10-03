"""Model-specific TP, checkpointing, compile and FSDP for DFlash-family drafts."""

from __future__ import annotations

import torch
from torch.distributed._functional_collectives import AsyncCollectiveTensor, wait_tensor
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.overrides import TorchFunctionMode
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder


class HeterogeneousGradientNorms(TorchFunctionMode):
    """Allow Titan's native clipping to combine DP and DP+TP gradient norms.

    Unsharded auxiliary heads use a DP mesh; TP projections use a DP+TP mesh.
    PyTorch 2.14 can compute their individual DTensor norms, but ``aten.stack``
    raises when those scalar norms have different meshes. Resolve each mesh's norm vector
    before the native clipper combines it. Parameter/gradient placements and
    the optimizer are untouched. Only norm-partial scalar stacks are handled.
    """

    def __init__(self):
        super().__init__()
        self._mixed_norms = False

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.stack and args:
            values = args[0]
            if values and all(
                isinstance(value, DTensor)
                and value.ndim == 0
                and any(type(p).__name__ == "_NormPartial" for p in value.placements)
                for value in values
            ):
                groups = {}
                for value in values:
                    groups.setdefault(value.device_mesh, []).append(value)
                if len(groups) > 1:
                    self._mixed_norms = True
                    # Each vector's DTensor reduction accounts for its own
                    # sharded and replicated dimensions exactly once.
                    return torch.cat(
                        [torch.stack(group).full_tensor() for group in groups.values()]
                    )
        if self._mixed_norms and func is torch._foreach_mul_ and len(args) == 2:
            values, coefficient = args
            if (
                values
                and all(isinstance(value, DTensor) for value in values)
                and isinstance(coefficient, torch.Tensor)
                and not isinstance(coefficient, DTensor)
                and coefficient.ndim == 0
            ):
                groups = {}
                for value in values:
                    groups.setdefault(value.device_mesh, []).append(value)
                if len(groups) > 1:
                    # The native clipper applies one already-global scalar.
                    # A foreach op must also stay within a single DTensor mesh.
                    for group in groups.values():
                        torch._foreach_mul_(group, coefficient)
                    return tuple(values)
        return func(*args, **kwargs)


def group_optimizer_parameters_by_mesh(optimizers, model_parts, parallel_dims):
    """Keep native fused AdamW tensor lists within one DTensor mesh per group."""
    del model_parts
    if not parallel_dims.tp_enabled:
        return
    for optimizer in optimizers:
        groups = []
        for group in optimizer.param_groups:
            by_mesh = {}
            for index, parameter in enumerate(group["params"]):
                mesh = parameter.device_mesh if isinstance(parameter, DTensor) else None
                by_mesh.setdefault(mesh, []).append(index)
            for indices in by_mesh.values():
                split = {**group, "params": [group["params"][i] for i in indices]}
                if "param_names" in group:
                    split["param_names"] = [group["param_names"][i] for i in indices]
                groups.append(split)
        optimizer.param_groups = groups


def _materialize_rowwise_output(module, inputs, output):
    # RowwiseParallel can return a pending all-reduce wrapper. DFlash2 passes
    # this result directly to its Triton convolution after only view/contiguous
    # operations, which do not necessarily wait. Triton's raw-pointer access
    # cannot dispatch through AsyncCollectiveTensor; finish the collective at
    # this boundary and expose its real tensor without detaching autograd.
    if isinstance(output, AsyncCollectiveTensor):
        return wait_tensor(output)
    return output


def _apply_tensor_parallel(draft, tp_mesh) -> None:
    config = draft.config
    degree = tp_mesh.size()
    if draft.attention_mode != "gqa":
        raise ValueError("TorchTitan TP currently supports DFlash GQA attention only")
    if config.attention_bias:
        raise ValueError("TorchTitan TP currently requires attention_bias=false")
    for field in ("num_attention_heads", "num_key_value_heads", "intermediate_size"):
        if getattr(config, field) % degree:
            raise ValueError(
                f"{field} must be divisible by tensor parallel degree {degree}"
            )
    for _, layer in draft.layers.items():
        parallelize_module(
            layer,
            tp_mesh,
            {
                "self_attn.q_proj": ColwiseParallel(),
                "self_attn.k_proj": ColwiseParallel(),
                "self_attn.v_proj": ColwiseParallel(),
                "self_attn.o_proj": RowwiseParallel(
                    input_layouts=Shard(-1), output_layouts=Replicate()
                ),
                # Norm weights are shared across heads. A head-sharded input
                # produces partial weight gradients that DTensor must reduce.
                "self_attn.q_norm": SequenceParallel(
                    sequence_dim=2, use_local_output=True
                ),
                "self_attn.k_norm": SequenceParallel(
                    sequence_dim=2, use_local_output=True
                ),
                "mlp.gate_proj": ColwiseParallel(),
                "mlp.up_proj": ColwiseParallel(),
                "mlp.down_proj": RowwiseParallel(
                    input_layouts=Shard(-1), output_layouts=Replicate()
                ),
            },
        )
        layer.self_attn.o_proj.register_forward_hook(_materialize_rowwise_output)
        layer.mlp.down_proj.register_forward_hook(_materialize_rowwise_output)


def parallelize_dflash(
    model,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    ac_config,
    dump_folder: str,
):
    """TorchTitan ModelSpec entry point; unsupported modes fail before mutation."""
    if parallelism.spmd_backend != "partial_dtensor":
        raise ValueError("DFlash currently requires the partial_dtensor SPMD backend")
    if parallel_dims.pp_enabled and not getattr(model, "_is_pipeline_stage", False):
        raise ValueError("DFlash pipeline parallelism is not implemented")
    if parallel_dims.ep_enabled:
        raise ValueError("Dense DFlash models do not support expert parallelism")
    if getattr(parallelism, "enable_sequence_parallel", False):
        raise ValueError("DFlash sequence parallelism is not implemented")
    draft = model.draft_model
    if parallel_dims.tp_enabled:
        _apply_tensor_parallel(draft, parallel_dims.get_dense_tp_mesh())
    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(draft)
    if compile_config.enable and "model" in compile_config.components:
        apply_compile(draft, compile_config=compile_config, parallel_dims=parallel_dims)

    dp_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    apply_fsdp_to_decoder(
        draft,
        parallel_dims.get_mesh(dp_names),
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )
    # DFlash2 selector and DSpark heads are consumed by the objective after
    # draft.forward() returns. Keep their root group materialized until BWD.
    draft.set_reshard_after_forward(False, recurse=False)
    # Loss sums span DP examples and CP blocks, never TP replicas. In this
    # model CP partitions whole draft blocks and replicates teacher context.
    loss_mesh = parallel_dims.get_optional_mesh("loss", include_singleton_axes=True)
    cp_mesh = parallel_dims.get_optional_mesh("cp", include_singleton_axes=True)
    model.training_model.objective_process_group = loss_mesh.get_group()
    model.training_model.objective_context_group = cp_mesh.get_group()
    return model
