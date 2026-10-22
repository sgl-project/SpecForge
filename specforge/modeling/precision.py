"""Checkpoint tensors own their storage dtype; defaults apply to new tensors.

This module does not choose operator compute precision. Model implementations
must still define dtype transitions between modules, just as native mixed-
precision architectures do. No loaded tensor is rounded to a global default.
"""

import torch


class NativeDtypeLinear(torch.nn.Linear):
    """Project in checkpoint weight dtype and return native-dtype base logits.

    Output consumers own the FP32 conversion: ordinary target logits convert
    immediately; DSpark first adds its Markov correction, matching SGLang's
    dense DSpark path. Upcasting here would change BF16 addition rounding.
    """

    def forward(self, inputs):
        return super().forward(inputs.to(dtype=self.weight.dtype))


class CheckpointDtypeMixin:
    def _get_dtype_plan(self, dtype):
        """Tell the pinned Transformers loader not to cast checkpoint tensors.

        Transformers 5.12 uses None in a matched dtype-plan entry to load the
        source tensor unchanged. ``dtype`` still controls initialization of
        missing/new weights. This also works with sharded safetensors and its
        index: there is no single-file assumption or eager header scan here.
        """
        return {"*": None}


@torch.no_grad()
def copy_parameter_preserving_dtype(parameter, source):
    """Keep parameter identity, device and trainability, adopting source dtype."""
    if parameter.shape != source.shape:
        raise ValueError(
            f"checkpoint shape {source.shape} does not match {parameter.shape}"
        )
    parameter.data = source.detach().to(device=parameter.device, copy=True)


def load_state_dict_preserving_dtype(model, state_dict, *, strict=True):
    """Load before optimizer creation without rounding through destination dtype.

    Child pre-hooks see names after existing model normalization hooks have run.
    Unlike assign=True, this retains parameter objects, tied aliases, devices,
    and requires_grad. Missing parameters retain their initialized dtype.
    """
    seen = {}

    def prepare(module, state, prefix, *_args):
        for name, destination in (
            *module._parameters.items(),
            *module._buffers.items(),
        ):
            key = prefix + name
            source = state.get(key)
            if destination is None or not isinstance(source, torch.Tensor):
                continue
            previous = seen.setdefault(id(destination), source.dtype)
            if previous != source.dtype:
                raise ValueError(f"tied checkpoint parameter {key} has conflicting dtypes")
            # Let load_state_dict report shape errors with the usual key context.
            if destination.shape == source.shape and destination.dtype != source.dtype:
                destination.data = destination.to(dtype=source.dtype)

    handles = [
        module.register_load_state_dict_pre_hook(prepare)
        for module in model.modules()
    ]
    try:
        return model.load_state_dict(state_dict, strict=strict)
    finally:
        for handle in handles:
            handle.remove()


def mixed_dtype_wrap_modules(model, ignored_modules=()):
    """Find homogeneous parameter owners when FSDP1 cannot flatten the root.

    Only mixed models need these extra wrapping boundaries. Homogeneous models
    retain their existing transformer-block wrapping. A module that directly
    owns incompatible parameter dtypes cannot be flattened by FSDP1; reject it
    explicitly instead of silently rewriting weights.
    """
    ignored = {id(child) for parent in ignored_modules for child in parent.modules()}
    modules = [module for module in model.modules() if id(module) not in ignored]
    dtypes = {p.dtype for module in modules for p in module.parameters(recurse=False)}
    if len(dtypes) <= 1:
        return set()
    boundaries = set()
    for module in modules:
        params = list(module.parameters(recurse=False))
        if len({p.dtype for p in params}) > 1:
            raise ValueError(
                f"FSDP1 cannot flatten mixed dtypes directly owned by {type(module).__name__}; "
                "use NO_SHARD/DDP or separate these parameters into modules. "
                "Checkpoint dtypes will not be overwritten."
            )
        if params and module is not model:
            boundaries.add(module)
    return boundaries
