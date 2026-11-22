"""Validate canonical owned export state without reading target weights."""

from dataclasses import dataclass

import torch

from specforge.checkpoint import make_weight_plan
from specforge.checkpoint.tensors import TensorInfo, TensorRef
from specforge.loading import read_draft
from specforge.modeling import build_draft_model


@dataclass(frozen=True)
class _BindingMetadata:
    metadata: dict[str, TensorInfo]


def read_export(checkpoint, *, target=None):
    from specforge.checkpoint.resolve import resolve_checkpoint_path

    path = resolve_checkpoint_path(checkpoint)
    checkpoint = read_draft(path, target=target)
    # Meta construction validates every tensor name and shape. Bound modules
    # are intentionally absent from an artifact; no target tensor is read here.
    preview = build_draft_model(checkpoint.config, device="meta")
    binding_store = _BindingMetadata(
        {
            name: TensorInfo(
                tuple(preview.get_submodule(name).weight.shape), torch.float32
            )
            for name in checkpoint.config.target_bindings
        }
    )
    bindings = {name: TensorRef(binding_store, name) for name in binding_store.metadata}
    plan = make_weight_plan(preview, checkpoint.weights, bindings=bindings)
    state, seen = {}, set()
    for name, ref in plan.weights:
        if ref not in seen:
            state[name] = ref.read().contiguous()
            seen.add(ref)
    preview.validate_export_state(state)
    return checkpoint, state, type(preview)
