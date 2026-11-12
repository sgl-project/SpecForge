"""Validate owned export state; target tables are read only to check legacy copies."""

from dataclasses import dataclass
from pathlib import Path

import torch

from specforge.checkpoint import make_weight_plan
from specforge.checkpoint.tensors import TensorInfo, TensorRef
from specforge.loading import check_bound_copies, read_draft
from specforge.modeling import build_draft_model


@dataclass(frozen=True)
class _BindingMetadata:
    metadata: dict[str, TensorInfo]


def read_export(checkpoint, *, draft_config=None, target=None, vocab_mapping=None):
    path = Path(str(checkpoint).removeprefix("file://"))
    if path.is_dir() and not any(
        (path / name).exists() for name in ("config.json", "training_state.pt")
    ):
        from specforge.training.checkpoint import CheckpointManager

        path = Path(CheckpointManager.resolve_resume_dir(str(path)))
    directory = path.parent if path.is_file() else path
    supplementary = draft_config if not (directory / "config.json").exists() else None
    checkpoint = read_draft(
        path if path.exists() else checkpoint,
        target=target,
        legacy_config=supplementary,
    )
    if draft_config is not None and supplementary is None:
        raise ValueError("checkpoint already has config.json; omit --draft-config")
    if vocab_mapping is not None:
        raise ValueError(
            "restore vocabulary mappings before export; export never replaces checkpoint tensors"
        )
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
    # the artifact drops legacy copies of bound tables, so they must equal the target
    check_bound_copies(checkpoint, target=target)
    return checkpoint, state, type(preview)
