"""Persist a complete canonical model artifact independently of training state."""

import json
import os
import tempfile
from pathlib import Path

import torch.distributed as dist


def read_artifact_metadata(config):
    """Validate directory-format metadata without treating it as model structure."""
    if "specforge_artifact" not in config:
        return {}  # Ordinary HF save_pretrained directories have no marker.
    metadata = config["specforge_artifact"]
    if not isinstance(metadata, dict):
        raise ValueError("artifact metadata must be an object")
    if metadata.get("format") == "sglang":
        raise ValueError(
            "SGLang artifacts are serving-only; load the canonical checkpoint"
        )
    if (
        metadata.get("format") not in ("specforge", "hf")
        or metadata.get("version") != 1
    ):
        raise ValueError(f"unsupported artifact metadata: {metadata}")
    return metadata


def save_draft(model, path, *, max_shard_size="5GB"):
    """Write config, owned weights and format metadata to a new directory.

    Target-bound modules exclude themselves from state_dict. Tied parameters
    are stored once; their relationship comes from the canonical configuration.
    Existing nonempty directories are never mixed with new checkpoint shards.
    """
    from specforge.modeling.draft.hf_config import SpecForgeDraftConfig

    if not isinstance(model.config, SpecForgeDraftConfig):
        raise TypeError("save_draft requires a canonical draft model")
    if dist.is_initialized() and dist.get_rank() != 0:
        raise ValueError("save_draft must be called on global rank 0")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise FileExistsError(f"checkpoint directory is not empty: {destination}")
    state = model.state_dict()
    for key, value in state.items():
        if value.is_meta or hasattr(value, "to_local"):
            raise ValueError(f"{key}: save requires materialized, gathered tensors")
    with tempfile.TemporaryDirectory(prefix=".draft-", dir=destination.parent) as tmp:
        staging = Path(tmp) / "artifact"
        model.save_pretrained(staging, state_dict=state, max_shard_size=max_shard_size)
        config_path = staging / "config.json"
        config = json.loads(config_path.read_text())
        config["specforge_artifact"] = {"format": "specforge", "version": 1}
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        os.replace(staging, destination)
