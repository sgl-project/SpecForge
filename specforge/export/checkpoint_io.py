# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Shared exporter plumbing: resolve a training checkpoint, materialize the draft."""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any, Dict, Optional

import torch

STATE_FILE = "training_state.pt"
_DISABLE_LEGACY_ROPE_SCALING_ENV = "SPECFORGE_DISABLE_LEGACY_ROPE_SCALING"


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def apply_legacy_rope_scaling(output_dir: str) -> bool:
    """Keep modern and legacy RoPE scaling fields compatible in an export.

    Transformers 5 writes ``rope_parameters`` while older serving stacks read
    only ``rope_scaling`` and the top-level ``rope_theta``. Mirror non-default
    scaling representations and preserve the modern ``rope_theta`` for legacy
    readers. On by default; set
    ``SPECFORGE_DISABLE_LEGACY_ROPE_SCALING=1`` to skip. Returns whether the
    config was rewritten.
    """
    if _env_flag_enabled(_DISABLE_LEGACY_ROPE_SCALING_ENV):
        return False

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    rope_parameters = config.get("rope_parameters")
    rope_scaling = config.get("rope_scaling")

    def rope_kind(payload):
        return (payload or {}).get("rope_type") or (payload or {}).get("type")

    changed = False
    if rope_parameters and "rope_theta" in rope_parameters:
        rope_theta = rope_parameters["rope_theta"]
        if config.get("rope_theta") != rope_theta:
            config["rope_theta"] = rope_theta
            changed = True

    if (
        rope_parameters
        and not rope_scaling
        and rope_kind(rope_parameters) not in (None, "default")
    ):
        config["rope_scaling"] = {
            key: value for key, value in rope_parameters.items() if key != "rope_theta"
        }
        changed = True
    elif (
        rope_scaling
        and not rope_parameters
        and rope_kind(rope_scaling) not in (None, "default")
    ):
        mirrored = dict(rope_scaling)
        if "rope_theta" in config:
            mirrored.setdefault("rope_theta", config["rope_theta"])
        config["rope_parameters"] = mirrored
        changed = True

    if not changed:
        return False

    temporary = f"{config_path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, config_path)
    return True


def resolve_training_state(checkpoint_path: str) -> Dict[str, Any]:
    """Load the runtime training state from any checkpoint-shaped path.

    Accepts a ``training_state.pt`` file, a ``{run_id}-step{N}`` checkpoint
    directory, or a run ``output_dir`` (resolved through its run-scoped
    ``{run_id}-latest`` pointer — the CheckpointManager layout); a ``file://``
    URI of any of these (the ``Checkpoint.checkpoint_uri`` form) also works.
    """
    path = checkpoint_path
    if path.startswith("file://"):
        path = path[len("file://") :]
    if os.path.isfile(path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_dir = os.path.dirname(path)
        return _consolidate_export_state(state, checkpoint_dir)
    state_file = os.path.join(path, STATE_FILE)
    if os.path.isfile(state_file):
        state = torch.load(state_file, map_location="cpu", weights_only=False)
        return _consolidate_export_state(state, path)
    latest = [
        link
        for link in sorted(glob.glob(os.path.join(path, "*-latest")))
        if os.path.isfile(os.path.join(os.path.realpath(link), STATE_FILE))
    ]
    if len(latest) > 1:
        raise ValueError(
            f"{checkpoint_path!r} holds several runs "
            f"({[os.path.basename(p) for p in latest]}); pass the "
            f"{{run_id}}-step{{N}} checkpoint directory explicitly"
        )
    if latest:
        checkpoint_dir = os.path.realpath(latest[0])
        state = torch.load(
            os.path.join(checkpoint_dir, STATE_FILE),
            map_location="cpu",
            weights_only=False,
        )
        return _consolidate_export_state(state, checkpoint_dir)
    # no `{run_id}-latest` pointer (e.g. symlink-free filesystem): fall back to
    # the highest step directory, mirroring CheckpointManager.latest_dir().
    steps = []
    for cand in glob.glob(os.path.join(path, "*-step*")):
        m = re.search(r"-step(\d+)$", os.path.basename(cand))
        if m and os.path.isfile(os.path.join(cand, STATE_FILE)):
            steps.append((int(m.group(1)), cand))
    if steps:
        best = max(steps)[1]
        state = torch.load(
            os.path.join(best, STATE_FILE), map_location="cpu", weights_only=False
        )
        return _consolidate_export_state(state, best)
    raise FileNotFoundError(
        f"{checkpoint_path!r} is not a training_state.pt, a checkpoint directory, "
        f"or an output_dir with a '{{run_id}}-latest' pointer / step directories"
    )


def _consolidate_export_state(
    state: Dict[str, Any], checkpoint_dir: str
) -> Dict[str, Any]:
    if int(state.get("expert_parallel_size", 1)) <= 1:
        return state
    from specforge.training.checkpoint import consolidate_draft_state

    state = dict(state)
    state["draft_state_dict"] = consolidate_draft_state(checkpoint_dir, state)
    return state


def materialize_draft(
    state: Dict[str, Any],
    draft_config_path: str,
    *,
    vocab_mapping_path: Optional[str] = None,
):
    """Build the draft model and load the checkpoint's draft weights into it.

    Validates the state dict against the architecture: unexpected keys fail, and
    the only tolerated missing keys are the embeddings (excluded from draft
    checkpoints by design — serving loads them from the target). Legacy
    ``t2d``/``d2t`` buffers are tolerated only when ``vocab_mapping_path`` is
    supplied to restore them.
    """
    from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig

    draft_config = AutoDraftModelConfig.from_file(draft_config_path)
    model = AutoDraftModel.from_config(draft_config, torch_dtype=torch.bfloat16)
    missing, unexpected = model.load_state_dict(state["draft_state_dict"], strict=False)
    if unexpected:
        raise ValueError(
            f"checkpoint carries weights the {type(model).__name__} architecture "
            f"does not have: {sorted(unexpected)}"
        )
    tolerated = {"t2d", "d2t"} if vocab_mapping_path else set()
    non_embed_missing = [
        key for key in missing if "embed" not in key.lower() and key not in tolerated
    ]
    if non_embed_missing:
        raise ValueError(
            f"checkpoint is missing non-embedding draft weights: "
            f"{sorted(non_embed_missing)}"
        )
    if vocab_mapping_path:
        # Refresh current checkpoints and restore legacy checkpoints that predate
        # the mapping buffers.
        model.load_vocab_mapping(vocab_mapping_path)
    return model


__all__ = [
    "apply_legacy_rope_scaling",
    "resolve_training_state",
    "materialize_draft",
    "STATE_FILE",
]
