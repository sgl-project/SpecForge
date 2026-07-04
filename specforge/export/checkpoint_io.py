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
import os
import re
from typing import Any, Dict, Optional

import torch

STATE_FILE = "training_state.pt"


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
        return torch.load(path, map_location="cpu", weights_only=False)
    state_file = os.path.join(path, STATE_FILE)
    if os.path.isfile(state_file):
        return torch.load(state_file, map_location="cpu", weights_only=False)
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
        return torch.load(
            os.path.join(os.path.realpath(latest[0]), STATE_FILE),
            map_location="cpu",
            weights_only=False,
        )
    # no `{run_id}-latest` pointer (e.g. symlink-free filesystem): fall back to
    # the highest step directory, mirroring CheckpointManager.latest_dir().
    steps = []
    for cand in glob.glob(os.path.join(path, "*-step*")):
        m = re.search(r"-step(\d+)$", os.path.basename(cand))
        if m and os.path.isfile(os.path.join(cand, STATE_FILE)):
            steps.append((int(m.group(1)), cand))
    if steps:
        best = max(steps)[1]
        return torch.load(
            os.path.join(best, STATE_FILE), map_location="cpu", weights_only=False
        )
    raise FileNotFoundError(
        f"{checkpoint_path!r} is not a training_state.pt, a checkpoint directory, "
        f"or an output_dir with a '{{run_id}}-latest' pointer / step directories"
    )


def materialize_draft(
    state: Dict[str, Any],
    draft_config_path: str,
    *,
    vocab_mapping_path: Optional[str] = None,
):
    """Build the draft model and load the checkpoint's draft weights into it.

    Validates the state dict against the architecture: unexpected keys fail, and
    the only tolerated missing keys are the embeddings (excluded from draft
    checkpoints by design — serving loads them from the target).
    """
    from specforge.modeling.auto import AutoDraftModelConfig, AutoEagle3DraftModel

    draft_config = AutoDraftModelConfig.from_file(draft_config_path)
    model = AutoEagle3DraftModel.from_config(draft_config, torch_dtype=torch.bfloat16)
    missing, unexpected = model.load_state_dict(state["draft_state_dict"], strict=False)
    if unexpected:
        raise ValueError(
            f"checkpoint carries weights the {type(model).__name__} architecture "
            f"does not have: {sorted(unexpected)}"
        )
    tolerated = {"t2d", "d2t"} if vocab_mapping_path else set()
    non_embed_missing = [
        k for k in missing if "embed" not in k.lower() and k not in tolerated
    ]
    if non_embed_missing:
        raise ValueError(
            f"checkpoint is missing non-embedding draft weights: "
            f"{sorted(non_embed_missing)}"
        )
    if vocab_mapping_path:
        # tolerated above precisely so a legacy checkpoint that predates the
        # t2d/d2t buffers can be exported by supplying the mapping here.
        model.load_vocab_mapping(vocab_mapping_path)
    return model


__all__ = ["resolve_training_state", "materialize_draft", "STATE_FILE"]
