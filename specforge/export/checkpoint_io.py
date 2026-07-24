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

import os
from typing import Any, Dict, Optional

import torch

from specforge.training.checkpoint import STATE_FILE, CheckpointManager


def resolve_training_state(checkpoint_path: str) -> Dict[str, Any]:
    """Load state from a committed SpecForge runtime checkpoint.

    Accepts a ``training_state.pt`` file, a ``{run_id}-step{N}`` checkpoint
    directory, or a run ``output_dir``; a ``file://`` URI of any of these also
    works. All forms resolve through ``CheckpointManager`` so a payload without
    its ``_SUCCESS`` commit marker is never exported.
    """
    checkpoint_dir = CheckpointManager.resolve_committed_checkpoint_dir(checkpoint_path)
    return torch.load(
        os.path.join(checkpoint_dir, STATE_FILE),
        map_location="cpu",
        weights_only=False,
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


__all__ = ["resolve_training_state", "materialize_draft", "STATE_FILE"]
