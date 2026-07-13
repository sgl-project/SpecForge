# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Generic target-engine factory (Phase B).

``get_target_engine(strategy=..., backend=...)`` is the de-EAGLE3'd factory: it
dispatches on the *algorithm* (``eagle3`` / ``dflash``) and delegates to the
per-algorithm loaders, which in turn dispatch on the *backend* (sglang / hf /
custom / sglang_server). The legacy ``get_eagle3_target_model`` /
``get_dflash_target_model`` remain as thin shims (they ARE the per-algorithm
loaders this factory calls), so nothing that imports them breaks.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import TargetEngine

# strategy name -> per-algorithm engine. Domino reuses the DFlash engine (same
# capture: concatenated layer hidden states, no target distribution). The loaders
# are imported LAZILY inside the factory: dflash_target_model imports sglang
# internals unconditionally, so eager import here would break the design property
# that ``import specforge`` works even when the installed sglang lacks the pinned
# symbols (eagle3_target_model guards its imports; see its module docstring).
_ENGINE_STRATEGIES = ("eagle3", "dflash", "domino")


def available_target_engines():
    """Strategy names with a registered target-engine loader."""
    return sorted(_ENGINE_STRATEGIES)


def _resolve_loader(strategy: str):
    if strategy == "eagle3":
        from .eagle3_target_model import get_eagle3_target_model

        return get_eagle3_target_model
    if strategy in ("dflash", "domino"):
        from .dflash_target_model import get_dflash_target_model

        return get_dflash_target_model
    raise ValueError(
        f"no target engine for strategy {strategy!r}; "
        f"registered: {available_target_engines()}"
    )


def get_target_engine(
    pretrained_model_name_or_path: str,
    *,
    strategy: str = "eagle3",
    backend: str = "sglang",
    torch_dtype: Optional[torch.dtype] = None,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> TargetEngine:
    """Load a frozen :class:`TargetEngine` for ``strategy`` on ``backend``.

    The single, algorithm-agnostic entry point launch code should prefer. The
    older ``get_eagle3_target_model`` / ``get_dflash_target_model`` stay valid.
    """
    loader = _resolve_loader(strategy)
    return loader(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        backend=backend,
        torch_dtype=torch_dtype,
        device=device,
        cache_dir=cache_dir,
        **kwargs,
    )


__all__ = ["get_target_engine", "available_target_engines"]
