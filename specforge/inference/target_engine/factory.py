# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Generic target-engine factory.

``get_target_engine(strategy=..., backend=...)`` dispatches on the *algorithm*
and the *backend*. The built-in algorithms (``eagle3`` / ``dflash`` / ``domino``)
route through their per-algorithm loaders (kept import-compatible); any other
strategy with a registered :class:`~.capture_policy.CapturePolicy` gets the
generic per-backend engine — adding an algorithm is a policy registration, not a
new engine class per backend.
"""

from __future__ import annotations

from typing import Optional

import torch

from .base import TargetEngine

# Built-in strategies with a dedicated per-algorithm loader. Domino reuses the
# DFlash engine (same capture: concatenated layer hidden states, no target
# distribution). Loaders are imported LAZILY inside the factory so ``import
# specforge`` works even when the installed sglang lacks the pinned symbols.
_ENGINE_STRATEGIES = ("eagle3", "dflash", "domino")


def available_target_engines():
    """Strategy names with a target-engine loader (built-in or via policy)."""
    from .capture_policy import CAPTURE_POLICIES

    return sorted(set(_ENGINE_STRATEGIES) | set(CAPTURE_POLICIES))


def _generic_loader(strategy: str):
    """Loader over the generic per-backend engines for a registered policy."""
    from .capture_policy import resolve_capture_policy

    try:
        policy = resolve_capture_policy(strategy)
    except KeyError:
        raise ValueError(
            f"no target engine for strategy {strategy!r}; "
            f"registered: {available_target_engines()}"
        ) from None

    def load(pretrained_model_name_or_path, backend="sglang", **kwargs):
        if backend == "sglang":
            from .sglang import SGLangTargetEngine as engine_cls
        elif backend == "hf":
            from .hf import HFTargetEngine as engine_cls
        elif backend == "custom":
            from .custom import CustomTargetEngine as engine_cls
        else:
            raise ValueError(f"Invalid backend: {backend}")
        return engine_cls.from_pretrained(
            pretrained_model_name_or_path, policy=policy, **kwargs
        )

    return load


def _resolve_loader(strategy: str):
    if strategy == "eagle3":
        from .eagle3_target_model import get_eagle3_target_model

        return get_eagle3_target_model
    if strategy in ("dflash", "domino"):
        from .dflash_target_model import get_dflash_target_model

        return get_dflash_target_model
    return _generic_loader(strategy)


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
