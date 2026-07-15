# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Single target-engine factory for every capture strategy and backend."""

from __future__ import annotations

from typing import Optional

import torch

from .base import KNOWN_BACKENDS, TargetEngine
from .target_capture_policy import (
    TARGET_CAPTURE_POLICIES,
    resolve_target_capture_policy,
)


def available_target_engines():
    """Strategy names with a registered target-capture policy."""
    return sorted(TARGET_CAPTURE_POLICIES)


def _load_engine(
    pretrained_model_name_or_path: str,
    *,
    strategy: str,
    backend: str,
    torch_dtype: Optional[torch.dtype],
    device: Optional[str],
    cache_dir: Optional[str],
    **kwargs,
) -> TargetEngine:
    try:
        policy = resolve_target_capture_policy(strategy)
    except KeyError:
        raise ValueError(
            f"no target engine for strategy {strategy!r}; "
            f"registered: {available_target_engines()}"
        ) from None

    if backend == "hf":
        from .hf import HFTargetEngine

        return HFTargetEngine.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            policy=policy,
            **kwargs,
        )
    if backend == "sglang":
        from .sglang import SGLangTargetEngine

        return SGLangTargetEngine.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            policy=policy,
            **kwargs,
        )
    if backend == "custom":
        from .custom import CustomTargetEngine

        return CustomTargetEngine.from_pretrained(
            pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            policy=policy,
            **kwargs,
        )
    raise ValueError(f"Unknown backend {backend!r}; known: {KNOWN_BACKENDS}")


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
    """Load the policy-driven frozen target for ``strategy`` on ``backend``."""
    return _load_engine(
        pretrained_model_name_or_path,
        strategy=strategy,
        backend=backend,
        torch_dtype=torch_dtype,
        device=device,
        cache_dir=cache_dir,
        **kwargs,
    )


__all__ = ["get_target_engine", "available_target_engines"]
