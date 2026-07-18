# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Draft-architecture registry: the extension point for new draft model classes.

A draft *architecture* (the model class) is a separate axis from the
per-algorithm training strategy (``specforge.training.strategies.registry``):
one architecture can train under several algorithms and vice versa. Registering
a class here makes it constructible from a draft-config JSON whose
``architectures[0]`` names it — adding an architecture is a new file plus
``@register_draft``, not an edit to ``modeling/auto.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

DRAFT_REGISTRY: Dict[str, type] = {}


def register_draft(cls: Optional[type] = None, *, name: Optional[str] = None):
    """Class decorator registering a draft architecture.

    The key defaults to the class name — exactly what draft-config JSONs carry
    in ``architectures``. The class must declare ``config_class`` (the HF
    ``PretrainedConfig`` subclass it is built from) so the auto loaders can
    resolve both the model and its config from the one registry entry.
    """

    def _register(cls: type) -> type:
        key = name or cls.__name__
        if getattr(cls, "config_class", None) is None:
            raise TypeError(
                f"@register_draft: {cls.__name__} must declare config_class"
            )
        existing = DRAFT_REGISTRY.get(key)
        if existing is not None and existing is not cls:
            raise ValueError(
                f"draft architecture {key!r} already registered to "
                f"{existing.__name__}"
            )
        DRAFT_REGISTRY[key] = cls
        return cls

    return _register(cls) if cls is not None else _register


def resolve_draft(name: str) -> type:
    try:
        return DRAFT_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown draft architecture {name!r}; available: {available_drafts()}"
        ) from None


def available_drafts() -> List[str]:
    return sorted(DRAFT_REGISTRY)


__all__ = ["DRAFT_REGISTRY", "register_draft", "resolve_draft", "available_drafts"]
