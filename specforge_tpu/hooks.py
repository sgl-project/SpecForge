"""Minimal attribute-patching helpers for the TPU plugin."""

from __future__ import annotations

import functools
import importlib
from collections.abc import Callable
from typing import Any


def _resolve(target: str) -> tuple[Any, str]:
    """Split "pkg.mod.Class.attr" into (owner object, attribute name)."""
    parts = target.split(".")
    for split in range(len(parts) - 1, 0, -1):
        try:
            owner: Any = importlib.import_module(".".join(parts[:split]))
        except ModuleNotFoundError:
            continue
        for name in parts[split:-1]:
            owner = getattr(owner, name)
        return owner, parts[-1]
    raise ModuleNotFoundError(target)


def replace(target: str, fn: Callable[..., Any]) -> None:
    owner, name = _resolve(target)
    if not hasattr(owner, name):
        raise AttributeError(f"patch target {target} does not exist")
    setattr(owner, name, fn)


def around(target: str, wrapper: Callable[..., Any]) -> None:
    """Install wrapper(original, *args, **kwargs) in place of target."""
    owner, name = _resolve(target)
    original = getattr(owner, name)
    if getattr(original, "__specforge_tpu_wrapped__", False):
        return

    @functools.wraps(original)
    def patched(*args: Any, **kwargs: Any) -> Any:
        return wrapper(original, *args, **kwargs)

    patched.__specforge_tpu_wrapped__ = True  # type: ignore[attr-defined]
    setattr(owner, name, patched)
