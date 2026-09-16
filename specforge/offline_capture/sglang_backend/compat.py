"""Shims for SGLang API drift between the pinned release and sglang main.

The offline capture backend is pinned to ``sglang==0.5.18`` but is also run
against sglang main (for targets that the pinned release does not ship, e.g.
Ling-3.0's ``BailingMoeV3``).  Each helper here accepts both API shapes and
prefers the current one; none of them changes behaviour on the pinned version.

NOTE (sglang main, Sep 2026):
  * ``sglang.srt.utils.require_mlp_sync`` / ``require_mlp_tp_gather`` read the
    DP-attention flags from the runtime context and take no arguments; 0.5.18
    takes ``ServerArgs``.
  * ``ServerArgs.device`` stays ``None`` until the launcher's post-process pass
    resolves it; ``ModelRunner`` needs a concrete device.
  * ``sglang.srt.runtime_context.publish(...)`` must run before a
    ``ModelRunner`` is built; 0.5.18 exposes the same API but does not require
    it, and publishing there is harmless.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any

from sglang.srt import utils as sglang_utils

from specforge.utils import get_device_type

logger = logging.getLogger(__name__)


def _takes_arguments(fn: Any) -> bool:
    try:
        return len(inspect.signature(fn).parameters) > 0
    except (TypeError, ValueError):  # builtins / C functions: assume the old shape
        return True


def require_mlp_sync(server_args: Any) -> bool:
    """``sglang.srt.utils.require_mlp_sync`` for both signatures."""
    fn = sglang_utils.require_mlp_sync
    return bool(fn(server_args) if _takes_arguments(fn) else fn())


def require_mlp_tp_gather(server_args: Any) -> bool:
    """``sglang.srt.utils.require_mlp_tp_gather`` for both signatures."""
    fn = sglang_utils.require_mlp_tp_gather
    return bool(fn(server_args) if _takes_arguments(fn) else fn())


def resolve_device(server_args: Any) -> str:
    """Fill in ``ServerArgs.device`` when the launcher has not resolved it.

    Uses SpecForge's own resolver (``SPECFORGE_DEVICE``, then CUDA, then NPU,
    then CPU) so the capture backend lands on the same accelerator as training.
    """
    if getattr(server_args, "device", None) is None:
        server_args.device = get_device_type()
    return server_args.device


def publish_runtime_context(server_args: Any, *, role: str = "scheduler") -> bool:
    """Publish the SGLang runtime context once, if the installed SGLang has one.

    Returns True when this call published the context, False when it was
    already published or the API does not exist.
    """
    try:
        runtime_context = importlib.import_module("sglang.srt.runtime_context")
    except ImportError:
        return False
    publish = getattr(runtime_context, "publish", None)
    if publish is None:
        return False
    publish_role = getattr(runtime_context, "publish_role", None)
    if publish_role is not None:
        try:
            if publish_role() is not None:
                return False
        except Exception:  # pragma: no cover - defensive: treat as unpublished
            pass
    publish(server_args, role=role)
    logger.debug("published SGLang runtime context (role=%s)", role)
    return True
