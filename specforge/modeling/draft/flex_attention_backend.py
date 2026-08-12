"""FlexAttention backend selection shared by DFlash training components."""

from __future__ import annotations

import os

import torch
from packaging.version import InvalidVersion, Version

from specforge.torch_compat import patch_inductor_cutedsl_lowerings

_VALID_BACKENDS = {"AUTO", "TRITON", "FLASH", "TRITON_DECODE"}
_BACKEND_ENV = "SPECFORGE_FLEX_ATTENTION_BACKEND"


def _default_backend() -> str:
    """Choose a stable H200 path and the faster Blackwell path when supported."""
    try:
        torch_version = Version(torch.__version__.split("+", 1)[0])
    except InvalidVersion:
        torch_version = Version("0")
    if (
        torch_version >= Version("2.11")
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 10
    ):
        return "FLASH"
    return "TRITON"


_DEFAULT_BACKEND = _default_backend()


def flex_attention_backend() -> str:
    backend = os.environ.get(_BACKEND_ENV, "").upper()
    if not backend:
        # Keep compiler-visible forward paths to an immutable string. Capability
        # detection at module import avoids a Dynamo graph break per call.
        backend = _DEFAULT_BACKEND
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"{_BACKEND_ENV} must be one of {sorted(_VALID_BACKENDS)}, "
            f"got {backend!r}"
        )
    if backend == "FLASH":
        patch_inductor_cutedsl_lowerings()
    return backend


__all__ = ["flex_attention_backend"]
