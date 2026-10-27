"""FlexAttention backend selection shared by DFlash training components."""

from __future__ import annotations

import os
from typing import Optional

from specforge.torch_compat import patch_inductor_cutedsl_lowerings

_VALID_BACKENDS = {"AUTO", "TRITON", "FLASH", "TRITON_DECODE"}
_BACKEND_ENV = "SPECFORGE_FLEX_ATTENTION_BACKEND"


def flex_attention_backend() -> Optional[str]:
    backend = os.environ.get(_BACKEND_ENV, "").upper()
    if not backend:
        return None
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"{_BACKEND_ENV} must be one of {sorted(_VALID_BACKENDS)}, "
            f"got {backend!r}"
        )
    if backend == "FLASH":
        patch_inductor_cutedsl_lowerings()
    return backend


__all__ = ["flex_attention_backend"]
