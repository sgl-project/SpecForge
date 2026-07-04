# coding=utf-8
"""Import shim — moved to ``specforge.inference.target_engine.base``."""

from specforge.inference.target_engine.base import (  # noqa: F401
    KNOWN_BACKENDS,
    TargetEngine,
)

__all__ = ["TargetEngine", "KNOWN_BACKENDS"]
