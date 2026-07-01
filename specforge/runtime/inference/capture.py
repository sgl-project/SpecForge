# coding=utf-8
"""E0 import shim — moved to ``specforge.inference.capture``."""

from specforge.inference.capture import (  # noqa: F401
    CaptureConfig,
    CaptureMismatchError,
    verify_capture,
)

__all__ = ["CaptureConfig", "CaptureMismatchError", "verify_capture"]
