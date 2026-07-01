# coding=utf-8
"""E0 import shim — moved to ``specforge.inference.target_engine.sglang_backend``."""

from specforge.inference.target_engine.sglang_backend import (  # noqa: F401
    SGLangCaptureBackend,
    SGLangRunner,
    wrap_eagle3_logits_processors_in_module,
)

__all__ = [
    "SGLangRunner",
    "wrap_eagle3_logits_processors_in_module",
    "SGLangCaptureBackend",
]
