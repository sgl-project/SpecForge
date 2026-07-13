# coding=utf-8
"""Import shim — moved to ``specforge.inference.target_engine.factory``."""

from specforge.inference.target_engine.factory import (  # noqa: F401
    available_target_engines,
    get_target_engine,
)

__all__ = ["get_target_engine", "available_target_engines"]
