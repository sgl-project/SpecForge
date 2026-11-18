"""Dependency-lazy access to plugin-based offline feature preparation."""

from importlib import import_module

__all__ = ["OfflineCaptureClient"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.client"), name)
    globals()[name] = value
    return value
