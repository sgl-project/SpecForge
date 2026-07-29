"""Dependency-lazy access to the offline SGLang capture backend."""

from importlib import import_module

__all__ = ["OfflineSGLangCaptureBackend"]


def __getattr__(name):
    if name != "OfflineSGLangCaptureBackend":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.capture"), name)
    globals()[name] = value
    return value
