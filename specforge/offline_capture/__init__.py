"""Dependency-lazy offline target-feature preparation utilities.

Online training never imports this package: it captures through an external
SGLang server and the server-capture adapter.  The local SGLang integration
here exists only for the standalone hidden-state preparation script.
"""

from importlib import import_module

__all__ = [
    "OfflineEagle3CaptureBatch",
    "OfflineEagle3SGLangCapture",
    "load_offline_eagle3_capture",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.sglang"), name)
    globals()[name] = value
    return value
