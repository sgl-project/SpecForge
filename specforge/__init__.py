"""SpecForge package with lazily loaded compatibility exports."""

from importlib import import_module


def __getattr__(name):
    if name in {"core", "modeling"}:
        return import_module(f"{__name__}.{name}")
    for module_name in ("core", "modeling"):
        module = import_module(f"{__name__}.{module_name}")
        if name in getattr(module, "__all__", ()):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["modeling", "core"]
