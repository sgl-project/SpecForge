"""Native target registrations shared by initialization and serving export."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class NativeTarget:
    source_prefix: str
    validate: Callable
    export: Callable


_TARGETS = {}


def register_native_target(
    model_type, architecture, *, source_prefix, validate, export
):
    """Register the native implementation for one target/draft pair.

    validate(config, text_config) checks computation without file I/O.
    export(config, state, source, payload, output, *, copy_target) owns the
    family's configuration and weight layout and may reuse write_native.
    source_prefix selects complete from_target root copies during initialization.
    """
    key = (model_type, architecture)
    if key in _TARGETS:
        raise ValueError(f"duplicate native target registration: {key}")
    _TARGETS[key] = NativeTarget(source_prefix, validate, export)


def resolve_native_target(config, target):
    key = (target.get("model_type"), config.architecture)
    try:
        return _TARGETS[key]
    except KeyError:
        raise ValueError(f"no native target implementation for {key}") from None


def validate_target_initialization(config, target, from_target):
    # Only a root destination declares a complete native model initialization.
    entry = (from_target or {}).get("")
    source = entry.get("from") if isinstance(entry, dict) else entry
    text = target.get("text_config", target)
    native = _TARGETS.get((text.get("model_type"), config.architecture))
    if native is not None and source == native.source_prefix:
        native.validate(config, text)


# Family modules register after the registry is defined.
from .formats.native import qwen3_5  # noqa: E402,F401
