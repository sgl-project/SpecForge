"""Data APIs with lazy exports.

Pre-tokenized prompt ingestion is standard-library-only; importing it must not
eagerly load torch, datasets, or distributed data-loader helpers.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "build_eagle3_dataset": ("specforge.data.preprocessing", "build_eagle3_dataset"),
    "build_offline_eagle3_dataset": (
        "specforge.data.preprocessing",
        "build_offline_eagle3_dataset",
    ),
    "generate_vocab_mapping_file": (
        "specforge.data.preprocessing",
        "generate_vocab_mapping_file",
    ),
    "preprocess_conversations": (
        "specforge.data.preprocessing",
        "preprocess_conversations",
    ),
    "prepare_prompt_tasks": (
        "specforge.data.prompt_builder",
        "prepare_prompt_tasks",
    ),
    "prepare_dp_dataloaders": (
        "specforge.data.utils",
        "prepare_dp_dataloaders",
    ),
    "ChatTemplate": ("specforge.data.template", "ChatTemplate"),
}

_SUBMODULES = {
    "preprocessing",
    "prompt_builder",
    "template",
    "utils",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _SUBMODULES:
        value = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = value
        return value
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
