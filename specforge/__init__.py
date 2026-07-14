"""SpecForge public API with import-light lazy exports.

Configuration parsing and disaggregated producers must not import PyTorch,
Transformers, or SGLang just because Python initializes the top-level package.
The historical public model names remain available and resolve on first access.
"""

from __future__ import annotations

import importlib


_EXPORTS = {
    # Composite training models.
    "OnlineDFlashModel": ("specforge.core", "OnlineDFlashModel"),
    "OnlineDominoModel": ("specforge.core", "OnlineDominoModel"),
    "OnlineDSparkModel": ("specforge.core", "OnlineDSparkModel"),
    "OnlineEagle3Model": ("specforge.core", "OnlineEagle3Model"),
    "OnlinePEagleModel": ("specforge.core", "OnlinePEagleModel"),
    "QwenVLOnlineEagle3Model": ("specforge.core", "QwenVLOnlineEagle3Model"),
    # Draft/target modeling surface.
    "LlamaForCausalLMEagle3": ("specforge.modeling", "LlamaForCausalLMEagle3"),
    "PEagleDraftModel": ("specforge.modeling", "PEagleDraftModel"),
    "TargetEngine": ("specforge.modeling", "TargetEngine"),
    "Eagle3TargetEngine": ("specforge.modeling", "Eagle3TargetEngine"),
    "SGLangEagle3TargetEngine": (
        "specforge.modeling",
        "SGLangEagle3TargetEngine",
    ),
    "HFEagle3TargetEngine": ("specforge.modeling", "HFEagle3TargetEngine"),
    "CustomEagle3TargetEngine": (
        "specforge.modeling",
        "CustomEagle3TargetEngine",
    ),
    "get_target_engine": ("specforge.modeling", "get_target_engine"),
    "get_eagle3_target_model": (
        "specforge.modeling",
        "get_eagle3_target_model",
    ),
    "SGLangEagle3TargetModel": (
        "specforge.modeling",
        "SGLangEagle3TargetModel",
    ),
    "HFEagle3TargetModel": ("specforge.modeling", "HFEagle3TargetModel"),
    "CustomEagle3TargetModel": (
        "specforge.modeling",
        "CustomEagle3TargetModel",
    ),
    "AutoDraftModelConfig": ("specforge.modeling", "AutoDraftModelConfig"),
    "AutoEagle3DraftModel": ("specforge.modeling", "AutoEagle3DraftModel"),
}

_SUBMODULES = {
    "core",
    "distributed",
    "launch",
    "modeling",
    "utils",
}

__all__ = sorted([*_EXPORTS, *_SUBMODULES])


def __getattr__(name: str):
    if name in _SUBMODULES:
        value = importlib.import_module(f"specforge.{name}")
    elif name in _EXPORTS:
        module_name, attribute = _EXPORTS[name]
        value = getattr(importlib.import_module(module_name), attribute)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
