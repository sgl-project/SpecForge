"""Public SpecForge package facade.

Heavy model and algorithm modules are resolved lazily so lightweight contracts,
configuration tools, and planners do not initialize PyTorch or optional model
backends merely by importing the package.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Keep the historical package-level API visible to type checkers and IDEs
    # without importing model implementations at runtime.
    from .core import (  # noqa: F401
        OnlineDFlashModel,
        OnlineDominoModel,
        OnlineDSparkModel,
        OnlineEagle3Model,
        OnlinePEagleModel,
        QwenVLOnlineEagle3Model,
    )
    from .modeling import (  # noqa: F401
        AutoDraftModelConfig,
        AutoEagle3DraftModel,
        CustomEagle3TargetEngine,
        CustomEagle3TargetModel,
        Eagle3TargetEngine,
        HFEagle3TargetEngine,
        HFEagle3TargetModel,
        LlamaForCausalLMEagle3,
        PEagleDraftModel,
        SGLangEagle3TargetEngine,
        SGLangEagle3TargetModel,
        TargetEngine,
        get_eagle3_target_model,
        get_target_engine,
    )

_LAZY_MODULES = {
    "core": "specforge.core",
    "modeling": "specforge.modeling",
}

_LAZY_EXPORTS = {
    # Algorithm wrappers historically re-exported by ``specforge.core``.
    "OnlineDFlashModel": ("specforge.core", "OnlineDFlashModel"),
    "OnlineDominoModel": ("specforge.core", "OnlineDominoModel"),
    "OnlineDSparkModel": ("specforge.core", "OnlineDSparkModel"),
    "OnlineEagle3Model": ("specforge.core", "OnlineEagle3Model"),
    "OnlinePEagleModel": ("specforge.core", "OnlinePEagleModel"),
    "QwenVLOnlineEagle3Model": ("specforge.core", "QwenVLOnlineEagle3Model"),
    # Modeling facade retained for direct ``from specforge import Name`` users.
    "AutoDraftModelConfig": ("specforge.modeling", "AutoDraftModelConfig"),
    "AutoEagle3DraftModel": ("specforge.modeling", "AutoEagle3DraftModel"),
    "CustomEagle3TargetEngine": (
        "specforge.modeling",
        "CustomEagle3TargetEngine",
    ),
    "CustomEagle3TargetModel": ("specforge.modeling", "CustomEagle3TargetModel"),
    "Eagle3TargetEngine": ("specforge.modeling", "Eagle3TargetEngine"),
    "HFEagle3TargetEngine": ("specforge.modeling", "HFEagle3TargetEngine"),
    "HFEagle3TargetModel": ("specforge.modeling", "HFEagle3TargetModel"),
    "LlamaForCausalLMEagle3": ("specforge.modeling", "LlamaForCausalLMEagle3"),
    "PEagleDraftModel": ("specforge.modeling", "PEagleDraftModel"),
    "SGLangEagle3TargetEngine": (
        "specforge.modeling",
        "SGLangEagle3TargetEngine",
    ),
    "SGLangEagle3TargetModel": ("specforge.modeling", "SGLangEagle3TargetModel"),
    "TargetEngine": ("specforge.modeling", "TargetEngine"),
    "get_eagle3_target_model": (
        "specforge.modeling",
        "get_eagle3_target_model",
    ),
    "get_target_engine": ("specforge.modeling", "get_target_engine"),
}


def __getattr__(name: str) -> object:
    module_name = _LAZY_MODULES.get(name)
    if module_name is not None:
        value = import_module(module_name)
    else:
        export = _LAZY_EXPORTS.get(name)
        if export is None:
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        module_name, attribute = export
        value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_MODULES, *_LAZY_EXPORTS})


__all__ = ["modeling", "core"]
