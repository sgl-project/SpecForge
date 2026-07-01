# from .auto import AutoDistributedTargetModel, AutoDraftModelConfig, AutoEagle3DraftModel
from .auto import AutoDraftModelConfig, AutoEagle3DraftModel
from .draft.llama3_eagle import LlamaForCausalLMEagle3
from .draft.peagle import PEagleDraftModel
from .target import (
    CustomEagle3TargetEngine,
    CustomEagle3TargetModel,
    Eagle3TargetEngine,
    HFEagle3TargetEngine,
    HFEagle3TargetModel,
    SGLangEagle3TargetEngine,
    SGLangEagle3TargetModel,
    TargetEngine,
    get_eagle3_target_model,
    get_target_engine,
)

__all__ = [
    "LlamaForCausalLMEagle3",
    "PEagleDraftModel",
    # Generic (Phase B) surface
    "TargetEngine",
    "Eagle3TargetEngine",
    "SGLangEagle3TargetEngine",
    "HFEagle3TargetEngine",
    "CustomEagle3TargetEngine",
    "get_target_engine",
    "get_eagle3_target_model",
    # Back-compat aliases (pre-Phase-B names)
    "SGLangEagle3TargetModel",
    "HFEagle3TargetModel",
    "CustomEagle3TargetModel",
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
]
