from .auto import AutoDraftModelConfig, AutoEagle3DraftModel
from .base import Eagle3DraftModel, load_param_from_target_model
from .llama3_eagle import LlamaForCausalLMEagle3

__all__ = [
    "AutoEagle3DraftModel",
    "Eagle3DraftModel",
    "LlamaForCausalLMEagle3",
    "AutoDraftModelConfig",
    "load_param_from_target_model",
]
