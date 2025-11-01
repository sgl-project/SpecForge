from .auto import AutoDraftModelConfig, AutoEagle3DraftModel
from .base import Eagle3DraftModel
from .llama3_eagle import LlamaForCausalLMEagle3

__all__ = [
    "AutoEagle3DraftModel",
    "Eagle3DraftModel",
    "LlamaForCausalLMEagle3",
    "AutoDraftModelConfig",
]
