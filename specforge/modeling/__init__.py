from .auto import AutoDistributedTargetModel, AutoDraftModelConfig, AutoEagle3DraftModel
from .draft.llama3_eagle import LlamaForCausalLMEagle3
from .draft.qwen2_eagle import Qwen2ForCausalLMEagle3

__all__ = [
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
    "AutoDistributedTargetModel",
    "LlamaForCausalLMEagle3",
    "Qwen2ForCausalLMEagle3",
]
