from .auto import AutoDistributedTargetModel, AutoDraftModelConfig, AutoEagle3DraftModel
from .draft.llama3_eagle import LlamaForCausalLMEagle3
from .draft.deepseekv3_eagle import DeepseekV3ForCausalLMEagle3

__all__ = [
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
    "AutoDistributedTargetModel",
    "LlamaForCausalLMEagle3",
    "DeepseekV3ForCausalLMEagle3",
]
