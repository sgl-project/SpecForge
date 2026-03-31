# from .auto import AutoDistributedTargetModel, AutoDraftModelConfig, AutoEagle3DraftModel
from .auto import AutoDraftModelConfig, AutoEagle3DraftModel
from .draft.llama3_eagle import LlamaForCausalLMEagle3
from .draft.qwen3_moe_eagle import Qwen3MoEForCausalLMEagle3
from .draft.router_moe import Qwen3MoERouterForCausalLMEagle3
from .target.eagle3_target_model import (
    CustomEagle3TargetModel,
    HFEagle3TargetModel,
    SGLangEagle3TargetModel,
    get_eagle3_target_model,
)

__all__ = [
    "LlamaForCausalLMEagle3",
    "Qwen3MoERouterForCausalLMEagle3",
    "SGLangEagle3TargetModel",
    "HFEagle3TargetModel",
    "CustomEagle3TargetModel",
    "get_eagle3_target_model",
    "AutoDraftModelConfig",
    "AutoEagle3DraftModel",
]
