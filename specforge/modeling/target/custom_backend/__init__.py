try:
    from .glm4_moe_lite import Glm4MoeLiteForCausalLM
    _GLM4_AVAILABLE = True
except ImportError:
    Glm4MoeLiteForCausalLM = None
    _GLM4_AVAILABLE = False

from .gpt_oss import GptOssForCausalLM
from .llama import LlamaForCausalLM
from .llama4 import Llama4ForCausalLM
from .phi3 import Phi3ForCausalLM
from .qwen2 import Qwen2ForCausalLM
from .qwen3 import Qwen3ForCausalLM
from .qwen3_moe import Qwen3MoeForCausalLM

__all__ = [
    "Glm4MoeLiteForCausalLM",
    "GptOssForCausalLM",
    "LlamaForCausalLM",
    "Llama4ForCausalLM",
    "Phi3ForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
]
