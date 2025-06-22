from transformers import AutoModelForCausalLM as AutoModelForCausalLMBase
from transformers import LlamaConfig
from sgl_eagle.modeling.base.llama4.llama4_17x16 import Llama4ForCausalLM
from sgl_eagle.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3

class AutoModelForCausalLM(AutoModelForCausalLMBase):
    # the model mapping is currently hardcoded, we should support lazy model mapping via registry
    _model_mapping = {
        LlamaConfig: [Llama4ForCausalLM, LlamaForCausalLMEagle3],
    }
