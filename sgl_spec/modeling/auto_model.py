from transformers import AutoModelForCausalLM as AutoModelForCausalLMBase
from transformers import LlamaConfig
from .draft.llama3_eagle import LlamaForCausalLMEagle3



class AutoEagle3DraftModel(AutoModelForCausalLMBase):
    # the model mapping is currently hardcoded, we should support lazy model mapping via registry
    _model_mapping = {
        LlamaConfig: [LlamaForCausalLMEagle3],
    }
