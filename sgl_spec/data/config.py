from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ModelType(Enum):
    LLAMA3 = "llama3"
    LLAMA4 = "llama4"
    QWEN = "qwen"
    CUSTOM = "custom"


# Predefined chat template configurations
CHAT_TEMPLATES = {
    ModelType.LLAMA3: {
        "assistant_header": "<|start_header_id|>assistant<|end_header_id|>\n\n",
        "user_header": "<|start_header_id|>user<|end_header_id|>",
        "system_prompt": (
            "You are a helpful, respectful and honest assistant. Always answer as "
            "helpfully as possible, while being safe.  Your answers should not "
            "include any harmful, unethical, racist, sexist, toxic, dangerous, or "
            "illegal content. Please ensure that your responses are socially unbiased "
            "and positive in nature.\n\nIf a question does not make any sense, or is "
            "not factually coherent, explain why instead of answering something not "
            "correct. If you don't know the answer to a question, please don't share "
            "false information."
        ),
    },
    ModelType.LLAMA4: {
        "assistant_header": "<|header_start|>assistant<|header_end|>\n\n",
        "user_header": "<|header_start|>user<|header_end|>",
        "system_prompt": (
            "You are a helpful, respectful and honest assistant. Always answer as "
            "helpfully as possible, while being safe.  Your answers should not "
            "include any harmful, unethical, racist, sexist, toxic, dangerous, or "
            "illegal content. Please ensure that your responses are socially unbiased "
            "and positive in nature.\n\nIf a question does not make any sense, or is "
            "not factually coherent, explain why instead of answering something not "
            "correct. If you don't know the answer to a question, please don't share "
            "false information."
        ),
    },
    ModelType.QWEN: {
        "assistant_header": "<|im_start|>assistant\n",
        "user_header": "<|im_start|>user\n",
        "system_prompt": "You are a helpful assistant.",
    },
}


@dataclass
class DataConfig:
    batch_size: int = 8  # Training batch size
    num_processes: int = 8
    max_length: int = 2048
    test_size: float = 0.1
    shuffle_seed: int = 42
    pin_memory: bool = True
    num_workers: int = 4
    load_from_cache_file: bool = True

    # Performance settings
    preprocess_batch_size: int = 1024  # Batch size for data preprocessing

    model_type: ModelType = ModelType.LLAMA4
    custom_assistant_header: Optional[str] = None
    custom_user_header: Optional[str] = None
    custom_system_prompt: Optional[str] = None

    def get_chat_template(self) -> Dict[str, str]:
        """Get the chat template configuration for the current model"""
        if self.model_type == ModelType.CUSTOM:
            return {
                "assistant_header": self.custom_assistant_header
                or CHAT_TEMPLATES[ModelType.LLAMA4]["assistant_header"],
                "user_header": self.custom_user_header
                or CHAT_TEMPLATES[ModelType.LLAMA4]["user_header"],
                "system_prompt": self.custom_system_prompt
                or CHAT_TEMPLATES[ModelType.LLAMA4]["system_prompt"],
            }
        return CHAT_TEMPLATES[self.model_type]
