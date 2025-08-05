# Adapted from: https://github.com/sgl-project/sglang/blob/main/python/sglang/lang/chat_template.py#L13
from typing import List

from pydantic import BaseModel


class ChatTemplate(BaseModel):
    """
    This is a dataclass for the chat template.

    Args:
        assistant_header(str): The header for the assistant.
        user_header(str): The header for the user.
        system_prompt(str): The system prompt.
        end_of_turn_token(str): The end token of a turn of conversation.
    """

    assistant_header: str
    user_header: str
    system_prompt: str
    end_of_turn_token: str


class TemplateRegistry:
    """
    This is a registry for the chat template. Sgl-spec will register some common chat templates here.
    If you have a custom chat template, you can register it via the example below.

    Example:
    ```python
        from specforge.data.template import TEMPLATE_REGISTRY, ChatTemplate
        TEMPLATE_REGISTRY.register(
            name="custom",
            template=ChatTemplate(
                assistant_header="<|start_header_id|>assistant<|end_header_id|>\n\n",
                user_header="<|start_header_id|>user<|end_header_id|>",
                system_prompt="You are a helpful assistant.",
                end_of_turn_token="<|eot_id|>"
            )
        )
    ```
    """

    def __init__(self):
        self.templates = {}

    def register(self, name: str, template: ChatTemplate, override: bool = False):
        """
        Register a chat template for a model type.

        Args:
            name(str): The name of the chat template.
            template(ChatTemplate): The chat template.
            override(bool): Whether to override the existing template, default to False
        """
        assert (
            not override and name not in self.templates
        ), f"Chat template for the model type {name} has already been registered"
        self.templates[name] = template

    def get(self, name: str) -> ChatTemplate:
        """
        Get the chat template for a model type.

        Args:
            name(str): The name of the chat template.

        Returns:
            ChatTemplate: The chat template.
        """
        return self.templates[name]

    def get_all_template_names(self) -> List[str]:
        """
        Get all the template names.

        Returns:
            List[str]: The list of template names.
        """
        return list(self.templates.keys())


# global registry
TEMPLATE_REGISTRY = TemplateRegistry()

# Register the common template here
TEMPLATE_REGISTRY.register(
    name="llama3",
    template=ChatTemplate(
        assistant_header="<|start_header_id|>assistant<|end_header_id|>\n\n",
        user_header="<|start_header_id|>user<|end_header_id|>",
        system_prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
        end_of_turn_token="<|eot_id|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="llama4",
    template=ChatTemplate(
        assistant_header="<|header_start|>assistant<|header_end|>\n\n",
        user_header="<|header_start|>user<|header_end|>",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|eot|>",
    ),
)

TEMPLATE_REGISTRY.register(
    name="qwen",
    template=ChatTemplate(
        assistant_header="<|im_start|>assistant\n",
        user_header="<|im_start|>user\n",
        system_prompt="You are a helpful assistant.",
        end_of_turn_token="<|im_end|>\n",
    ),
)


TEMPLATE_REGISTRY.register(
    name="qwen3",
    template=ChatTemplate(
        # 角色头部标识
        system_header="<|im_start|>system\n",
        user_header="<|im_start|>user\n",
        assistant_header="<|im_start|>assistant\n",
        system_prompt="You are a helpful assistant.",
        tool_header="<|im_start|>tool\n",
        
        # 工具相关标记
        tools_declaration_prefix="# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>",
        tools_declaration_suffix="\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>",
        
        # 工具调用标记
        tool_call_begin="<tool_call>\n",
        tool_call_end="\n</tool_call>",
        tool_response_wrapper="<tool_response>\n{content}\n</tool_response>",
        
        # 思考过程标记
        reasoning_wrapper="<|FunctionCallBegin|>\n{reasoning}\n<|FunctionCallEnd|>\n\n",
        
        # 轮次结束标记
        end_of_turn_token="<|im_end|>\n",
        
        # 生成提示
        generation_prompt="<|im_start|>assistant\n",
        default_thinking_prompt="<|FunctionCallBegin|>\n\n</think>\n\n"
    ),
)


TEMPLATE_REGISTRY.register(
    name="kimi_k2",
    template=ChatTemplate(
        # 系统提示相关配置
        system_header="<|im_system|>system<|im_middle|>",
        system_prompt="You are a helpful assistant.",
        
        # 角色前缀配置
        user_header="<|im_user|>user<|im_middle|>",
        assistant_header="<|im_assistant|>assistant<|im_middle|>",
        tool_header="<|im_system|>tool<|im_middle|>",
        
        # 工具调用相关标记
        tool_declare_prefix="<|im_system|>tool_declare<|im_middle|>",
        tool_calls_section_begin="<|tool_calls_section_begin|>",
        tool_calls_section_end="<|tool_calls_section_end|>",
        tool_call_begin="<|tool_call_begin|>",
        tool_call_argument_begin="<|tool_call_argument_begin|>",
        tool_call_end="<|tool_call_end|>",
        
        # 媒体内容标记
        media_start="<|media_start|>image<|media_content|><|media_pad|>",
        media_end="<|media_end|>",
        
        # 轮次结束标记
        end_of_turn_token="<|im_end|>\n",
        
        # 生成提示前缀
        generation_prompt="<|im_assistant|>assistant<|im_middle|>"
    ),
)