import re
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
from transformers import PreTrainedTokenizer

from .template import ChatTemplate
from specforge.utils import print_on_rank0
__all__ = ["GeneralParser", "HarmonyParser", "DeepSeekParser", "DeepSeek3Parser"]


class Parser(ABC):

    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        self.tokenizer = tokenizer
        self.chat_template = chat_template

    @abstractmethod
    def parse(
        self, conversation: "Conversation", max_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parse the conversation into a list of tensors.

        Args:
            conversation: The conversation to parse.

        Returns:
            A list of tensors: [input_ids, loss_mask]
        """


_harmony_encoding = None


class GeneralParser(Parser):

    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        super().__init__(tokenizer, chat_template)
        self.system_prompt = chat_template.system_prompt
        self.user_message_separator = (
            f"{chat_template.end_of_turn_token or ''}{chat_template.user_header}"
        )
        self.assistant_message_separator = (
            f"{chat_template.end_of_turn_token or ''}{chat_template.assistant_header}"
        )

    def parse(
        self,
        conversation: "Conversation",
        max_length: int,
        preformatted: bool = False,
        **kwargs,
    ) -> Dict[str, List[torch.Tensor]]:
        if not preformatted:
            messages = []

            if conversation[0]["role"] == "system":
                warnings.warn(
                    f"The first message is from system, we will use the system prompt from the data and ignore the system prompt from the template"
                )
                messages.append(
                    {"role": "system", "content": conversation[0]["content"]}
                )
                conversation = conversation[1:]
            else:
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})

            convroles = ["user", "assistant"]
            for j, sentence in enumerate(conversation):
                role = sentence["role"]
                if role != convroles[j % 2]:
                    warnings.warn(
                        f"Conversation truncated due to unexpected role '{role}'. Expected '{convroles[j % 2]}'."
                    )
                    break
                messages.append(sentence)

            conversation = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, **kwargs
            )
            # print_on_rank0(f"conversation = {conversation}")
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        encoding = self.tokenizer(
            conversation,
            return_offsets_mapping=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        offsets = encoding.offset_mapping[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        # Find spans of assistant responses using regex
        assistant_pattern = (
            re.escape(self.assistant_message_separator)
            + r"(.*?)(?="
            + re.escape(self.user_message_separator)
            + "|$)"
        )
        for match in re.finditer(assistant_pattern, conversation, re.DOTALL):
            # Assistant response text span (excluding assistant_header itself)
            assistant_start_char = match.start(1)
            assistant_end_char = match.end(1)

            # Mark tokens overlapping with assistant response
            for idx, (token_start, token_end) in enumerate(offsets):
                # Token is part of the assistant response span
                if token_end <= assistant_start_char:
                    continue  # token before assistant text
                if token_start > assistant_end_char:
                    continue  # token after assistant text
                loss_mask[idx] = 1
        # print_on_rank0(f"input_ids = {input_ids}, loss_mask = {loss_mask}")

        # 重写loss_mask逻辑：学special token 后的所有输出，包括eos_token

        return input_ids, loss_mask


class HarmonyParser(Parser):

    def build_single_turn_prompt(
        self,
        user_msg: str,
        analysis_message: str,
        commentary_message: str,
        final_message: str,
        reasoning_level: str,
    ) -> str:
        """Embed user message into the required prompt template."""

        prompt_text = f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-06-28\n\nReasoning: {reasoning_level.lower()}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        prompt_text += f"<|start|>user<|message|>{user_msg}<|end|>"
        if analysis_message:
            prompt_text += f"<|start|>assistant<|channel|>analysis<|message|>{analysis_message}<|end|>"
        if commentary_message:
            prompt_text += f"<|start|>assistant<|channel|>commentary<|message|>{commentary_message}<|end|>"
        if final_message:
            prompt_text += (
                f"<|start|>assistant<|channel|>final<|message|>{final_message}<|end|>"
            )
        return prompt_text

    def parse(
        self, conversation: "Conversation", max_length: int, preformatted: bool = False
    ) -> List[torch.Tensor]:
        if not preformatted:
            user_message = None
            analysis_message = None
            commentary_message = None
            final_message = None
            reasoning_level = "Low"

            for j, message in enumerate(conversation):
                if message["role"] == "user":
                    user_message = message["content"]
                if message["role"] == "assistant_analysis":
                    analysis_message = message["content"]
                elif message["role"] == "assistant_commentary":
                    commentary_message = message["content"]
                elif message["role"] == "assistant_final":
                    final_message = message["content"]
                elif message["role"] == "assistant_reasoning_effort":
                    reasoning_level = message["content"]

            conversation = self.build_single_turn_prompt(
                user_message,
                analysis_message,
                commentary_message,
                final_message,
                reasoning_level,
            )

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        encoding = self.tokenizer(
            conversation,
            return_offsets_mapping=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        offsets = encoding.offset_mapping[0]
        loss_mask = torch.zeros(len(input_ids), dtype=torch.long)

        # Find spans of assistant responses using regex
        response = "<|end|>".join(
            conversation.split("<|end|><|start|>user<|message|>")[1].split("<|end|>")[
                1:
            ]
        )
        num_response_chars = len(response)
        num_system_chars = len(conversation) - num_response_chars

        # Mark tokens overlapping with assistant response
        for idx, (char_start, char_end) in enumerate(offsets):
            if char_end <= num_system_chars:
                continue
            loss_mask[idx] = 1
        return input_ids, loss_mask


class DeepSeekParser(Parser):
    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        super().__init__(tokenizer, chat_template)
        self.system_prompt = chat_template.system_prompt
        self.user_message_separator = (
            f"{chat_template.end_of_turn_token}{chat_template.user_header}"
        )
        self.assistant_message_separator = (
            f"{chat_template.end_of_turn_token}{chat_template.assistant_header}"
        )
        self.bos = chat_template.bos_token

    def parse(
        self,
        conversation: "Conversation",
        max_length: int,
        preformatted: bool = False,
        **kwargs,
    ) -> Dict[str, List[torch.Tensor]]:
        if not preformatted:
            messages = []
            labels = []
            if conversation[0]["role"] == "system":
                warnings.warn(
                    f"The first message is from system, we will use the system prompt from the data and ignore the system prompt from the template"
                )
                messages.append(
                    {"role": "system", "content": conversation[0]["content"]}
                )
                conversation = conversation[1:]
            else:
                if self.system_prompt:
                    messages.append({"role": "system", "content": self.system_prompt})


            for j, msg in enumerate(conversation):
                role = msg["role"]
                if role == "assistant":
                    if not msg.get("content"): msg["content"] = "" # 保证apply_chat_template编码正确
                    labels.append(msg)

                messages.append(msg)


            conversation = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False, **kwargs
            ) # 所有inputs

            conversation_labels = self.tokenizer.apply_chat_template(
                labels, tokenize=False, add_generation_prompt=False, **kwargs
            ).split(self.bos)[-1] #  个性化 <think> </think> 有问题



        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        encoding = self.tokenizer(
            conversation,
            return_offsets_mapping=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        encoding_label = self.tokenizer(
            conversation_labels,
            return_offsets_mapping=True,
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = encoding.input_ids[0]
        label_ids = encoding_label.input_ids[0]
        if len(input_ids) > max_length:
            print_on_rank0(f"data len > max length")
            return None, None
        loss_mask = create_mask(input_ids, label_ids)

        return input_ids, loss_mask


def create_mask(input_ids, lable_ids):
    """
        找到input_ids里label_ids中位置的, 返回一个与mask列表，包含的部分标记为1，非包含的部分标记为0
    """
    # result = [0] * len(input_ids)
    i, j = 0, 0  # i遍历text1，j遍历text2
    loss_mask = torch.zeros(len(input_ids), dtype=torch.long)
    while i < len(input_ids) and j < len(lable_ids):
        if input_ids[i] == lable_ids[j]:
            # result[i] = 1
            loss_mask[i] = 1
            j += 1  # 匹配成功，移动到text2的下一个字符
        i += 1
    # print("result", result)
    # return result
    return loss_mask
class DeepSeek3Parser(Parser): # 改进
    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        super().__init__(tokenizer, chat_template)
        self.system_prompt = chat_template.system_prompt
        self.user_message_separator = (
            f"{chat_template.end_of_turn_token}{chat_template.user_header}"
        )
        self.assistant_message_separator = (
            f"{chat_template.end_of_turn_token}{chat_template.assistant_header}"
        )
        self.bos = chat_template.bos_token
        self.eos = chat_template.end_of_turn_token

    def parse(
        self,
        conversation: "Conversation",
        max_length: int,
        preformatted: bool = False,
        **kwargs,
    ) -> Dict[str, List[torch.Tensor]]:
        if not preformatted:
            input_ids = []
            loss_mask = []
            if conversation[0]["role"] == "system":
                warnings.warn(
                    f"The first message is from system, we will use the system prompt from the data and ignore the system prompt from the template"
                )
                self.system_prompt = conversation[0]["content"]
                conversation = conversation[1:]


            # encode system
            if self.system_prompt:
                system_ids = self.tokenizer.encode(self.system_prompt, add_special_tokens=True) #
                input_ids.extend(system_ids)
                loss_mask.extend([0] * len(system_ids))

            for j, msg in enumerate(conversation):
                role = msg["role"]
                if role == "assistant":
                    assistant_token = "<｜Assistant｜></think>"
                    assistant_token_ids = self.tokenizer.encode(assistant_token, add_special_tokens=False)
                    # if msg.get("tool_calls"):  # TODO
                    #    f"<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>search_history_order<｜tool▁sep｜>{}<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"
                    assistant_content = assistant_token + msg.get("content", "") + self.eos
                    assistant_ids = self.tokenizer.encode(assistant_content, add_special_tokens=False)
                    input_ids.extend(assistant_ids)
                    loss_mask.extend([0] * len(assistant_token_ids) + [1] * (len(assistant_ids) - len(assistant_token_ids)))
                elif role == "user":
                    user_content = "<｜User｜>" + msg.get("content", "")
                    user_ids = self.tokenizer.encode(user_content, add_special_tokens=False)
                    input_ids.extend(user_ids)
                    loss_mask.extend([0] * len(user_ids))
                elif role == "tool":
                    pass


        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        assert len(input_ids) == len(loss_mask)

        if len(input_ids) > max_length:
            print_on_rank0(f"data len > max length")
            return None, None
        input_ids = torch.tensor(input_ids)
        loss_mask = torch.tensor(loss_mask)

        return input_ids, loss_mask



