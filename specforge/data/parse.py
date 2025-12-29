import re
import warnings
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
from transformers import PreTrainedTokenizer

from .template import ChatTemplate

__all__ = ["GeneralParser", "HarmonyParser"]


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
        self.user_message_separator = f"{chat_template.end_of_turn_token}"
        self.assistant_message_separator = f"{chat_template.assistant_header}"

    def apply_chat_template(self, messages, **kwargs) -> str:
        conversation = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, **kwargs
        )
        return conversation

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

            conversation = self.apply_chat_template(messages, **kwargs)

        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.unk_token_id

        assistant_pattern = f"{re.escape(self.assistant_message_separator)}(.*?(?:{re.escape(self.chat_template.end_of_turn_token)}|$))"
        try:
            # use fast tokenizer's offset mapping to create loss mask
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

            for match in re.finditer(assistant_pattern, conversation, re.DOTALL):
                assistant_start_char = match.start(1)
                assistant_end_char = match.end(1)

                for idx, (token_start, token_end) in enumerate(offsets):
                    # if token_end <= assistant_start_char:
                    #     continue
                    # if token_start > assistant_end_char:
                    #     continue
                    if (
                        assistant_start_char
                        <= token_start
                        <= token_end
                        <= assistant_end_char
                    ):
                        loss_mask[idx] = 1

        except (NotImplementedError, TypeError, ValueError):
            assistant_pattern = (
                re.escape(self.assistant_message_separator)
                + r"([\s\S]*?"
                + re.escape(self.chat_template.end_of_turn_token)
                + r")"  # 结束符现在被包含在 match.group(1) 中
            )
            # get input_ids
            encoding = self.tokenizer(
                conversation,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            input_ids = encoding.input_ids[0]
            loss_mask = torch.zeros(len(input_ids), dtype=torch.long)
            for match in re.finditer(assistant_pattern, conversation, re.DOTALL):
                content_start_char = match.start(1)
                content_end_char = match.end(1)

                # --- Core Alternative Operation: Calculate Token Index Based on Prefix String Length ---
                # Encode the text "assistant start", the length of which is the position of the starting token.
                prefix_ids = self.tokenizer.encode(
                    conversation[:content_start_char], add_special_tokens=False
                )
                # Encodes the text "assistant end", the length of which is the position of the end token.
                full_ids = self.tokenizer.encode(
                    conversation[:content_end_char], add_special_tokens=False
                )

                start_token_idx = len(prefix_ids)
                end_token_idx = len(full_ids)

                # Handling out-of-bounds errors caused by truncation
                actual_start = min(start_token_idx, len(input_ids))
                actual_end = min(end_token_idx, len(input_ids))

                if actual_start < actual_end:
                    loss_mask[actual_start:actual_end] = 1

        return input_ids, loss_mask


class HarmonyParser(Parser):
    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        super().__init__(tokenizer, chat_template)
        self.reasoning_levels = ["low", "medium", "high"]
        self.default_reasoning_level = "low"

    def build_single_turn_prompt(
        self,
        prompt_text: str,
        role: str,
        content: str,
    ) -> str:
        """Embed user message into the required prompt template."""
        if role == "system":
            prompt_text = f"<|start|>system<|message|>{content}<|end|>"
        elif role == "assistant_reasoning_effort":
            prompt_text = f"<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\nCurrent date: 2025-06-28\n\nReasoning: {content.lower()}\n\n# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>"
        elif role == "user":
            prompt_text += f"<|start|>user<|message|>{content}<|end|>"
        elif role == "assistant_analysis":
            prompt_text += (
                f"<|start|>assistant<|channel|>analysis<|message|>{content}<|end|>"
            )
        elif role == "assistant_commentary":
            prompt_text += (
                f"<|start|>assistant<|channel|>commentary<|message|>{content}<|end|>"
            )
        elif role == "assistant_final":
            prompt_text += (
                f"<|start|>assistant<|channel|>final<|message|>{content}<|end|>"
            )
        else:
            raise ValueError(f"Unknown role: {role}")
        return prompt_text

    def parse(
        self, conversation: "Conversation", max_length: int, preformatted: bool = False
    ) -> List[torch.Tensor]:
        # conversation = process_harmony_conversations(conversation)
        if not preformatted:
            prompt_text = ""
            for j, message in enumerate(conversation):
                if j == 0 and (
                    message["role"] != "system"
                    or message["role"] != "assistant_reasoning_effort"
                ):
                    prompt_text = self.build_single_turn_prompt(
                        prompt_text,
                        "assistant_reasoning_effort",
                        self.default_reasoning_level,
                    )
                prompt_text = self.build_single_turn_prompt(
                    prompt_text, message["role"], message["content"]
                )
            conversation = prompt_text

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
        # We match `<|start|>assistant` and only extract the content following it.
        # This continues until `<|start|>user<|message|>` appears, or until the end of the string.
        pattern = re.compile(
            r"<\|start\|>assistant([\s\S]*?)(?=<\|start\|>user<\|message\|>|$)"
        )

        # Find all matching segments
        for match in pattern.finditer(conversation):
            # match.start(0) is the start index of the full match (including `<|start|>assistant`)
            # match.start(1) is the start index of the first capture group (excluding `<|start|>assistant`)
            # match.end(1) is the end index of the content
            start_char = match.start(1)
            end_char = match.end(1)

            # Map character indices to token indices
            for idx, (ts, te) in enumerate(offsets):
                # Set mask to 1 only if the token's character range falls entirely within the "content area"
                if ts >= start_char and te <= end_char:
                    loss_mask[idx] = 1

        return input_ids, loss_mask


class ThinkingParser(GeneralParser):
    def __init__(self, tokenizer: PreTrainedTokenizer, chat_template: ChatTemplate):
        super().__init__(tokenizer, chat_template)

    def apply_chat_template(self, messages, **kwargs) -> str:
        if messages[-1]["role"] == "assistant":
            conversation_history = self.tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
                add_special_tokens=False,
                **kwargs,
            )
            conversation = (
                conversation_history
                + messages[-1]["content"]
                + self.chat_template.end_of_turn_token
            )
            return conversation
        else:
            raise Exception(
                f"The last message is not assistant but {messages[-1]['role']}"
            )

    def parse(
        self,
        conversation: "Conversation",
        max_length: int,
        preformatted: bool = False,
        **kwargs,
    ) -> Dict[str, List[torch.Tensor]]:
        if self.chat_template.enable_thinking:
            kwargs["enable_thinking"] = True
        else:
            pass
        return super().parse(conversation, max_length, preformatted, **kwargs)
