import unittest
from types import SimpleNamespace

import torch

from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY


class _CharTokenizer:
    """One character per token, so a loss mask maps back onto the rendered text."""

    pad_token_id = 0
    unk_token_id = 0

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, tools=None, **kwargs
    ):
        del tokenize, add_generation_prompt, tools
        if "enable_thinking" in kwargs:
            raise AssertionError("muse rendering must not pass enable_thinking")
        parts = []
        for message in messages:
            role = message["role"]
            content = message.get("content") or ""
            if role == "system":
                parts.append(f"<|start|>system<|message|>{content}<|eot|>")
            elif role == "user":
                parts.append(f"<|start|>user<|message|>{content}<|eot|>")
            elif role == "tool":
                name = message.get("name") or "tool"
                parts.append(f"<|start|>tool {name}<|message|>{content}<|eot|>")
            elif role == "assistant":
                reasoning = message.get("reasoning_content")
                if reasoning:
                    parts.append(
                        f"<|start|>assistant to=self<|message|>{reasoning}<|eom|>"
                    )
                tool_calls = message.get("tool_calls") or []
                if tool_calls:
                    if content:
                        parts.append(
                            "<|start|>assistant to=commentary<|message|>"
                            f"{content}<|eom|>"
                        )
                    for index, tool_call in enumerate(tool_calls):
                        name = tool_call["function"]["name"]
                        arguments = tool_call["function"]["arguments"]
                        closer = "<|eot|>" if index == len(tool_calls) - 1 else "<|eom|>"
                        parts.append(
                            f"<|start|>assistant to={name}<|message|>{arguments}{closer}"
                        )
                else:
                    parts.append(
                        f"<|start|>assistant to=user<|message|>{content}<|eot|>"
                    )
            else:
                raise AssertionError(f"unexpected role {role}")
        return "".join(parts)

    def encode(self, text, add_special_tokens=False, truncation=False, max_length=None):
        del add_special_tokens
        ids = [ord(char) for char in text]
        if truncation and max_length is not None:
            return ids[:max_length]
        return ids

    def __call__(
        self,
        text,
        max_length=None,
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    ):
        del return_tensors
        ids = self.encode(
            text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            max_length=max_length,
        )
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))


def _masked_text(result):
    input_ids = result["input_ids"][0][0].tolist()
    loss_mask = result["loss_mask"][0][0].tolist()
    return "".join(chr(token) for token, mask in zip(input_ids, loss_mask) if mask)


class MuseParserTest(unittest.TestCase):
    def test_reasoning_and_answer_are_supervised(self):
        messages = [
            [
                {"role": "user", "content": "USER_PROMPT"},
                {
                    "role": "assistant",
                    "reasoning_content": "REASONING_BODY",
                    "content": "ANSWER_BODY",
                },
            ]
        ]

        result = preprocess_conversations(
            _CharTokenizer(),
            messages,
            TEMPLATE_REGISTRY.get("muse"),
            max_length=4096,
            tools=[[]],
        )
        loss_mask = result["loss_mask"][0][0]
        masked = _masked_text(result)

        self.assertGreaterEqual(int(loss_mask.sum()), 2)
        self.assertIn("REASONING_BODY", masked)
        self.assertIn("ANSWER_BODY", masked)
        self.assertIn("<|eom|>", masked)
        self.assertIn("<|eot|>", masked)
        self.assertNotIn("USER_PROMPT", masked)
        self.assertNotIn("<|start|>assistant to=self<|message|>", masked)
        self.assertNotIn("<|start|>assistant to=user<|message|>", masked)

    def test_commentary_and_tool_calls_are_supervised(self):
        messages = [
            [
                {"role": "user", "content": "USER_PROMPT"},
                {
                    "role": "assistant",
                    "reasoning_content": "REASONING_BODY",
                    "content": "COMMENTARY_BODY",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"location": "TOOL_ARG"},
                            },
                        }
                    ],
                },
                {"role": "tool", "name": "get_weather", "content": "TOOL_RESULT"},
                {"role": "assistant", "content": "ANSWER_BODY"},
            ]
        ]

        result = preprocess_conversations(
            _CharTokenizer(),
            messages,
            TEMPLATE_REGISTRY.get("muse"),
            max_length=4096,
            tools=[[]],
        )
        masked = _masked_text(result)

        self.assertIn("REASONING_BODY", masked)
        self.assertIn("COMMENTARY_BODY", masked)
        self.assertIn("TOOL_ARG", masked)
        self.assertIn("ANSWER_BODY", masked)
        self.assertNotIn("USER_PROMPT", masked)
        self.assertNotIn("TOOL_RESULT", masked)
        self.assertNotIn("<|start|>assistant to=commentary<|message|>", masked)
        self.assertNotIn("<|start|>assistant to=get_weather<|message|>", masked)


if __name__ == "__main__":
    unittest.main()
