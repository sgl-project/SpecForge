import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers.utils.chat_template_utils import render_jinja_template

from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY

# Pinned Muse Spark chat template. The mock tokenizer below is checked
# against this text, rather than against a copy fetched at test time.
_MUSE_TEMPLATE = (
    Path(__file__).parent / "fixtures" / "muse_spark_chat_template.jinja"
).read_text(encoding="utf-8")
_ASSISTANT_SPAN = re.compile(
    r"<\|start\|>assistant to=.*?<\|message\|>([\s\S]*?(?:<\|eom\|>|<\|eot\|>))",
    re.DOTALL,
)


class _CharTokenizer:
    """One character per token, so a loss mask maps back onto the rendered text."""

    pad_token_id = 0
    unk_token_id = 0

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        **kwargs,
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
                        closer = (
                            "<|eot|>" if index == len(tool_calls) - 1 else "<|eom|>"
                        )
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


def _preprocess(messages, train_only_last_turn=False):
    return preprocess_conversations(
        _CharTokenizer(),
        [messages],
        TEMPLATE_REGISTRY.get("muse"),
        max_length=4096,
        tools=[[]],
        train_only_last_turn=train_only_last_turn,
    )


def _render_pinned(messages, tools=None):
    rendered, _ = render_jinja_template(
        [messages],
        tools=tools,
        chat_template=_MUSE_TEMPLATE,
        add_generation_prompt=False,
        current_date="2020-01-01",
    )
    return rendered[0]


class MuseParserTest(unittest.TestCase):
    def test_reasoning_and_answer_are_supervised(self):
        messages = [
            {"role": "user", "content": "USER_PROMPT"},
            {
                "role": "assistant",
                "reasoning_content": "REASONING_BODY",
                "content": "ANSWER_BODY",
            },
        ]

        result = _preprocess(messages)
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

        result = _preprocess(messages)
        masked = _masked_text(result)

        self.assertIn("REASONING_BODY", masked)
        self.assertIn("COMMENTARY_BODY", masked)
        self.assertIn("TOOL_ARG", masked)
        self.assertIn("ANSWER_BODY", masked)
        self.assertNotIn("USER_PROMPT", masked)
        self.assertNotIn("TOOL_RESULT", masked)
        self.assertNotIn("<|start|>assistant to=commentary<|message|>", masked)
        self.assertNotIn("<|start|>assistant to=get_weather<|message|>", masked)

    def test_train_only_last_turn_keeps_every_span_of_that_turn(self):
        messages = [
            {"role": "user", "content": "FIRST_USER"},
            {
                "role": "assistant",
                "reasoning_content": "FIRST_REASONING",
                "content": "FIRST_ANSWER",
            },
            {"role": "user", "content": "LAST_USER"},
            {
                "role": "assistant",
                "reasoning_content": "LAST_REASONING",
                "content": "LAST_ANSWER",
            },
        ]

        masked = _masked_text(_preprocess(messages, train_only_last_turn=True))

        self.assertIn("LAST_REASONING", masked)
        self.assertIn("LAST_ANSWER", masked)
        self.assertIn("<|eom|>", masked)
        self.assertIn("<|eot|>", masked)
        self.assertNotIn("FIRST_REASONING", masked)
        self.assertNotIn("FIRST_ANSWER", masked)
        self.assertNotIn("FIRST_USER", masked)
        self.assertNotIn("LAST_USER", masked)
        self.assertIn("FIRST_REASONING", _masked_text(_preprocess(messages)))

    def test_tool_result_starts_a_new_turn(self):
        messages = [
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

        masked = _masked_text(_preprocess(messages, train_only_last_turn=True))

        self.assertIn("ANSWER_BODY", masked)
        self.assertNotIn("REASONING_BODY", masked)
        self.assertNotIn("COMMENTARY_BODY", masked)
        self.assertNotIn("TOOL_ARG", masked)
        self.assertNotIn("TOOL_RESULT", masked)

    def test_pinned_template_matches_the_mock_assistant_spans(self):
        messages = [
            {"role": "user", "content": "FIRST_USER"},
            {
                "role": "assistant",
                "reasoning_content": "FIRST_REASONING",
                "content": "FIRST_ANSWER",
            },
            {"role": "user", "content": "LAST_USER"},
            {
                "role": "assistant",
                "reasoning_content": "LAST_REASONING",
                "content": "LAST_ANSWER",
            },
        ]
        pinned = _render_pinned(messages)
        mock = _CharTokenizer().apply_chat_template(messages, tokenize=False)
        expected = [
            "FIRST_REASONING<|eom|>",
            "FIRST_ANSWER<|eot|>",
            "LAST_REASONING<|eom|>",
            "LAST_ANSWER<|eot|>",
        ]

        self.assertEqual(_ASSISTANT_SPAN.findall(pinned), expected)
        self.assertEqual(_ASSISTANT_SPAN.findall(mock), expected)

        rendered = preprocess_conversations(
            _CharTokenizer(),
            [pinned],
            TEMPLATE_REGISTRY.get("muse"),
            max_length=20000,
            is_preformatted=True,
            tools=[[]],
            train_only_last_turn=True,
        )
        masked = _masked_text(rendered)
        self.assertIn("LAST_REASONING", masked)
        self.assertIn("LAST_ANSWER", masked)
        self.assertNotIn("FIRST_REASONING", masked)
        self.assertNotIn("FIRST_ANSWER", masked)
        self.assertNotIn("You are a helpful AI assistant.", masked)

    def test_pinned_template_supervises_tool_arguments_not_the_schema(self):
        messages = [
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
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                },
            }
        ]
        pinned = _render_pinned(messages, tools=tools)

        self.assertIn(
            "<|start|>assistant to=self<|message|>REASONING_BODY<|eom|>", pinned
        )
        self.assertIn(
            "<|start|>assistant to=commentary<|message|>COMMENTARY_BODY<|eom|>",
            pinned,
        )
        self.assertIn(
            '<atem:invoke name="get_weather">',
            pinned,
        )
        self.assertIn("TOOL_ARG", pinned)
        self.assertIn("example_function_name_1", pinned)
        self.assertIn("<|start|>tool get_weather<|message|>", pinned)

        rendered = preprocess_conversations(
            _CharTokenizer(),
            [pinned],
            TEMPLATE_REGISTRY.get("muse"),
            max_length=20000,
            is_preformatted=True,
            tools=[[]],
            train_only_last_turn=False,
        )
        masked = _masked_text(rendered)
        self.assertIn("REASONING_BODY", masked)
        self.assertIn("COMMENTARY_BODY", masked)
        self.assertIn("TOOL_ARG", masked)
        self.assertIn("ANSWER_BODY", masked)
        self.assertNotIn("TOOL_RESULT", masked)
        self.assertNotIn("example_function_name_1", masked)
        self.assertNotIn("USER_PROMPT", masked)


if __name__ == "__main__":
    unittest.main()
