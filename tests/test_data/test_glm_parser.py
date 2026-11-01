"""GLM reasoning and tool-use parsing with pinned, official tokenizers."""

import unittest

from transformers import AutoTokenizer

from specforge.data.parse import GLMParser
from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY


def loss_spans(tokenizer, input_ids, loss_mask):
    spans, current = [], []
    for token_id, keep in zip(input_ids.tolist(), loss_mask.tolist()):
        if keep:
            current.append(token_id)
        elif current:
            spans.append(tokenizer.decode(current, skip_special_tokens=False))
            current = []
    if current:
        spans.append(tokenizer.decode(current, skip_special_tokens=False))
    return spans


class TestGLM52Parser(unittest.TestCase):
    model = "zai-org/GLM-5.2"
    revision = "cf457fa734ab149ffef225f80893eb38c6ff5cdc"

    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model, revision=cls.revision)

    def setUp(self):
        self.template = TEMPLATE_REGISTRY.get("glm-5.2")
        self.parser = GLMParser(self.tokenizer, self.template)

    def parse(self, conversation, max_length=512, **kwargs):
        return self.parser.parse(conversation, max_length=max_length, **kwargs)

    def assert_spans(self, parsed, expected):
        self.assertEqual(loss_spans(self.tokenizer, *parsed), expected)

    def assert_rendering(self, input_ids, conversation, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        rendered = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=False, **kwargs
        )
        self.assertEqual(
            input_ids.tolist(),
            self.tokenizer.encode(rendered, add_special_tokens=False),
        )

    def tool_conversation(self):
        return [
            {"role": "user", "content": "Get the weather."},
            {
                "role": "assistant",
                "content": "<think>I need a tool.</think>",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "weather", "arguments": {}},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "Sunny"},
            {
                "role": "assistant",
                "content": "<think>I have the result.</think>It is sunny.",
            },
        ]

    def test_split_reasoning_matches_inline_reasoning(self):
        question = {"role": "user", "content": "What is 2+2?"}
        split = [
            question,
            {
                "role": "assistant",
                "reasoning_content": "Two plus two equals four.",
                "content": "4",
            },
        ]
        inline = [
            question,
            {
                "role": "assistant",
                "content": "<think>Two plus two equals four.</think>4",
            },
        ]
        parsed = self.parse(split)
        inline_parsed = self.parse(inline)
        for actual, expected in zip(parsed, inline_parsed):
            self.assertEqual(actual.tolist(), expected.tolist())
        self.assert_rendering(parsed[0], split)
        self.assert_spans(parsed, ["Two plus two equals four.</think>4"])

    def test_multiple_turns_preserve_reasoning_when_requested(self):
        conversation = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "reasoning_content": "Add two.", "content": "4"},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "reasoning_content": "Add three.", "content": "6"},
        ]
        parsed = self.parse(conversation, clear_thinking=False)
        self.assert_rendering(parsed[0], conversation, clear_thinking=False)
        self.assert_spans(parsed, ["Add two.</think>4", "Add three.</think>6"])

    def test_tool_results_and_assistant_headers_are_not_supervised(self):
        conversation = self.tool_conversation()
        parsed = self.parse(conversation)
        self.assert_rendering(parsed[0], conversation)
        self.assert_spans(
            parsed,
            [
                "I need a tool.</think><tool_call>weather</tool_call>",
                "I have the result.</think>It is sunny.",
            ],
        )

    def test_last_turn_selects_only_the_final_assistant_after_tools(self):
        self.assert_spans(
            self.parse(self.tool_conversation(), train_only_last_turn=True),
            ["I have the result.</think>It is sunny."],
        )

    def test_tool_ids_preserve_official_result_ordering(self):
        conversation = [
            {"role": "user", "content": "Get the weather in two cities."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call-{city}",
                        "type": "function",
                        "function": {"name": "weather", "arguments": {"city": city}},
                    }
                    for city in ("Paris", "London")
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-London",
                "content": "LondonResult",
            },
            {
                "role": "tool",
                "tool_call_id": "call-Paris",
                "content": "ParisResult",
            },
            {"role": "assistant", "content": "Done."},
        ]
        parsed = self.parse(conversation)
        self.assert_rendering(parsed[0], conversation)
        self.assert_spans(
            parsed,
            [
                "<tool_call>weather<arg_key>city</arg_key>"
                "<arg_value>Paris</arg_value></tool_call>"
                "<tool_call>weather<arg_key>city</arg_key>"
                "<arg_value>London</arg_value></tool_call>",
                "Done.",
            ],
        )

    def test_preformatted_role_boundaries(self):
        for boundary in (
            "<|user|>Another question.",
            "<|observation|>A tool result.",
            "<|system|>A system instruction.",
            "",  # Adjacent assistant headers must remain available to the next match.
        ):
            with self.subTest(boundary=boundary):
                conversation = (
                    "[gMASK]<sop><|user|>Question."
                    "<|assistant|><think>First thought.</think>First answer."
                    + boundary
                    + "<|assistant|><think>Second thought.</think>Second answer."
                )
                self.assert_spans(
                    self.parse(conversation, preformatted=True),
                    [
                        "First thought.</think>First answer.",
                        "Second thought.</think>Second answer.",
                    ],
                )

    def test_truncation_inside_a_tool_result_does_not_supervise_it(self):
        conversation = self.tool_conversation()
        input_ids, _ = self.parse(conversation)
        observation_id = self.tokenizer.convert_tokens_to_ids("<|observation|>")
        max_length = input_ids.tolist().index(observation_id) + 3
        parsed = self.parse(conversation, max_length=max_length)
        self.assertEqual(len(parsed[0]), max_length)
        self.assert_spans(
            parsed, ["I need a tool.</think><tool_call>weather</tool_call>"]
        )
        self.assert_spans(
            self.parse(conversation, max_length=max_length, train_only_last_turn=True),
            [],
        )

    def test_empty_think_block_and_thinking_flags_remain_compatible(self):
        conversation = [
            {"role": "user", "content": "Hello."},
            {"role": "assistant", "content": "Hi."},
        ]
        for kwargs in ({}, {"enable_thinking": True, "reasoning_effort": "low"}):
            with self.subTest(kwargs=kwargs):
                parsed = self.parse(conversation, **kwargs)
                self.assert_rendering(parsed[0], conversation, **kwargs)
                self.assert_spans(parsed, ["Hi."])

    def test_preprocessing_routes_to_glm_parser(self):
        conversation = self.tool_conversation()
        conversation[-1] = {
            "role": "assistant",
            "reasoning_content": "I have the result.",
            "content": "It is sunny.",
        }
        results = preprocess_conversations(
            self.tokenizer,
            [conversation],
            self.template,
            max_length=512,
            train_only_last_turn=True,
        )
        self.assert_spans(
            (results["input_ids"][0][0], results["loss_mask"][0][0]),
            ["I have the result.</think>It is sunny."],
        )


class TestGLM53Parser(TestGLM52Parser):
    model = "zai-org/GLM-5.3"
    revision = "aca966e4e02791568aa6a4ced368624b3d897f42"


class TestGLM53FlashParser(TestGLM52Parser):
    model = "zai-org/GLM-5.3-Flash"
    revision = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"


if __name__ == "__main__":
    unittest.main()
