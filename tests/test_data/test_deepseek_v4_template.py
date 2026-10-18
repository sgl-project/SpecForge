"""DeepSeek-V4 Python encoder registration and rendering contract."""

import unittest

import torch

from specforge.data.parse import DeepSeekV4Parser
from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY


class CharacterTokenizer:
    pad_token_id = 0
    unk_token_id = 0

    @staticmethod
    def _ids(text, max_length=None):
        values = [ord(character) for character in text]
        if max_length is not None:
            values = values[:max_length]
        return torch.tensor(values, dtype=torch.long)

    def __call__(
        self,
        text,
        max_length,
        truncation,
        return_tensors,
        add_special_tokens,
    ):
        class Encoding:
            pass

        encoding = Encoding()
        encoding.input_ids = self._ids(text, max_length)[None, :]
        return encoding

    def encode(self, text, add_special_tokens=False, truncation=True, max_length=None):
        return self._ids(text, max_length).tolist()


class TestDeepseekV4Template(unittest.TestCase):
    @staticmethod
    def _render(messages, tools=None, **kwargs):
        template = TEMPLATE_REGISTRY.get("deepseek-v4")
        parser = DeepSeekV4Parser(None, template)
        return parser.apply_chat_template(messages, tool=tools, **kwargs)

    def test_registered_with_python_encoder(self):
        template = TEMPLATE_REGISTRY.get("deepseek-v4")
        self.assertIsNotNone(template)
        self.assertEqual(template.parser_type, "deepseek-v4")
        self.assertFalse(template.enable_thinking)
        self.assertEqual(template.assistant_header, "<｜Assistant｜>")
        self.assertEqual(template.user_header, "<｜User｜>")
        self.assertEqual(
            template.end_of_turn_token,
            "<｜end▁of▁sentence｜>",
        )

    def test_python_encoder_basic_chat(self):
        rendered = self._render(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2 + 2 = 4."},
            ]
        )
        self.assertEqual(
            rendered,
            "<｜begin▁of▁sentence｜>You are a helpful assistant."
            "<｜User｜>What is 2+2?"
            "<｜Assistant｜></think>2 + 2 = 4.<｜end▁of▁sentence｜>",
        )

    def test_chat_mode_drops_reasoning_content(self):
        rendered = self._render(
            [
                {"role": "user", "content": "Question"},
                {
                    "role": "assistant",
                    "reasoning_content": "must not be encoded",
                    "content": "Answer",
                },
            ]
        )
        self.assertEqual(
            rendered,
            "<｜begin▁of▁sentence｜><｜User｜>Question"
            "<｜Assistant｜></think>Answer<｜end▁of▁sentence｜>",
        )

    def test_thinking_mode_keeps_only_latest_turn_reasoning(self):
        rendered = self._render(
            [
                {"role": "user", "content": "First"},
                {
                    "role": "assistant",
                    "reasoning_content": "old reasoning",
                    "content": "First answer",
                },
                {"role": "user", "content": "Second"},
                {
                    "role": "assistant",
                    "reasoning_content": "latest reasoning",
                    "content": "Second answer",
                },
            ],
            enable_thinking=True,
        )
        self.assertEqual(
            rendered,
            "<｜begin▁of▁sentence｜><｜User｜>First"
            "<｜Assistant｜></think>First answer<｜end▁of▁sentence｜>"
            "<｜User｜>Second"
            "<｜Assistant｜><think>latest reasoning</think>"
            "Second answer<｜end▁of▁sentence｜>",
        )

    def test_generation_prompt_uses_selected_mode(self):
        messages = [{"role": "user", "content": "Question"}]
        self.assertEqual(
            self._render(messages),
            "<｜begin▁of▁sentence｜><｜User｜>Question<｜Assistant｜></think>",
        )
        self.assertEqual(
            self._render(messages, enable_thinking=True),
            "<｜begin▁of▁sentence｜><｜User｜>Question<｜Assistant｜><think>",
        )

    def test_external_tools_use_official_dsml_encoding(self):
        rendered = self._render(
            [
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                },
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "description": "Weather",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
        self.assertIn("### Available Tool Schemas", rendered)
        self.assertIn('<｜DSML｜invoke name="weather">', rendered)
        self.assertIn(
            '<｜DSML｜parameter name="city" string="true">Paris' "</｜DSML｜parameter>",
            rendered,
        )

    def test_sanitizer_preserves_encoder_tool_argument_contract(self):
        template = TEMPLATE_REGISTRY.get("deepseek-v4")
        parser = DeepSeekV4Parser(None, template)
        message = parser._sanitize_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "weather",
                            "arguments": {"city": "Paris"},
                        },
                    }
                ],
            }
        )
        self.assertEqual(
            message["tool_calls"][0]["function"]["arguments"],
            '{"city": "Paris"}',
        )

    def test_preprocessing_selects_parser_and_masks_assistant_tokens(self):
        template = TEMPLATE_REGISTRY.get("deepseek-v4")
        processed = preprocess_conversations(
            CharacterTokenizer(),
            [
                [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Answer"},
                ]
            ],
            template,
            max_length=512,
            tools=[[]],
        )
        supervised_text = "</think>Answer<｜end▁of▁sentence｜>"
        self.assertEqual(
            processed["loss_mask"][0].sum().item(),
            len(supervised_text),
        )
        self.assertTrue(torch.all(processed["attention_mask"][0] == 1))

    def test_encoder_errors_do_not_fall_back_to_generic_rendering(self):
        template = TEMPLATE_REGISTRY.get("deepseek-v4")
        parser = DeepSeekV4Parser(CharacterTokenizer(), template)
        with self.assertRaises(TypeError):
            parser.parse(
                [
                    {"role": "user", "content": ["not", "text"]},
                    {"role": "assistant", "content": "Answer"},
                ],
                max_length=512,
            )


if __name__ == "__main__":
    unittest.main()
