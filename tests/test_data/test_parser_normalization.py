import unittest

from specforge.data.parse import GeneralParser, ThinkingParser
from specforge.data.template import ChatTemplate


class DummyTokenizer:
    pad_token_id = 0
    unk_token_id = 0
    bos_token = None

    def __init__(self):
        self.messages = None
        self.template_kwargs = None

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        **kwargs,
    ):
        self.messages = messages
        self.template_kwargs = kwargs
        return "".join(
            f"<{message['role']}>{message['content']}</eot>" for message in messages
        )

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

    def _ids(self, text, max_length=None):
        import torch

        values = [ord(char) % 251 for char in text]
        if max_length is not None:
            values = values[:max_length]
        return torch.tensor(values, dtype=torch.long)


class TestParserNormalization(unittest.TestCase):
    def test_general_parser_normalizes_sharegpt_keys_and_drops_leading_non_user(self):
        tokenizer = DummyTokenizer()
        parser = GeneralParser(
            tokenizer,
            ChatTemplate(
                assistant_header="<assistant>",
                user_header="<user>",
                system_prompt=None,
                end_of_turn_token="</eot>",
            ),
        )

        with self.assertWarnsRegex(Warning, "Dropping leading 'assistant'"):
            parser.parse(
                [
                    {"from": "gpt", "value": "orphan assistant"},
                    {"from": "human", "value": "question"},
                    {"from": "gpt", "value": "answer"},
                ],
                max_length=512,
            )

        self.assertEqual(
            tokenizer.messages,
            [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        )

    def test_thinking_mode_defaults_to_tokenizer_behavior(self):
        self.assertIsNone(ChatTemplate().enable_thinking)

    def test_thinking_mode_forwarding_and_template_precedence(self):
        for mode in (None, False, True):
            for caller_kwargs in (
                {},
                {"enable_thinking": False},
                {"enable_thinking": True},
            ):
                with self.subTest(mode=mode, caller_kwargs=caller_kwargs):
                    tokenizer = DummyTokenizer()
                    parser = ThinkingParser(
                        tokenizer,
                        ChatTemplate(
                            assistant_header="<assistant>",
                            end_of_turn_token="</eot>",
                            parser_type="thinking",
                            enable_thinking=mode,
                        ),
                    )
                    parser.parse(
                        [
                            {"role": "user", "content": "question"},
                            {"role": "assistant", "content": "answer"},
                        ],
                        max_length=512,
                        **caller_kwargs,
                    )
                    if mode is None and not caller_kwargs:
                        self.assertNotIn("enable_thinking", tokenizer.template_kwargs)
                    else:
                        expected = (
                            caller_kwargs["enable_thinking"] if mode is None else mode
                        )
                        self.assertIs(
                            tokenizer.template_kwargs["enable_thinking"], expected
                        )

    def test_thinking_parser_sanitize_keeps_tool_identity_and_drops_extras(self):
        # Tool-result rendering needs `name`/`tool_call_id` to survive
        # sanitization; chat templates place the tool name inside the block.
        parser = ThinkingParser(
            DummyTokenizer(),
            ChatTemplate(
                assistant_header="<assistant>",
                user_header="<user>",
                system_prompt=None,
                end_of_turn_token="</eot>",
                parser_type="thinking",
            ),
        )
        cleaned = parser._sanitize_message(
            {
                "role": "tool",
                "content": "sunny",
                "name": "weather",
                "tool_call_id": "call-7",
                "private": "discard",
            }
        )
        self.assertEqual(cleaned["name"], "weather")
        self.assertEqual(cleaned["tool_call_id"], "call-7")
        self.assertNotIn("private", cleaned)

    def test_general_parser_requires_end_of_turn_token(self):
        with self.assertRaisesRegex(ValueError, "end_of_turn_token"):
            GeneralParser(
                DummyTokenizer(),
                ChatTemplate(
                    assistant_header="<assistant>",
                    user_header="<user>",
                    system_prompt=None,
                    end_of_turn_token=None,
                ),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
