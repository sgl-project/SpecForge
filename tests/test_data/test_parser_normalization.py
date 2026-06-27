import unittest

from specforge.data.parse import GeneralParser
from specforge.data.template import ChatTemplate


class DummyTokenizer:
    pad_token_id = 0
    unk_token_id = 0
    bos_token = None

    def __init__(self):
        self.messages = None

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        **kwargs,
    ):
        self.messages = messages
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
