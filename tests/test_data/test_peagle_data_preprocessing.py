import unittest

import torch

from datasets import Dataset
from specforge.data.preprocessing import build_eagle3_dataset
from specforge.data.template import TEMPLATE_REGISTRY, ChatTemplate


class DummyTokenizer:
    pad_token_id = 0
    unk_token_id = 0
    bos_token = None

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        **kwargs,
    ):
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
        values = [ord(char) % 251 for char in text]
        if max_length is not None:
            values = values[:max_length]
        return torch.tensor(values, dtype=torch.long)


class TestPEagleDataPreprocessing(unittest.TestCase):
    template_name = "unit-test-peagle-data"

    @classmethod
    def setUpClass(cls):
        if cls.template_name not in TEMPLATE_REGISTRY.get_all_template_names():
            TEMPLATE_REGISTRY.register(
                cls.template_name,
                ChatTemplate(
                    assistant_header="<assistant>",
                    user_header="<user>",
                    system_prompt=None,
                    end_of_turn_token="</eot>",
                ),
            )

    def test_minimum_valid_tokens_filters_empty_loss_samples(self):
        dataset = Dataset.from_list(
            [
                {
                    "conversations": [
                        {"role": "user", "content": "question"},
                        {"role": "assistant", "content": "answer"},
                    ]
                },
                {
                    "conversations": [
                        {"role": "user", "content": "question"},
                    ]
                },
            ]
        )

        result = build_eagle3_dataset(
            dataset=dataset,
            tokenizer=DummyTokenizer(),
            chat_template=self.template_name,
            max_length=128,
            num_proc=1,
            cache_dir=None,
            cache_key=None,
            minimum_valid_tokens=1,
        )

        self.assertEqual(len(result), 1)
        self.assertGreater(result[0]["loss_mask"].sum().item(), 0)

    def test_minimum_valid_tokens_rejects_negative_values(self):
        with self.assertRaisesRegex(ValueError, "minimum_valid_tokens"):
            build_eagle3_dataset(
                dataset=Dataset.from_list([]),
                tokenizer=DummyTokenizer(),
                chat_template=self.template_name,
                minimum_valid_tokens=-1,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
