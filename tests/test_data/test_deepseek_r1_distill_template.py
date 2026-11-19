"""DeepSeek-R1-Distill template registration and loss-mask contract."""

import unittest

import torch

from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY

EOS = "<｜end▁of▁sentence｜>"


class R1DistillTokenizer:
    """Character-level tokenizer that renders the R1-Distill chat format.

    One token per character keeps loss-mask positions aligned with the
    rendered text, so the test needs no checkpoint download.
    """

    pad_token_id = 0
    unk_token_id = 0
    bos_token = "<｜begin▁of▁sentence｜>"

    def __init__(self):
        self.rendered = None

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=False,
        tools=None,
        **kwargs,
    ):
        parts = [self.bos_token]
        for message in messages:
            if message["role"] == "user":
                parts.append("<｜User｜>" + message["content"])
            elif message["role"] == "assistant":
                parts.append("<｜Assistant｜>" + message["content"] + EOS)
        self.rendered = "".join(parts)
        return self.rendered

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

    @staticmethod
    def _ids(text, max_length=None):
        values = [ord(char) for char in text][:max_length]
        return torch.tensor(values, dtype=torch.long)


class TestDeepSeekR1DistillTemplate(unittest.TestCase):
    def test_assistant_turns_end_with_end_of_sentence(self):
        template = TEMPLATE_REGISTRY.get("deepseek-r1-distill")
        self.assertEqual(template.end_of_turn_token, EOS)
        self.assertEqual(
            template.end_of_turn_token,
            TEMPLATE_REGISTRY.get("deepseek-v3").end_of_turn_token,
        )

    def test_loss_mask_covers_each_answer_and_its_eos(self):
        tokenizer = R1DistillTokenizer()
        results = preprocess_conversations(
            tokenizer=tokenizer,
            conversations=[
                [
                    {"role": "user", "content": "Who are you?"},
                    {"role": "assistant", "content": "I am R1."},
                    {"role": "user", "content": "And now?"},
                    {"role": "assistant", "content": "Still R1."},
                ]
            ],
            chat_template=TEMPLATE_REGISTRY.get("deepseek-r1-distill"),
            max_length=512,
        )

        loss_mask = results["loss_mask"][0].squeeze(0).tolist()
        supervised = "".join(
            char for char, keep in zip(tokenizer.rendered, loss_mask) if keep
        )
        self.assertEqual(supervised, f"I am R1.{EOS}Still R1.{EOS}")


if __name__ == "__main__":
    unittest.main()
