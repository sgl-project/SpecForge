"""Ling-3.0 (Bailing V3) template registration and loss-mask contract."""

import unittest

from transformers import AutoTokenizer

from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY

MODEL_PATH = "inclusionAI/Ling-3.0-tiny"


def _loss_spans(tokenizer, input_ids, loss_mask):
    """Decode the contiguous runs of tokens the loss mask selects."""
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


class TestLing3TemplateRegistration(unittest.TestCase):
    def test_non_thinking_header_covers_the_empty_think_block(self):
        t = TEMPLATE_REGISTRY.get("ling-3.0")
        self.assertIsNotNone(t)
        # The template emits '\n<think></think>' as part of the generation
        # prompt, so it is never model output and must sit inside the header.
        self.assertEqual(t.assistant_header, "<role>ASSISTANT</role>\n<think></think>")
        self.assertEqual(t.user_header, "<role>HUMAN</role>")
        self.assertEqual(t.end_of_turn_token, "<|role_end|>")
        self.assertEqual(t.parser_type, "ling")
        self.assertFalse(t.enable_thinking)

    def test_thinking_header_stops_at_the_opening_tag(self):
        t = TEMPLATE_REGISTRY.get("ling-3.0-thinking")
        self.assertIsNotNone(t)
        # Reasoning is supervised, so only the opening tag is excluded.
        self.assertEqual(t.assistant_header, "<role>ASSISTANT</role>\n<think>")
        self.assertEqual(t.parser_type, "thinking")
        self.assertTrue(t.enable_thinking)

    def test_ling_3_0_does_not_reuse_the_ling_2_0_entry(self):
        # Ling-flash-2.0 has no think block; sharing its header would supervise
        # Ling-3.0's boilerplate instead of the answer.
        self.assertNotEqual(
            TEMPLATE_REGISTRY.get("ling-flash-2.0").assistant_header,
            TEMPLATE_REGISTRY.get("ling-3.0").assistant_header,
        )


class TestLing3LossMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, trust_remote_code=True
        )

    def _spans(self, template_name, conversation):
        results = preprocess_conversations(
            tokenizer=self.tokenizer,
            conversations=[conversation],
            chat_template=TEMPLATE_REGISTRY.get(template_name),
            max_length=512,
        )
        return _loss_spans(
            self.tokenizer,
            results["input_ids"][0].squeeze(),
            results["loss_mask"][0].squeeze(),
        )

    def test_non_thinking_supervises_answers_only(self):
        conversation = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "It is 4."},
            {"role": "user", "content": "And 3+3?"},
            {"role": "assistant", "content": "That is 6."},
        ]
        self.assertEqual(
            self._spans("ling-3.0", conversation),
            ["It is 4.<|role_end|>", "That is 6.<|role_end|>"],
        )

    def test_thinking_supervises_reasoning_and_answer(self):
        conversation = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "<think>add two and two</think>It is 4.",
            },
        ]
        self.assertEqual(
            self._spans("ling-3.0-thinking", conversation),
            ["add two and two</think>It is 4.<|role_end|>"],
        )

    def test_non_thinking_template_on_thinking_data_masks_everything(self):
        # Regression guard: the empty-think header cannot match a turn that
        # carries reasoning, and the failure is silent (zero supervised
        # tokens), so the two templates must stay separate.
        conversation = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "<think>add two and two</think>It is 4.",
            },
        ]
        self.assertEqual(self._spans("ling-3.0", conversation), [])

    def test_thinking_flag_is_passed_explicitly(self):
        # The Bailing V3 template defaults its thinking option to on, so the
        # ling parser has to pass the flag: otherwise the draft trains on a
        # system prefix it never sees when served with thinking off. Compare
        # renderings rather than pinning the template's wording.
        conversation = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "It is 4."},
        ]
        results = preprocess_conversations(
            tokenizer=self.tokenizer,
            conversations=[conversation],
            chat_template=TEMPLATE_REGISTRY.get("ling-3.0"),
            max_length=512,
        )
        rendered = self.tokenizer.decode(
            results["input_ids"][0].squeeze(), skip_special_tokens=False
        )
        explicit_off = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
            enable_thinking=False,
        )
        default_rendering = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        self.assertEqual(rendered, explicit_off)
        self.assertNotEqual(explicit_off, default_rendering)


if __name__ == "__main__":
    unittest.main()
