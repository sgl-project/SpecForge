"""Built-in chat templates stay independent of removed VLM integrations."""

import unittest

from specforge.data.template import TEMPLATE_REGISTRY


class TemplateRegistryTest(unittest.TestCase):
    def test_qwen2_vl_is_not_a_builtin_template(self):
        self.assertNotIn("qwen2-vl", TEMPLATE_REGISTRY.get_all_template_names())

    def test_deepseek_v2_uses_its_plain_text_tokenizer_headers(self):
        template = TEMPLATE_REGISTRY.get("deepseek-v2")

        self.assertEqual("User: ", template.user_header)
        self.assertEqual("Assistant: ", template.assistant_header)
        self.assertIsNone(template.system_prompt)
        self.assertEqual("<｜end▁of▁sentence｜>", template.end_of_turn_token)
        self.assertNotEqual(
            TEMPLATE_REGISTRY.get("deepseek-v3").assistant_header,
            template.assistant_header,
        )

    def test_kimi_k3_template_matches_target_xtml_contract(self):
        template = TEMPLATE_REGISTRY.get("kimi-k3-thinking")
        self.assertEqual(
            template.assistant_header,
            '<|open|>message role="assistant"<|sep|><|open|>think<|sep|>',
        )
        self.assertEqual(
            template.user_header,
            '<|open|>message role="user"<|sep|>',
        )
        self.assertEqual(template.end_of_turn_token, "<|end_of_msg|>")
        self.assertEqual(template.parser_type, "thinking")
        self.assertFalse(template.enable_thinking)
        self.assertEqual(template.ignore_token, ["<|end_of_msg|>"])


if __name__ == "__main__":
    unittest.main()
