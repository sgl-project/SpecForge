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


if __name__ == "__main__":
    unittest.main()
