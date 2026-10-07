"""GLM-5.3 template registration and draft-config contract."""

import unittest

from specforge.data.template import TEMPLATE_REGISTRY


class TestGlm53Template(unittest.TestCase):
    def test_registered_with_glm_contract(self):
        t = TEMPLATE_REGISTRY.get("glm-5.3")
        self.assertIsNotNone(t)
        self.assertEqual(t.parser_type, "glm")
        self.assertEqual(t.assistant_pattern_type, "glm")
        # The generation prompt ends inside the think block: the chat template
        # emits the opening tag right after the assistant header, so it is
        # never model output.
        self.assertEqual(t.assistant_header, "<|assistant|><think>")
        self.assertEqual(t.user_header, "<|user|>")
        self.assertEqual(t.end_of_turn_token, "<|user|>")
        self.assertIn("<|user|>", t.ignore_token)

    def test_matches_glm_52_layout(self):
        # GLM-5.3 renders the same role/think tokens as GLM-5.2 (verified
        # against the zai-org GLM-5.3 chat_template.jinja), so the two
        # registrations must stay in sync until the formats diverge.
        a = TEMPLATE_REGISTRY.get("glm-5.3")
        b = TEMPLATE_REGISTRY.get("glm-5.2")
        for field in (
            "assistant_header",
            "user_header",
            "system_prompt",
            "end_of_turn_token",
            "parser_type",
            "assistant_pattern_type",
            "ignore_token",
        ):
            with self.subTest(field=field):
                self.assertEqual(getattr(a, field), getattr(b, field))


if __name__ == "__main__":
    unittest.main()
