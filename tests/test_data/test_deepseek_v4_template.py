"""DeepSeek-V4 template registration and rendering contract."""

import unittest

from specforge.data.template import TEMPLATE_REGISTRY


class TestDeepseekV4Template(unittest.TestCase):
    def test_registered_with_thinking_contract(self):
        t = TEMPLATE_REGISTRY.get("deepseek-v4")
        self.assertIsNotNone(t)
        self.assertEqual(t.parser_type, "thinking")
        self.assertFalse(t.enable_thinking)
        self.assertEqual(t.assistant_header, "<｜Assistant｜>")
        self.assertEqual(t.user_header, "<｜User｜>")
        self.assertEqual(t.end_of_turn_token, "<｜end▁of▁sentence｜>")
        # The checkpoint ships no chat template, so ours must carry its own Jinja.
        self.assertIsNotNone(t.jinja_chat_template)
        self.assertIn("<｜begin▁of▁sentence｜>", t.jinja_chat_template)
        self.assertIn("</think>", t.jinja_chat_template)

    def test_jinja_matches_reference_encoder_basic_chat(self):
        from jinja2 import Environment

        t = TEMPLATE_REGISTRY.get("deepseek-v4")
        env = Environment()
        rendered = env.from_string(t.jinja_chat_template).render(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "2 + 2 = 4."},
            ],
            add_generation_prompt=False,
        )
        self.assertEqual(
            rendered,
            "<｜begin▁of▁sentence｜>You are a helpful assistant."
            "<｜User｜>What is 2+2?"
            "<｜Assistant｜></think>2 + 2 = 4.<｜end▁of▁sentence｜>",
        )


if __name__ == "__main__":
    unittest.main()
