"""DeepSeek-V4 template registration and rendering contract."""

import unittest

from jinja2 import Environment

from specforge.data.template import TEMPLATE_REGISTRY


class TestDeepseekV4Template(unittest.TestCase):
    @staticmethod
    def _render(messages, **kwargs):
        def raise_exception(message):
            raise ValueError(message)

        env = Environment()
        env.globals["raise_exception"] = raise_exception
        template = TEMPLATE_REGISTRY.get("deepseek-v4").jinja_chat_template
        return env.from_string(template).render(
            messages=messages,
            tools=kwargs.pop("tools", None),
            add_generation_prompt=kwargs.pop("add_generation_prompt", False),
            enable_thinking=kwargs.pop("enable_thinking", False),
            **kwargs,
        )

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
            self._render(messages, add_generation_prompt=True),
            "<｜begin▁of▁sentence｜><｜User｜>Question" "<｜Assistant｜></think>",
        )
        self.assertEqual(
            self._render(
                messages,
                add_generation_prompt=True,
                enable_thinking=True,
            ),
            "<｜begin▁of▁sentence｜><｜User｜>Question" "<｜Assistant｜><think>",
        )

    def test_unsupported_tool_messages_fail_loudly(self):
        with self.assertRaisesRegex(ValueError, "does not support tools"):
            self._render(
                [{"role": "user", "content": "Question"}],
                tools=[{"type": "function"}],
            )

        with self.assertRaisesRegex(ValueError, "does not support role: tool"):
            self._render([{"role": "tool", "content": "result"}])


if __name__ == "__main__":
    unittest.main()
