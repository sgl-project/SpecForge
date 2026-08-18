"""Kimi-K3 template registration and draft-config contract."""

import unittest

from specforge.data.template import TEMPLATE_REGISTRY


class TestKimiK3Template(unittest.TestCase):
    def test_registered_with_thinking_contract(self):
        t = TEMPLATE_REGISTRY.get("kimi-k3-thinking")
        self.assertIsNotNone(t)
        self.assertEqual(t.parser_type, "thinking")
        # Reasoning is stored inline in assistant content (deepspec-style
        # regenerations), so the split-field thinking path stays off.
        self.assertFalse(t.enable_thinking)
        # The assistant header must end inside the think block: the chat
        # template emits the opening think tag as part of the generation
        # prompt, so it is never model output.
        self.assertTrue(t.assistant_header.endswith("<|open|>think<|sep|>"))
        self.assertEqual(t.end_of_turn_token, "<|end_of_msg|>")
        self.assertIn("<|end_of_msg|>", t.ignore_token)


if __name__ == "__main__":
    unittest.main()
