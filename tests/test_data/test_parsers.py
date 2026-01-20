import json
import os
import unittest
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY


class TestTemplatePreprocessing(unittest.TestCase):
    # Configuration section
    SAVE_REFERENCE = False
    REF_DIR = os.path.join(os.path.dirname(__file__), "test_references")

    @classmethod
    def setUpClass(cls):
        """Initialize standard test data"""
        cls.max_length = 65535
        if not os.path.exists(cls.REF_DIR):
            os.makedirs(cls.REF_DIR)

        # 1. General model test data (Qwen, DeepSeek, etc.)
        cls.standard_messages = [
            [
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": "My name is Qwen2."},
                {"role": "user", "content": "How old are you?"},
                {"role": "assistant", "content": "11 years old."},
            ]
        ]

        # 2. GPT-OSS Dedicated Test Data (Including Analysis and Final Channel)
        cls.gpt_oss_messages = [
            [
                {"role": "user", "content": "Explain Quantum Physics."},
                {
                    "role": "assistant_analysis",
                    "content": "The user wants a summary of quantum physics. I should cover wave-particle duality and uncertainty principle.",
                },
                {
                    "role": "assistant_final",
                    "content": "Quantum physics is the study of matter and energy at the most fundamental level...",
                },
                {"role": "user", "content": "Explain Quantum Physics."},
                {"role": "assistant_final", "content": "I'm Qwen"},
            ]
        ]

        # 3. Tool-Use Test Data
        cls.tool_use_messages = [
            [
                {
                    "role": "user",
                    "content": "What's the weather like in Beijing today?",
                },
                {
                    "role": "assistant",
                    "content": "I'll check the current weather in Beijing for you.",
                },
                {
                    "role": "tool",
                    "content": '{"location": "Beijing", "temperature": 22, "condition": "Sunny"}',
                },
                {
                    "role": "assistant",
                    "content": "The current weather in Beijing is sunny with a temperature of 22°C.",
                },
                {
                    "role": "tool",
                    "content": '{"unit": "Celsius", "forecast": "Clear skies all day."}',
                },
                {
                    "role": "tool",
                    "content": '{"unit": "Celsius", "forecast": "Clear skies all day."}',
                },
                {
                    "role": "user",
                    "content": "Great! Can you also tell me if it will rain tomorrow?",
                },
                {
                    "role": "assistant",
                    "content": "Based on the forecast, there will be no rain tomorrow—expect clear skies all day.",
                },
            ]
        ]

    def _get_ref_path(self, template_key: str, message_label: str = "standard"):
        return os.path.join(self.REF_DIR, f"{template_key}_{message_label}_ref.json")

    def _run_template_test(
        self,
        model_name: str,
        template_key: str,
        messages: Optional[List[List[Dict[str, str]]]] = None,
    ):
        """Encapsulate common test and regression validation logic"""

        # Use the input message or the default standard message.
        target_messages = messages if messages is not None else self.standard_messages
        message_label = None
        if target_messages == self.standard_messages:
            message_label = "standard"
        elif target_messages == self.gpt_oss_messages:
            message_label = "gpt-oss"
        elif target_messages == self.tool_use_messages:
            message_label = "tool-use"
        else:
            raise ValueError("Invalid message set")
        print(f"\n>>> Running: {template_key} ({model_name}) {message_label}")

        # 1. Initialize tokenizer and template
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        chat_template = TEMPLATE_REGISTRY.get(template_key)

        # 2. Preprocess conversations
        res = preprocess_conversations(
            tokenizer, target_messages, chat_template, self.max_length
        )
        # Extract current result
        current_data = {
            "input_ids": res["input_ids"][0][0].tolist(),
            "loss_mask": res["loss_mask"][0][0].tolist(),
        }

        ref_path = self._get_ref_path(template_key, message_label)
        # 3. Branch logic: update reference or perform comparison
        if self.SAVE_REFERENCE:
            with open(ref_path, "w", encoding="utf-8") as f:
                json.dump(current_data, f)
            print(f" [INFO] Reference saved to {ref_path}")
        else:
            if not os.path.exists(ref_path):
                self.fail(
                    f"Reference file not found for {template_key}. Set SAVE_REFERENCE=True."
                )

            with open(ref_path, "r", encoding="utf-8") as f:
                ref_data = json.load(f)

            self.assertListEqual(current_data["input_ids"], ref_data["input_ids"])
            self.assertListEqual(current_data["loss_mask"], ref_data["loss_mask"])
            print(f" [PASS] Regression test passed for {template_key}")

        # 4. Debug output
        self.debug_show_loss_mask(res, tokenizer)

    @staticmethod
    def debug_show_loss_mask(res: Dict[str, Any], tokenizer: AutoTokenizer):
        input_ids = res["input_ids"][0][0].tolist()
        loss_mask = res["loss_mask"][0][0].tolist()
        RED, RESET = "\033[91m", "\033[0m"
        print("-" * 30)
        for tid, m in zip(input_ids, loss_mask):
            txt = tokenizer.decode([tid])
            txt = txt.replace("\n", "\\n")
            print(f"{RED if m == 1 else ''}{txt}{RESET}", end="")
        print("\n" + "-" * 30)

    ## The Following are tests. Each test corresponds to a specific model and template.

    def test_deepseek(self):
        self._run_template_test("deepseek-ai/DeepSeek-V3", "deepseek-v3")

    def test_deepseek_v32(self):
        self._run_template_test("deepseek-ai/DeepSeek-V3.2", "deepseek-v32")

    def test_qwen3_thinking(self):
        self._run_template_test("Qwen/Qwen3-0.6B", "qwen3-thinking")

    def test_qwen3_instruct(self):
        self._run_template_test("Qwen/Qwen3-0.6B", "qwen3-instruct")

    def test_qwen3_next_instruct(self):
        self._run_template_test("Qwen/Qwen3-Next-80B-A3B-Instruct", "qwen")

    def test_kimi_k2_thinking(self):
        self._run_template_test("moonshotai/Kimi-K2-Thinking", "kimi-k2-thinking")

    def test_kimi_k2_instruct(self):
        self._run_template_test("moonshotai/Kimi-K2-Instruct", "kimi-k2-instruct")

    def test_qwen3_next_thinking(self):
        self._run_template_test(
            "Qwen/Qwen3-Next-80B-A3B-Thinking", "qwen3-next-thinking"
        )

    def test_gpt_oss(self):
        self._run_template_test(
            "openai/gpt-oss-120b", "gpt-oss", messages=self.gpt_oss_messages
        )

    def test_ling_flash_2_0(self):
        self._run_template_test("inclusionAI/Ling-flash-2.0", "ling-flash-2.0")

    def test_qwen3_instruct_with_tools(self):
        self._run_template_test(
            "Qwen/Qwen3-0.6B", "qwen3-instruct", messages=self.tool_use_messages
        )


if __name__ == "__main__":
    unittest.main()
