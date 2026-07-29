import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gates" / "run_dflash_chat_serving_gate.py"
SPEC = importlib.util.spec_from_file_location("dflash_serving_gate", SCRIPT)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def encode(text):
    return [ord(char) for char in text]


class TestDflashServingGate(unittest.TestCase):
    def setUp(self):
        self.target = "abcdefghijklmnop target continuation"
        self.artifact = {
            "prompt_messages": [
                {
                    "role": "user",
                    "content": "question",
                    "reasoning_content": "must not be sent",
                }
            ],
            "target_suffix": self.target,
            "enable_thinking": False,
        }
        self.payload = gate.build_chat_payload(self.artifact, "test-model", 16)

    def evaluate(self, response, server_info=None):
        return gate.evaluate_response(
            response_json=response,
            server_info=server_info or {"speculative_algorithm": "DFLASH"},
            payload=self.payload,
            target_ids=encode(self.target),
            encode=encode,
            block_size=16,
        )

    def test_payload_is_non_reasoning_chat_history(self):
        self.assertEqual(
            self.payload["chat_template_kwargs"], {"enable_thinking": False}
        )
        self.assertTrue(self.payload["return_meta_info"])
        self.assertEqual(gate.request_messages_with_reasoning_content(self.payload), 0)
        self.assertNotIn("reasoning_content", self.payload["messages"][0])

    def test_passes_clean_choice_meta_info_block(self):
        result = self.evaluate(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": self.target[:16]},
                        "meta_info": {"spec_accept_length": 16.0},
                    }
                ]
            }
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["target_prefix_match_tokens"], 16)
        self.assertEqual(result["clean_block_tokens"], 16)

    def test_reasoning_and_content_are_combined_for_structured_target(self):
        target = "reasoning answer"
        payload = gate.build_chat_payload(
            {
                "prompt_messages": [{"role": "user", "content": "question"}],
                "target_suffix": target,
                "enable_thinking": True,
            },
            "test-reasoning-model",
            16,
        )
        result = gate.evaluate_response(
            response_json={
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "reasoning ",
                            "content": "answer",
                        },
                        "meta_info": {"spec_accept_length": 16.0},
                    }
                ]
            },
            server_info={"speculative_algorithm": "DFLASH"},
            payload=payload,
            target_ids=encode(target),
            encode=encode,
            block_size=16,
        )

        self.assertTrue(result["passed"])
        self.assertTrue(payload["chat_template_kwargs"]["enable_thinking"])

    def test_rejects_root_meta_info_instead_of_choice_meta_info(self):
        result = self.evaluate(
            {
                "meta_info": {"spec_accept_length": 16.0},
                "choices": [
                    {"message": {"role": "assistant", "content": self.target[:16]}}
                ],
            }
        )

        self.assertFalse(result["passed"])
        self.assertIn(
            "missing choices[0].meta_info.spec_accept_length", result["errors"]
        )

    def test_rejects_non_dflash_or_diverged_prefix(self):
        result = self.evaluate(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "wrong"},
                        "meta_info": {"spec_accept_length": 16.0},
                    }
                ]
            },
            {"speculative_algorithm": None},
        )

        self.assertFalse(result["passed"])
        self.assertTrue(any("expected 'DFLASH'" in error for error in result["errors"]))
        self.assertTrue(
            any("target prefix match" in error for error in result["errors"])
        )


if __name__ == "__main__":
    unittest.main()
