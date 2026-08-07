import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gates" / "run_dflash_chat_serving_gate.py"
SPEC = importlib.util.spec_from_file_location("dflash_serving_gate", SCRIPT)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def encode(text):
    return [ord(char) for char in text]


class TemplateTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, **_kwargs):
        rendered = "".join(
            f"<{message['role']}>{message.get('reasoning_content', '')}"
            f"{message.get('content', '')}"
            for message in messages
        )
        if add_generation_prompt:
            return rendered + "<assistant>"
        return rendered + "<eos>"


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

    def test_loads_the_single_training_jsonl_format(self):
        row = {
            "id": 16,
            "conversations": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "sample.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            artifact = gate.load_training_jsonl(
                str(path),
                TemplateTokenizer(),
                enable_thinking=False,
            )

        self.assertEqual(artifact["id"], 16)
        self.assertEqual(artifact["prompt_messages"], row["conversations"][:-1])
        self.assertEqual(artifact["target_suffix"], "answer<eos>")
        self.assertFalse(artifact["enable_thinking"])

    def test_training_jsonl_matches_general_qwen_normalization(self):
        row = {
            "id": 17,
            "conversations": [
                {"role": "user", "content": "question"},
                {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "training drops this",
                },
            ],
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "sample.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            artifact = gate.load_training_jsonl(
                str(path),
                TemplateTokenizer(),
                enable_thinking=False,
                system_prompt="You are a helpful assistant.",
            )

        self.assertEqual(
            artifact["prompt_messages"],
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "question"},
            ],
        )
        self.assertEqual(artifact["target_suffix"], "answer<eos>")

    def test_rejects_multiple_training_jsonl_records(self):
        row = {
            "conversations": [
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ]
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "samples.jsonl"
            path.write_text(
                json.dumps(row) + "\n" + json.dumps(row) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "exactly one record"):
                gate.load_training_jsonl(
                    str(path),
                    TemplateTokenizer(),
                    enable_thinking=False,
                )

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

    def test_accepts_dspark_as_dflash_family_serving_algorithm(self):
        result = self.evaluate(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": self.target[:16]},
                        "meta_info": {"spec_accept_length": 16.0},
                    }
                ]
            },
            server_info={"speculative_algorithm": "DSPARK"},
        )

        self.assertTrue(result["passed"])

    def test_summary_omits_large_request_and_response_fields(self):
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
        result["input_format"] = "training_jsonl"
        summary = gate.result_summary(result, "serving-gate.json")

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["spec_accept_length"], 16.0)
        self.assertEqual(summary["target_prefix_match_tokens"], 16)
        for hidden in (
            "request_payload",
            "sglang_response",
            "sglang_server_info_before",
            "choice_meta_info",
            "generated_content",
            "generated_reasoning",
        ):
            self.assertNotIn(hidden, summary)

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
        self.assertTrue(any("expected one of" in error for error in result["errors"]))
        self.assertTrue(
            any("target prefix match" in error for error in result["errors"])
        )


if __name__ == "__main__":
    unittest.main()
