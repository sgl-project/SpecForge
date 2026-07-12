import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run_qwen3_8b_dflash_serving_gate.py"
SPEC = importlib.util.spec_from_file_location("qwen3_8b_serving_gate", SCRIPT)
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def load_hf_exporter():
    names = ("specforge", "specforge.export", "specforge.export.checkpoint_io")
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name in names[:2]:
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
        checkpoint_io = types.ModuleType(names[2])
        checkpoint_io.materialize_draft = None
        checkpoint_io.resolve_training_state = None
        sys.modules[names[2]] = checkpoint_io
        path = ROOT / "specforge" / "export" / "to_hf.py"
        spec = importlib.util.spec_from_file_location("_test_to_hf", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


hf_exporter = load_hf_exporter()


def encode(text):
    return [ord(char) for char in text]


class TestQwen3DflashServingGate(unittest.TestCase):
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
        self.payload = gate.build_chat_payload(self.artifact, "qwen3-8b", 16)

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

    def test_reasoning_and_content_are_combined_for_qwen36_target(self):
        target = "reasoning answer"
        payload = gate.build_chat_payload(
            {
                "prompt_messages": [{"role": "user", "content": "question"}],
                "target_suffix": target,
                "enable_thinking": True,
            },
            "qwen3.6-27b",
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

    def test_shell_launcher_uses_matching_dflash_block_flags(self):
        launcher = ROOT / "examples/disagg/run_qwen3_8b_domino_real_serving_gate.sh"
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        text = launcher.read_text()
        self.assertIn('--speculative-num-draft-tokens "$BLOCK_SIZE"', text)
        self.assertIn('--speculative-dflash-block-size "$BLOCK_SIZE"', text)
        self.assertIn('--tp-size "$SERVING_TP"', text)
        self.assertIn("REASONING_PARSER_ARGS=(--reasoning-parser", text)
        self.assertIn("SERVING_PORT is already occupied", text)

    def test_qwen36_domino_config_contract(self):
        config = json.loads(
            (ROOT / "configs" / "qwen3.6-27b-domino-full-attention.json").read_text()
        )
        self.assertEqual(config["layer_types"], ["full_attention"] * 5)
        self.assertEqual(config["dflash_config"]["projector_type"], "domino")
        self.assertEqual(
            config["dflash_config"]["target_layer_ids"], [1, 16, 31, 46, 61]
        )
        self.assertFalse(config["dflash_config"]["shift_label"])


class TestDominoHfExport(unittest.TestCase):
    def test_model_without_load_embedding_gets_target_embedding_in_state(self):
        embedding = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
        source_key = "model.embed_tokens.weight"
        state = {"draft_state_dict": {"draft.weight": torch.ones(1)}}
        saved_state = {}

        class DominoLikeModel:
            def state_dict(self):
                return {"draft.weight": torch.zeros(1)}

            def save_pretrained(self, output_dir, *, state_dict):
                saved_state.update(state_dict)

        with tempfile.TemporaryDirectory() as tmp:
            shard = Path(tmp) / "model-00001-of-00001.safetensors"
            save_file({source_key: embedding}, shard)
            (Path(tmp) / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {source_key: shard.name}}),
                encoding="utf-8",
            )
            with (
                patch.object(hf_exporter, "resolve_training_state", return_value=state),
                patch.object(
                    hf_exporter,
                    "materialize_draft",
                    return_value=DominoLikeModel(),
                ),
            ):
                hf_exporter.export_to_hf(
                    "checkpoint",
                    "config",
                    str(Path(tmp) / "out"),
                    embedding_source=tmp,
                    embedding_key=source_key,
                )

        torch.testing.assert_close(saved_state["embed_tokens.weight"], embedding)
        torch.testing.assert_close(saved_state["draft.weight"], torch.ones(1))


if __name__ == "__main__":
    unittest.main()
