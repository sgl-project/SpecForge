import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from specforge.data.parse import ThinkingParser
from specforge.data.template import TEMPLATE_REGISTRY
from specforge.training.assembly import _prompt_cache_key


def _cache_config(path: str):
    return SimpleNamespace(
        data=SimpleNamespace(
            prompts_path="",
            train_data_path=path,
            max_length=4096,
            chat_template="inkling-thinking",
            is_preformatted=False,
            train_only_last_turn=False,
            max_prompts=None,
        ),
        model=SimpleNamespace(
            target_model_path="thinkingmachines/Inkling",
            draft_model_config="configs/inkling-dspark.json",
            draft_checkpoint_path=None,
            draft_num_hidden_layers=None,
            draft_block_size=None,
            input_modality="text",
        ),
        training=SimpleNamespace(strategy="dspark"),
    )


class TestInklingTemplate(unittest.TestCase):
    def test_tool_response_identity_survives_sanitization(self):
        parser = ThinkingParser(
            SimpleNamespace(),
            TEMPLATE_REGISTRY.get("inkling-thinking"),
        )
        cleaned = parser._sanitize_message(
            {
                "role": "tool",
                "content": "sunny",
                "name": "weather",
                "tool_call_id": "call-7",
                "private": "discard",
            }
        )
        self.assertEqual(cleaned["name"], "weather")
        self.assertEqual(cleaned["tool_call_id"], "call-7")
        self.assertNotIn("private", cleaned)

    def test_prompt_cache_key_tracks_source_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "prompts.jsonl")
            path.write_text('{"text":"first"}\n', encoding="utf-8")
            config = _cache_config(str(path))
            first = _prompt_cache_key(config)
            path.write_text('{"text":"second"}\n', encoding="utf-8")
            second = _prompt_cache_key(config)

        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
