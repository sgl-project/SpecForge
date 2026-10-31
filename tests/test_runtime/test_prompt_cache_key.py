"""Prompt-cache identity: keys must change when any tokenization input changes."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from specforge.training.assembly import _prompt_cache_key


def _cache_config(path: str):
    return SimpleNamespace(
        data=SimpleNamespace(
            prompts_path="",
            train_data_path=path,
            max_length=4096,
            chat_template="llama3",
            is_preformatted=False,
            train_only_last_turn=False,
            max_prompts=None,
        ),
        model=SimpleNamespace(
            target_model_path="org/target-model",
            draft_model_config="configs/draft.json",
            draft_checkpoint_path=None,
            draft_num_hidden_layers=None,
            draft_block_size=None,
            input_modality="text",
        ),
        training=SimpleNamespace(strategy="eagle3"),
    )


class TestPromptCacheKey(unittest.TestCase):
    def test_tracks_source_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "prompts.jsonl")
            path.write_text('{"text":"first"}\n', encoding="utf-8")
            config = _cache_config(str(path))
            first = _prompt_cache_key(config)
            path.write_text('{"text":"second"}\n', encoding="utf-8")
            second = _prompt_cache_key(config)

        self.assertNotEqual(first, second)

    def test_tracks_tokenizer_chat_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "prompts.jsonl")
            path.write_text('{"text":"same"}\n', encoding="utf-8")
            config = _cache_config(str(path))
            first = _prompt_cache_key(
                config, tokenizer=SimpleNamespace(chat_template="template-a")
            )
            second = _prompt_cache_key(
                config, tokenizer=SimpleNamespace(chat_template="template-b")
            )
            missing = _prompt_cache_key(config, tokenizer=None)

        self.assertNotEqual(first, second)
        self.assertNotEqual(first, missing)


if __name__ == "__main__":
    unittest.main()
