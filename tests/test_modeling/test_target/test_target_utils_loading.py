import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from specforge.modeling.target.target_utils import (
    TargetEmbeddingsAndHead,
    load_target_config,
)


class _FakeSafeOpen:
    def __init__(self, tensors):
        self._tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def keys(self):
        return self._tensors.keys()

    def get_tensor(self, key):
        return self._tensors[key]


class TargetEmbeddingsAndHeadLoadingTest(unittest.TestCase):
    embed_key = "model.embed_tokens.weight"
    head_key = "lm_head.weight"

    def _module(self):
        config = SimpleNamespace(vocab_size=4, hidden_size=3, pad_token_id=None)
        return TargetEmbeddingsAndHead(config)

    def _safe_open(self, tensors):
        return patch(
            "specforge.modeling.target.target_utils.safe_open",
            side_effect=lambda *_args, **_kwargs: _FakeSafeOpen(tensors),
        )

    def _single_file_checkpoint(self, directory):
        checkpoint = Path(directory) / "model.safetensors"
        checkpoint.touch()
        return checkpoint

    def test_load_file_content_reports_only_tensors_it_copied(self):
        module = self._module()
        embedding = torch.full_like(module.embed_tokens.weight, 3.0)

        with self._safe_open({self.embed_key: embedding}):
            loaded = module._load_file_content(
                "model.safetensors",
                [self.embed_key, self.head_key],
                self.embed_key,
                self.head_key,
            )

        self.assertEqual({self.embed_key}, loaded)
        torch.testing.assert_close(module.embed_tokens.weight, embedding)

    def test_single_file_checkpoint_fails_when_required_tensor_is_missing(self):
        for missing_key in (self.embed_key, self.head_key):
            with (
                self.subTest(missing_key=missing_key),
                tempfile.TemporaryDirectory() as tmp,
            ):
                module = self._module()
                self._single_file_checkpoint(tmp)
                tensors = {
                    self.embed_key: torch.ones_like(module.embed_tokens.weight),
                    self.head_key: torch.ones_like(module.lm_head.weight),
                }
                del tensors[missing_key]

                with (
                    self._safe_open(tensors),
                    self.assertRaisesRegex(RuntimeError, missing_key),
                ):
                    module._load_weights(
                        tmp, self.embed_key, self.head_key, tie_weights=False
                    )

    def test_sharded_checkpoint_fails_for_missing_index_or_shard_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = self._module()
            index = Path(tmp) / "model.safetensors.index.json"
            index.write_text(
                json.dumps({"weight_map": {self.embed_key: "model-00001.safetensors"}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, self.head_key):
                module._load_weights(
                    tmp, self.embed_key, self.head_key, tie_weights=False
                )

        with tempfile.TemporaryDirectory() as tmp:
            module = self._module()
            shard = Path(tmp) / "model-00001.safetensors"
            shard.touch()
            index = Path(tmp) / "model.safetensors.index.json"
            index.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            self.embed_key: shard.name,
                            self.head_key: shard.name,
                        }
                    }
                ),
                encoding="utf-8",
            )
            tensors = {self.embed_key: torch.ones_like(module.embed_tokens.weight)}

            with (
                self._safe_open(tensors),
                self.assertRaisesRegex(RuntimeError, self.head_key),
            ):
                module._load_weights(
                    tmp, self.embed_key, self.head_key, tie_weights=False
                )

    def test_tied_checkpoint_requires_only_embeddings(self):
        module = self._module()
        embedding = torch.full_like(module.embed_tokens.weight, 5.0)

        with tempfile.TemporaryDirectory() as tmp:
            self._single_file_checkpoint(tmp)
            with self._safe_open({self.embed_key: embedding}):
                loaded = module._load_weights(
                    tmp, self.embed_key, self.head_key, tie_weights=True
                )

        self.assertEqual({self.embed_key}, loaded)
        self.assertIs(module.lm_head.weight, module.embed_tokens.weight)
        torch.testing.assert_close(module.lm_head.weight, embedding)

    def test_shape_mismatch_fails_closed(self):
        for bad_key in (self.embed_key, self.head_key):
            with (
                self.subTest(bad_key=bad_key),
                tempfile.TemporaryDirectory() as tmp,
            ):
                module = self._module()
                self._single_file_checkpoint(tmp)
                tensors = {
                    self.embed_key: torch.ones_like(module.embed_tokens.weight),
                    self.head_key: torch.ones_like(module.lm_head.weight),
                }
                tensors[bad_key] = torch.ones(1, 1)

                with (
                    self._safe_open(tensors),
                    self.assertRaisesRegex(
                        RuntimeError, f"Shape mismatch for {bad_key}"
                    ),
                ):
                    module._load_weights(
                        tmp, self.embed_key, self.head_key, tie_weights=False
                    )

    def test_nested_text_config_uses_padded_vocab(self):
        config = SimpleNamespace(
            text_config=SimpleNamespace(
                vocab_size=4,
                padded_vocab_size=6,
                hidden_size=3,
                pad_token_id=5,
            )
        )
        module = TargetEmbeddingsAndHead(config)
        self.assertEqual(tuple(module.embed_tokens.weight.shape), (6, 3))
        self.assertEqual(tuple(module.lm_head.weight.shape), (6, 3))
        self.assertEqual(module.embed_tokens.padding_idx, 5)

    def test_public_raw_config_fallback_preserves_nested_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "not-yet-registered",
                        "text_config": {
                            "vocab_size": 4,
                            "padded_vocab_size": 6,
                            "hidden_size": 3,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch(
                "specforge.modeling.target.target_utils.AutoConfig.from_pretrained",
                side_effect=ValueError("unknown model type"),
            ):
                config = load_target_config(tmp)

        self.assertEqual(config.text_config.hidden_size, 3)
        self.assertEqual(config.text_config.padded_vocab_size, 6)

    def test_logits_mup_multiplier_is_folded_into_frozen_head(self):
        config = SimpleNamespace(
            tie_word_embeddings=False,
            text_config=SimpleNamespace(
                vocab_size=4,
                padded_vocab_size=6,
                hidden_size=3,
                pad_token_id=None,
                logits_mup_width_multiplier=4.0,
            ),
        )

        def load_weights(instance, *_args, **_kwargs):
            instance.embed_tokens.weight.data.fill_(2.0)
            instance.lm_head.weight.data.fill_(8.0)
            return {self.embed_key, self.head_key}

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "specforge.modeling.target.target_utils.load_target_config",
                return_value=config,
            ),
            patch.object(TargetEmbeddingsAndHead, "_load_weights", load_weights),
        ):
            module = TargetEmbeddingsAndHead.from_pretrained(
                tmp,
                embed_key=self.embed_key,
                lm_head_key=self.head_key,
                device="cpu",
                dtype=torch.float32,
            )

        torch.testing.assert_close(
            module.embed_tokens.weight,
            torch.full_like(module.embed_tokens.weight, 2.0),
        )
        torch.testing.assert_close(
            module.lm_head.weight,
            torch.full_like(module.lm_head.weight, 2.0),
        )
        self.assertEqual(module.lm_head_mup_folded, 4.0)
        self.assertFalse(module.lm_head.weight.requires_grad)


if __name__ == "__main__":
    unittest.main()
