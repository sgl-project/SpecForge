import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead


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


if __name__ == "__main__":
    unittest.main()
