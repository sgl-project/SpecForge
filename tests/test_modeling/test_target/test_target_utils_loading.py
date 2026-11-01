import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from specforge.modeling.target.target_utils import (
    TargetEmbeddingsAndHead,
    load_target_config,
)


class TargetEmbeddingsAndHeadLoadingTest(unittest.TestCase):
    embed_key = "model.embed_tokens.weight"
    head_key = "lm_head.weight"

    def _module(self):
        config = SimpleNamespace(vocab_size=4, hidden_size=3, pad_token_id=None)
        return TargetEmbeddingsAndHead(config)

    def _single_file_checkpoint(self, directory, tensors):
        checkpoint = Path(directory) / "model.safetensors"
        save_file(tensors, checkpoint)
        return checkpoint

    def test_single_file_checkpoint_loads_required_tensors(self):
        module = self._module()
        embedding = torch.full_like(module.embed_tokens.weight, 3.0)
        lm_head = torch.full_like(module.lm_head.weight, 7.0)

        with tempfile.TemporaryDirectory() as tmp:
            self._single_file_checkpoint(
                tmp,
                {
                    self.embed_key: embedding,
                    self.head_key: lm_head,
                },
            )
            loaded = module._load_weights(
                tmp, self.embed_key, self.head_key, tie_weights=False
            )

        self.assertEqual({self.embed_key, self.head_key}, loaded)
        torch.testing.assert_close(module.embed_tokens.weight, embedding)
        torch.testing.assert_close(module.lm_head.weight, lm_head)

    def test_single_file_checkpoint_fails_when_required_tensor_is_missing(self):
        for missing_key in (self.embed_key, self.head_key):
            with (
                self.subTest(missing_key=missing_key),
                tempfile.TemporaryDirectory() as tmp,
            ):
                module = self._module()
                tensors = {
                    self.embed_key: torch.ones_like(module.embed_tokens.weight),
                    self.head_key: torch.ones_like(module.lm_head.weight),
                }
                del tensors[missing_key]
                self._single_file_checkpoint(tmp, tensors)

                with self.assertRaisesRegex(RuntimeError, missing_key):
                    module._load_weights(
                        tmp, self.embed_key, self.head_key, tie_weights=False
                    )

    def test_sharded_checkpoint_fails_for_missing_index_or_shard_tensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = self._module()
            shard = Path(tmp) / "model-00001.safetensors"
            save_file(
                {self.embed_key: torch.ones_like(module.embed_tokens.weight)},
                shard,
            )
            index = Path(tmp) / "model.safetensors.index.json"
            index.write_text(
                json.dumps({"weight_map": {self.embed_key: shard.name}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, self.head_key):
                module._load_weights(
                    tmp, self.embed_key, self.head_key, tie_weights=False
                )

        with tempfile.TemporaryDirectory() as tmp:
            module = self._module()
            shard = Path(tmp) / "model-00001.safetensors"
            save_file(
                {self.embed_key: torch.ones_like(module.embed_tokens.weight)},
                shard,
            )
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

            with self.assertRaisesRegex(RuntimeError, self.head_key):
                module._load_weights(
                    tmp, self.embed_key, self.head_key, tie_weights=False
                )

    def test_tied_checkpoint_resolves_available_weights(self):
        logger = "specforge.modeling.target.target_utils"
        for embed_value, head_value, dtype, tied in (
            (5.0, None, torch.float32, True),
            (None, 7.0, torch.float32, True),
            (5.0, 5.0, torch.float32, True),
            (5.0, 7.0, torch.float32, False),
            (1.0, 1.001, torch.float32, False),
            (1.0, 1.001, torch.bfloat16, True),
        ):
            with (
                self.subTest(embed=embed_value, head=head_value, dtype=dtype),
                tempfile.TemporaryDirectory() as tmp,
            ):
                module = self._module().to(dtype=dtype)
                tensors = {
                    key: torch.full((4, 3), value)
                    for key, value in (
                        (self.embed_key, embed_value),
                        (self.head_key, head_value),
                    )
                    if value is not None
                }
                self._single_file_checkpoint(tmp, tensors)
                log_check = (
                    self.assertNoLogs(logger, level="WARNING")
                    if tied
                    else self.assertLogs(logger, level="WARNING")
                )
                with log_check:
                    loaded = module._load_weights(
                        tmp, self.embed_key, self.head_key, tie_weights=True
                    )

                self.assertEqual(set(tensors), loaded)
                self.assertEqual(
                    module.lm_head.weight is module.embed_tokens.weight, tied
                )
                expected_embed = embed_value if embed_value is not None else head_value
                expected_head = head_value if head_value is not None else embed_value
                torch.testing.assert_close(
                    module.embed_tokens.weight,
                    torch.full((4, 3), expected_embed, dtype=dtype),
                )
                torch.testing.assert_close(
                    module.lm_head.weight,
                    torch.full((4, 3), expected_head, dtype=dtype),
                )

    def test_tied_checkpoint_accepts_same_embedding_and_head_key(self):
        module = self._module()
        embedding = torch.full_like(module.embed_tokens.weight, 6.0)

        with tempfile.TemporaryDirectory() as tmp:
            self._single_file_checkpoint(tmp, {self.embed_key: embedding})
            loaded = module._load_weights(
                tmp,
                self.embed_key,
                self.embed_key,
                tie_weights=True,
            )

        self.assertEqual({self.embed_key}, loaded)
        self.assertIs(module.lm_head.weight, module.embed_tokens.weight)
        torch.testing.assert_close(module.embed_tokens.weight, embedding)

    def test_shape_mismatch_fails_closed(self):
        for bad_key in (self.embed_key, self.head_key):
            with (
                self.subTest(bad_key=bad_key),
                tempfile.TemporaryDirectory() as tmp,
            ):
                module = self._module()
                tensors = {
                    self.embed_key: torch.ones_like(module.embed_tokens.weight),
                    self.head_key: torch.ones_like(module.lm_head.weight),
                }
                tensors[bad_key] = torch.ones(1, 1)
                self._single_file_checkpoint(tmp, tensors)

                with self.assertRaisesRegex(
                    RuntimeError, f"Shape mismatch for {bad_key}"
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

        logger = "specforge.modeling.target.target_utils"
        for tie_weights in (False, True):
            with (
                self.subTest(tie_weights=tie_weights),
                tempfile.TemporaryDirectory() as tmp,
                patch(
                    "specforge.modeling.target.target_utils.load_target_config",
                    return_value=config,
                ),
            ):
                config.tie_word_embeddings = tie_weights
                self._single_file_checkpoint(
                    tmp,
                    {
                        self.embed_key: torch.full((6, 3), 2.0),
                        self.head_key: torch.full((6, 3), 8.0),
                    },
                )
                log_check = (
                    self.assertLogs(logger, level="WARNING")
                    if tie_weights
                    else self.assertNoLogs(logger, level="WARNING")
                )
                with log_check:
                    module = TargetEmbeddingsAndHead.from_pretrained(
                        tmp, device="cpu", dtype=torch.float32
                    )

                torch.testing.assert_close(
                    module.embed_tokens.weight, torch.full((6, 3), 2.0)
                )
                torch.testing.assert_close(
                    module.lm_head.weight, torch.full((6, 3), 2.0)
                )
                self.assertIsNot(module.lm_head.weight, module.embed_tokens.weight)
                self.assertEqual(module.lm_head_mup_folded, 4.0)
                self.assertFalse(module.lm_head.weight.requires_grad)
                self.assertFalse(module.embed_tokens.weight.requires_grad)
                self.assertFalse(module.training)

                if tie_weights:
                    self._single_file_checkpoint(
                        tmp, {self.embed_key: torch.full((6, 3), 2.0)}
                    )
                    with self.assertRaisesRegex(RuntimeError, "cannot fold"):
                        TargetEmbeddingsAndHead.from_pretrained(
                            tmp, device="cpu", dtype=torch.float32
                        )


if __name__ == "__main__":
    unittest.main()
