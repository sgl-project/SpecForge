import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from transformers import Qwen2Config

from specforge.modeling.target.target_head import TargetHead
from specforge.modeling.target.target_utils import load_target_config


class TargetHeadPreprocessTest(unittest.TestCase):
    def test_shifts_loss_mask_with_inputs_and_target(self):
        input_ids = torch.tensor([[10, 11, 12, 13]])
        target = torch.tensor([[[20], [21], [22], [23]]])
        loss_mask = torch.tensor([[0, 0, 1, 1]])

        shifted_ids, shifted_target, shifted_mask = TargetHead.preprocess(
            None,
            input_ids,
            target,
            loss_mask,
        )

        self.assertEqual(shifted_ids.tolist(), [[11, 12, 13, 0]])
        self.assertEqual(shifted_target.squeeze(-1).tolist(), [[21, 22, 23, 0]])
        self.assertEqual(shifted_mask.squeeze(-1).tolist(), [[0, 1, 1, 0]])

    def test_shifts_batched_disjoint_and_boundary_masks(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        target = torch.arange(20).reshape(2, 5, 2)
        loss_mask = torch.tensor(
            [[True, False, True, True, False], [False, False, False, False, True]]
        )

        _, _, shifted_mask = TargetHead.preprocess(
            None,
            input_ids,
            target,
            loss_mask,
        )

        self.assertEqual(shifted_mask.dtype, torch.bool)
        self.assertEqual(
            shifted_mask.squeeze(-1).tolist(),
            [
                [False, True, True, False, False],
                [False, False, False, True, False],
            ],
        )

    def test_single_token_sequence_has_no_trainable_eagle_row(self):
        _, _, shifted_mask = TargetHead.preprocess(
            None,
            torch.tensor([[7]]),
            torch.tensor([[[11, 12]]]),
            torch.tensor([[1]]),
        )

        self.assertEqual(shifted_mask.squeeze(-1).tolist(), [[0]])


class TargetHeadLoadingTest(unittest.TestCase):
    embed_key = "model.embed_tokens.weight"
    head_key = "lm_head.weight"

    def _checkpoint(self, directory, tensors, *, tie_word_embeddings):
        Qwen2Config(
            vocab_size=4,
            hidden_size=3,
            intermediate_size=6,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            tie_word_embeddings=tie_word_embeddings,
        ).save_pretrained(directory)
        save_file(tensors, Path(directory) / "model.safetensors")

    def test_loads_lm_head_from_untied_checkpoint(self):
        lm_head = torch.full((4, 3), 7.0)
        with tempfile.TemporaryDirectory() as tmp:
            self._checkpoint(
                tmp,
                {self.embed_key: torch.ones(4, 3), self.head_key: lm_head},
                tie_word_embeddings=False,
            )
            head = TargetHead(load_target_config(tmp))
            head._load_weights(tmp)

        torch.testing.assert_close(head.fc.weight, lm_head)

    def test_tied_checkpoint_uses_available_weight(self):
        weight = torch.full((4, 3), 5.0)
        for key in (self.embed_key, self.head_key):
            with (
                self.subTest(key=key),
                tempfile.TemporaryDirectory() as tmp,
                patch(
                    "specforge.modeling.target.target_head.get_local_device",
                    return_value="cpu",
                ),
            ):
                self._checkpoint(tmp, {key: weight}, tie_word_embeddings=True)
                head = TargetHead.from_pretrained(tmp, dtype=torch.bfloat16)
                torch.testing.assert_close(head.fc.weight, weight.to(torch.bfloat16))
                self.assertFalse(head.training)
                self.assertFalse(head.fc.weight.requires_grad)

    def test_tied_checkpoint_preserves_stored_lm_head(self):
        embedding = torch.ones(4, 3)
        logger = "specforge.modeling.target.target_head"
        for dtype, value, tied in (
            (torch.float32, 1.0, True),
            (torch.float32, 7.0, False),
            (torch.float32, 1.001, False),
            (torch.bfloat16, 1.001, True),
        ):
            with (
                self.subTest(dtype=dtype, value=value),
                tempfile.TemporaryDirectory() as tmp,
            ):
                lm_head = torch.full((4, 3), value)
                self._checkpoint(
                    tmp,
                    {self.embed_key: embedding, self.head_key: lm_head},
                    tie_word_embeddings=True,
                )
                log_check = (
                    self.assertNoLogs(logger, level="WARNING")
                    if tied
                    else self.assertLogs(logger, level="WARNING")
                )
                with (
                    patch(
                        "specforge.modeling.target.target_head.get_local_device",
                        return_value="cpu",
                    ),
                    log_check,
                ):
                    head = TargetHead.from_pretrained(tmp, dtype=dtype)
                torch.testing.assert_close(head.fc.weight, lm_head.to(dtype))

    def test_untied_checkpoint_without_lm_head_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._checkpoint(
                tmp, {self.embed_key: torch.ones(4, 3)}, tie_word_embeddings=False
            )
            head = TargetHead(load_target_config(tmp))
            with self.assertRaisesRegex(RuntimeError, "lm_head.weight"):
                head._load_weights(tmp)

    def test_tied_checkpoint_missing_both_keys_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._checkpoint(
                tmp, {"other.weight": torch.ones(4, 3)}, tie_word_embeddings=True
            )
            head = TargetHead(load_target_config(tmp))
            with self.assertRaisesRegex(RuntimeError, self.embed_key):
                head._load_weights(tmp, tie_weights=True)


if __name__ == "__main__":
    unittest.main()
