import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

import torch
from transformers import LlamaConfig

from scripts.train_peagle import resolve_mask_token_id
from specforge.core.peagle import OnlinePEagleModel, compute_peagle_metrics
from specforge.modeling.draft.peagle import PEagleDraftModel


class TestPEagleTrainingSemantics(unittest.TestCase):
    def _tiny_config(self):
        return LlamaConfig(
            vocab_size=32,
            draft_vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            max_position_embeddings=64,
            pad_token_id=0,
            rms_norm_eps=1e-5,
        )

    def test_mask_hidden_is_part_of_draft_checkpoint_state(self):
        config = self._tiny_config()
        model = PEagleDraftModel(config)

        with torch.no_grad():
            model.mask_hidden.fill_(3.0)

        reloaded = PEagleDraftModel(config)
        reloaded.load_state_dict(model.state_dict())

        torch.testing.assert_close(reloaded.mask_hidden, model.mask_hidden)

    def test_online_wrapper_uses_draft_model_mask_hidden(self):
        config = self._tiny_config()
        draft_model = PEagleDraftModel(config)
        wrapper = OnlinePEagleModel(draft_model=draft_model, mask_token_id=0)

        self.assertIs(wrapper.draft_model.mask_hidden, draft_model.mask_hidden)
        self.assertNotIn("mask_hidden", dict(wrapper.named_parameters(recurse=False)))

    def test_peagle_embeddings_are_trainable_by_default(self):
        config = self._tiny_config()
        model = PEagleDraftModel(config)

        self.assertTrue(model.embed_tokens.weight.requires_grad)

    def test_compute_metrics_masks_targets_outside_draft_vocab(self):
        logits = torch.tensor(
            [
                [
                    [0.0, 4.0],
                    [4.0, 0.0],
                    [0.0, 4.0],
                ]
            ],
            dtype=torch.float32,
        )
        targets = torch.full((1, 3, 4), -10.0, dtype=torch.float32)
        targets[0, 0, 1] = 10.0
        targets[0, 1, 2] = 10.0
        targets[0, 2, 0] = 10.0
        loss_mask = torch.ones(1, 3)
        anchor_pos = torch.tensor([0, 1, 2])
        depth = torch.tensor([0, 0, 0])
        t2d = torch.tensor([True, True, False, False])

        _loss, metrics = compute_peagle_metrics(
            logits=logits,
            targets=targets,
            loss_mask=loss_mask,
            anchor_pos=anchor_pos,
            depth=depth,
            num_depths=1,
            t2d=t2d,
        )

        self.assertEqual(metrics["position_0_acc_total"].item(), 2.0)
        self.assertEqual(metrics["position_0_acc_sum"].item(), 1.0)


class TestPEagleMaskTokenResolution(unittest.TestCase):
    def _args(self, mask_token_id=None):
        return Namespace(
            mask_token_id=mask_token_id,
            target_model_path="target",
            trust_remote_code=False,
        )

    def test_explicit_mask_token_is_validated(self):
        with self.assertRaises(ValueError):
            resolve_mask_token_id(self._args(mask_token_id=33), embedding_vocab_size=32)
        with self.assertRaises(ValueError):
            resolve_mask_token_id(self._args(mask_token_id=-1), embedding_vocab_size=32)

        self.assertEqual(
            resolve_mask_token_id(
                self._args(mask_token_id=31), embedding_vocab_size=32
            ),
            31,
        )

    @patch("scripts.train_peagle.AutoTokenizer")
    def test_tokenizer_mask_token_takes_priority(self, mock_auto_tokenizer):
        tokenizer = MagicMock()
        tokenizer.mask_token_id = 7
        mock_auto_tokenizer.from_pretrained.return_value = tokenizer

        self.assertEqual(resolve_mask_token_id(self._args(), 32), 7)

    @patch("scripts.train_peagle.AutoTokenizer")
    def test_unused_embedding_slot_takes_priority_over_pad(self, mock_auto_tokenizer):
        tokenizer = MagicMock()
        tokenizer.mask_token_id = None
        tokenizer.pad_token_id = 3
        tokenizer.eos_token_id = 4
        tokenizer.unk_token_id = 5
        tokenizer.__len__.return_value = 30
        mock_auto_tokenizer.from_pretrained.return_value = tokenizer

        self.assertEqual(resolve_mask_token_id(self._args(), 32), 30)

    @patch("scripts.train_peagle.AutoTokenizer")
    def test_pad_fallback_when_no_mask_or_unused_slot(self, mock_auto_tokenizer):
        tokenizer = MagicMock()
        tokenizer.mask_token_id = None
        tokenizer.pad_token_id = 3
        tokenizer.eos_token_id = 4
        tokenizer.unk_token_id = 5
        tokenizer.__len__.return_value = 32
        mock_auto_tokenizer.from_pretrained.return_value = tokenizer

        self.assertEqual(resolve_mask_token_id(self._args(), 32), 3)

    @patch("scripts.train_peagle.AutoTokenizer")
    def test_fallback_token_must_fit_embedding_vocab(self, mock_auto_tokenizer):
        tokenizer = MagicMock()
        tokenizer.mask_token_id = None
        tokenizer.pad_token_id = 33
        tokenizer.eos_token_id = None
        tokenizer.unk_token_id = None
        tokenizer.__len__.return_value = 32
        mock_auto_tokenizer.from_pretrained.return_value = tokenizer

        with self.assertRaises(ValueError):
            resolve_mask_token_id(self._args(), 32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
