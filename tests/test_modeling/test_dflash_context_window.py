"""dflash_config.context_window: DSpark's block-anchored context window on the draft."""

import unittest

import torch
from transformers import Qwen3Config

from specforge.modeling.draft.dflash import (
    DFlashDraftModel,
    dflash_attention_resume_contract,
)


def _config(context_window=None):
    dflash_config = {"target_layer_ids": [1]}
    if context_window is not None:
        dflash_config["context_window"] = context_window
    config = Qwen3Config(
        architectures=["DFlashDraftModel"],
        block_size=3,
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        num_target_layers=4,
        head_dim=4,
        max_position_embeddings=64,
        vocab_size=32,
        layer_types=["full_attention"],
        dflash_config=dflash_config,
    )
    config._attn_implementation = "sdpa"
    return config


class ContextWindowTest(unittest.TestCase):
    def test_config_reaches_the_model_and_the_resume_contract(self):
        model = DFlashDraftModel(_config(context_window=3))
        self.assertEqual(model.context_window, 3)
        contract = dflash_attention_resume_contract(model)
        self.assertEqual(contract["draft_attention_mask_semantics"][-1], 3)
        self.assertEqual(
            dflash_attention_resume_contract(DFlashDraftModel(_config())), {}
        )

    def test_invalid_values_are_rejected(self):
        for value in (0, -1, True, 2.5):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "context_window"):
                    DFlashDraftModel(_config(context_window=value))

    def test_generation_masks_the_full_layer_with_the_window(self):
        """spec_generate builds its own masks; the window must reach the full layer."""
        model = DFlashDraftModel(_config(context_window=2))
        hidden = torch.zeros(1, model.block_size, 8)
        masks = model._prepare_attention_mask(None, hidden, torch.zeros(1, 5, 8), None)
        mask = masks["full_attention"]
        self.assertIsNotNone(mask)
        for query in range(model.block_size):
            visible = mask[0, 0, query].nonzero().flatten().tolist()
            self.assertEqual(visible, [3, 4, 5, 6, 7])

    def test_forward_runs_with_the_window(self):
        model = DFlashDraftModel(_config(context_window=2))
        output = model(
            position_ids=torch.arange(5 + 3)[None],
            noise_embedding=torch.randn(1, 3, 8),
            target_hidden=torch.randn(1, 5, 8),
        )
        self.assertEqual(tuple(output.shape), (1, 3, 8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
