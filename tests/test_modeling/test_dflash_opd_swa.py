"""Serving causality must survive the OPD replay mask construction."""

import unittest

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import (
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.algorithms.common.dflash_opd import OPDDFlash2Model
from specforge.modeling.draft.dflash2 import DFlash2DraftModel


class OPDSlidingCausalityTests(unittest.TestCase):
    def test_noncausal_block_keeps_future_draft_but_not_future_target(self):
        anchors = torch.tensor([[6, 10]])
        keep = torch.tensor([[True, False]])
        args = dict(
            anchor_positions=anchors,
            block_keep_mask=keep,
            S=12,
            block_size=8,
            device="cpu",
            sliding_window=8,
            sliding_draft_causal=False,
        )
        dense = create_dflash_sdpa_mask(**args)
        flex = create_dflash_block_mask(**args)
        for q in range(16):
            for k in range(28):
                self.assertEqual(
                    bool(dense[0, 0, q, k]),
                    bool(
                        flex.mask_mod(
                            torch.tensor(0),
                            torch.tensor(0),
                            torch.tensor(q),
                            torch.tensor(k),
                        )
                    ),
                )
        self.assertTrue(bool(dense[0, 0, :8, 12:20].all()))
        self.assertFalse(bool(dense[0, 0, :, 6:12].any()))
        self.assertFalse(bool(dense[0, 0, :, 20:].any()))
        self.assertFalse(bool(dense[0, 0, 8:].any()))
        self.assertFalse(bool(dense[0, 0, 7, :6].any()))

    def test_legacy_causal_default_preserved(self):
        args = dict(
            anchor_positions=torch.tensor([[6]]),
            block_keep_mask=torch.tensor([[True]]),
            S=12,
            block_size=8,
            device="cpu",
            sliding_window=8,
        )
        default = create_dflash_sdpa_mask(**args)
        explicit = create_dflash_sdpa_mask(**args, sliding_draft_causal=True)
        self.assertTrue(torch.equal(default, explicit))
        self.assertTrue(
            torch.equal(
                default[0, 0, :, 12:], torch.ones(8, 8, dtype=torch.bool).tril()
            )
        )

    def test_noncausal_small_window_still_excludes_old_draft_positions(self):
        args = dict(
            anchor_positions=torch.tensor([[6]]),
            block_keep_mask=torch.tensor([[True]]),
            S=12,
            block_size=8,
            device="cpu",
            sliding_window=1,
            sliding_draft_causal=False,
        )
        dense = create_dflash_sdpa_mask(**args)
        flex = create_dflash_block_mask(**args)
        self.assertTrue(
            torch.equal(dense[0, 0, :, 12:], torch.ones(8, 8, dtype=torch.bool).triu())
        )
        for q in range(8):
            for k in range(20):
                self.assertEqual(
                    bool(dense[0, 0, q, k]),
                    bool(
                        flex.mask_mod(
                            torch.tensor(0),
                            torch.tensor(0),
                            torch.tensor(q),
                            torch.tensor(k),
                        )
                    ),
                )

    def test_noncausal_config_changes_replayed_hidden_states(self):
        torch.manual_seed(19)
        config = Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            num_hidden_layers=1,
            num_target_layers=4,
            vocab_size=32,
            max_position_embeddings=64,
            layer_types=["sliding_attention"],
            sliding_window=16,
            use_sliding_window=True,
            dflash_config=dict(
                block_size=8,
                mask_token_id=31,
                conv_group_size=4,
                conv_kernel_size=2,
                selector_rank=4,
                selector_top_k=3,
                target_layer_ids=[1],
            ),
        )
        draft = DFlash2DraftModel(config)
        draft.config._attn_implementation = "eager"
        model = OPDDFlash2Model(
            draft,
            nn.Linear(16, 32, bias=False),
            nn.Embedding(32, 16),
            31,
            block_size=8,
            attention_backend="eager",
        ).eval()
        taps = torch.randn(1, 12, 16)
        replayed = {}
        with torch.no_grad():
            for causal in (False, True, None, "absent"):
                if causal == "absent":
                    del draft.config.is_causal
                else:
                    draft.config.is_causal = causal
                _, _, replayed[causal] = model._forward_draft_blocks(
                    torch.arange(12)[None],
                    taps,
                    torch.ones(1, 12),
                    anchor_positions=torch.tensor([[4]]),
                    block_keep_mask=torch.tensor([[True]]),
                )
        self.assertFalse(torch.allclose(replayed[False], replayed[True]))
        for default in (None, "absent"):
            torch.testing.assert_close(
                replayed[default], replayed[True], rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
