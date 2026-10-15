"""The Flash-Next drafter's training blocks line up with the serving query block.

vLLM / vLLM-Ascend DSpark (sample_from_anchor) drafts from the bonus token
x_{p+1}: the context is the target's states for positions <= p, the bonus sits
at position p+1 in slot 0, and slot j predicts x_{p+2+j}. With anchor a = p+1,
training must place the block at positions a.., put x_a in slot 0, supervise
x_{a+1}.., and read no target state at or after a -- h_a is the state the
target predicts x_{a+1} from. A drafter trained with h_a visible learns to
copy the answer from it and loses its acceptance at serving time, where h_a
does not exist yet; a DeepSeek-V4 DSpark fine-tune lost its serving
acceptance exactly this way.

These tests pin the alignment for the Qwen MoE DSpark drafter Flash-Next
trains: gated attention, partial RoPE and routed experts, under both
attention implementations the NPU recipe can pick.
"""

import unittest
from unittest import mock

import torch
from torch import nn
from torch.testing import assert_close

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from tests.test_modeling.test_qwen_moe_dspark import tiny_draft_config

BLOCK, WIDTH, TAPS, LENGTH = 7, 64, 2, 40
ANCHORS = (12, 25)


def wrapper(attention_backend):
    from specforge.modeling.auto import AutoDraftModel

    torch.manual_seed(0)
    config = tiny_draft_config()
    config._attn_implementation = attention_backend
    return OnlineDSparkModel(
        draft_model=AutoDraftModel.from_config(config),
        target_lm_head=nn.Linear(WIDTH, config.vocab_size, bias=False),
        target_embed_tokens=nn.Embedding(config.vocab_size, WIDTH),
        mask_token_id=config.dflash_config["mask_token_id"],
        block_size=BLOCK,
        attention_backend=attention_backend,
        num_anchors=len(ANCHORS),
    ).eval()


class FlashNextAnchorAlignmentTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(3)
        self.input_ids = torch.randint(0, 298, (1, LENGTH), generator=generator)
        self.hidden = torch.randn(1, LENGTH, TAPS * WIDTH, generator=generator)
        self.anchors = torch.tensor([list(ANCHORS)])
        self.keep = torch.ones(1, len(ANCHORS), dtype=torch.bool)

    def blocks(self, model, states):
        """Each block's output, [num_blocks, block_size, width]."""

        with (
            mock.patch.object(
                model,
                "_sample_anchor_positions",
                return_value=(self.anchors, self.keep),
            ),
            torch.no_grad(),
        ):
            output = model._forward_draft_blocks(
                self.input_ids, states, torch.ones(1, LENGTH)
            )[2]
        return output.view(len(ANCHORS), BLOCK, WIDTH)

    def test_positions_slot_zero_and_labels(self):
        model = wrapper("sdpa")
        self.assertEqual(
            model._create_position_ids(self.anchors).view(len(ANCHORS), BLOCK).tolist(),
            [list(range(a, a + BLOCK)) for a in ANCHORS],
        )
        self.assertEqual(int(getattr(model.draft_model, "draft_position_offset", 0)), 0)
        self.assertFalse(getattr(model.draft_model, "include_anchor_context", False))

        noise = model._create_noise_embed(self.input_ids, self.anchors, self.keep)
        embedding = model.embed_tokens.weight
        for block, anchor in enumerate(ANCHORS):
            # Slot 0 carries x_a (the bonus token); the rest are mask tokens.
            assert_close(noise[0, block * BLOCK], embedding[self.input_ids[0, anchor]])
            assert_close(
                noise[0, block * BLOCK + 1],
                embedding[model.mask_token_id],
            )

        target_ids, _, _ = model._build_dspark_labels_and_mask(
            self.input_ids, torch.ones(1, LENGTH), self.anchors, self.keep
        )
        for block, anchor in enumerate(ANCHORS):
            self.assertEqual(
                target_ids[0, block].tolist(),
                self.input_ids[0, anchor + 1 : anchor + 1 + BLOCK].tolist(),
            )

    def test_no_block_reads_its_anchor_row_or_later(self):
        for backend in ("sdpa", "eager"):
            with self.subTest(attention=backend):
                model = wrapper(backend)
                clean = self.blocks(model, self.hidden)

                # Rows from the first anchor on: invisible to the first block,
                # partly visible to the second.
                first = self.hidden.clone()
                first[:, ANCHORS[0] :] = torch.randn_like(first[:, ANCHORS[0] :]) * 100
                moved = self.blocks(model, first)
                assert_close(moved[0], clean[0], rtol=0, atol=0)
                self.assertGreater((moved[1] - clean[1]).abs().max().item(), 1e-3)

                # Rows from the second anchor on: invisible to both.
                second = self.hidden.clone()
                second[:, ANCHORS[1] :] = (
                    torch.randn_like(second[:, ANCHORS[1] :]) * 100
                )
                assert_close(self.blocks(model, second), clean, rtol=0, atol=0)

    def test_the_row_before_the_anchor_is_read(self):
        # The guard above must not pass by reading no context at all.
        for backend in ("sdpa", "eager"):
            with self.subTest(attention=backend):
                model = wrapper(backend)
                clean = self.blocks(model, self.hidden)
                for block, anchor in enumerate(ANCHORS):
                    nudged = self.hidden.clone()
                    nudged[:, anchor - 1] += 1.0
                    moved = self.blocks(model, nudged)
                    self.assertGreater(
                        (moved[block] - clean[block]).abs().max().item(), 1e-4
                    )


if __name__ == "__main__":
    unittest.main()
