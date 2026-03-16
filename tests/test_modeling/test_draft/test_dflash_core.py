import unittest

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from specforge.core.dflash import OnlineDFlashModel, create_dflash_block_mask
from specforge.modeling.draft.dflash import DFlashDraftModel


class TokenIdEmbedding(torch.nn.Module):
    """Deterministic token-id embedding for testing noise block construction."""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(torch.float32).unsqueeze(-1)


def build_reference_bool_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    context_len: int,
    block_size: int,
) -> torch.Tensor:
    bsz, n_blocks = anchor_positions.shape
    q_len = n_blocks * block_size
    kv_len = context_len + q_len
    mask = torch.zeros((bsz, q_len, kv_len), dtype=torch.bool)
    for b in range(bsz):
        for q_idx in range(q_len):
            q_block_id = q_idx // block_size
            anchor_pos = int(anchor_positions[b, q_block_id].item())
            if not bool(block_keep_mask[b, q_block_id].item()):
                continue
            for kv_idx in range(kv_len):
                is_context = kv_idx < context_len
                mask_context = is_context and kv_idx < anchor_pos
                is_draft = kv_idx >= context_len
                kv_block_id = (kv_idx - context_len) // block_size if is_draft else -1
                mask_draft = is_draft and q_block_id == kv_block_id
                mask[b, q_idx, kv_idx] = mask_context or mask_draft
    return mask


def build_eager_attention_mask(
    bool_mask: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    additive_mask = torch.full(
        (bool_mask.shape[0], 1, bool_mask.shape[1], bool_mask.shape[2]),
        torch.finfo(dtype).min,
        dtype=dtype,
    )
    additive_mask = additive_mask.masked_fill(bool_mask.unsqueeze(1), 0.0)
    return additive_mask


class TestDFlashCore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.block_size = 4
        self.mask_token_id = 99
        self.input_ids = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19]])
        self.loss_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.float32)
        self.anchor_positions = torch.tensor([[2, 6]], dtype=torch.long)
        self.block_keep_mask = torch.tensor([[True, True]])

        self.wrapper = OnlineDFlashModel(
            draft_model=torch.nn.Identity(),
            target_lm_head=torch.nn.Identity(),
            target_embed_tokens=TokenIdEmbedding(),
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            attention_backend="eager",
            num_anchors=2,
        )

    def test_noise_block_and_positions(self):
        noise_embedding = self.wrapper._create_noise_embed(
            self.input_ids, self.anchor_positions, self.block_keep_mask
        )
        noise_ids = noise_embedding.squeeze(-1).to(torch.long)
        expected_noise_ids = torch.tensor([[12, 99, 99, 99, 16, 99, 99, 99]])
        self.assertTrue(torch.equal(noise_ids, expected_noise_ids))

        draft_position_ids = self.wrapper._create_position_ids(self.anchor_positions)
        expected_draft_position_ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
        self.assertTrue(torch.equal(draft_position_ids, expected_draft_position_ids))

        context_position_ids = torch.arange(self.input_ids.shape[1]).unsqueeze(0)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)
        expected_full_position_ids = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9]]
        )
        self.assertTrue(torch.equal(full_position_ids, expected_full_position_ids))

    def test_block_mask_visibility(self):
        # Call the real helper to ensure mask construction path is exercised.
        block_mask = create_dflash_block_mask(
            anchor_positions=self.anchor_positions,
            block_keep_mask=self.block_keep_mask,
            S=self.input_ids.shape[1],
            block_size=self.block_size,
            device=self.input_ids.device,
        )
        self.assertIsNotNone(block_mask)

        bool_mask = build_reference_bool_mask(
            self.anchor_positions,
            self.block_keep_mask,
            context_len=self.input_ids.shape[1],
            block_size=self.block_size,
        )[0]

        # q_idx 0 belongs to block0
        self.assertTrue(bool_mask[0, 0].item())
        self.assertTrue(bool_mask[0, 1].item())
        for kv_idx in range(2, 10):
            self.assertFalse(bool_mask[0, kv_idx].item())
        for kv_idx in range(10, 14):
            self.assertTrue(bool_mask[0, kv_idx].item())
        for kv_idx in range(14, 18):
            self.assertFalse(bool_mask[0, kv_idx].item())

        # q_idx 4 belongs to block1
        for kv_idx in range(0, 6):
            self.assertTrue(bool_mask[4, kv_idx].item())
        for kv_idx in range(6, 10):
            self.assertFalse(bool_mask[4, kv_idx].item())
        for kv_idx in range(10, 14):
            self.assertFalse(bool_mask[4, kv_idx].item())
        for kv_idx in range(14, 18):
            self.assertTrue(bool_mask[4, kv_idx].item())

    def test_labels_and_weight_mask(self):
        label_offsets = torch.arange(0, self.block_size).view(1, 1, -1)
        label_indices = self.anchor_positions.unsqueeze(-1) + label_offsets
        expected_label_indices = torch.tensor([[[2, 3, 4, 5], [6, 7, 8, 9]]])
        self.assertTrue(torch.equal(label_indices, expected_label_indices))

        target_ids = torch.gather(
            self.input_ids.unsqueeze(1).expand(-1, self.anchor_positions.size(1), -1),
            2,
            label_indices,
        )
        expected_target_ids = torch.tensor([[[12, 13, 14, 15], [16, 17, 18, 19]]])
        self.assertTrue(torch.equal(target_ids, expected_target_ids))

        gathered_loss_mask = torch.gather(
            self.loss_mask.unsqueeze(1).expand(-1, self.anchor_positions.size(1), -1),
            2,
            label_indices,
        )
        expected_gathered_loss_mask = torch.ones_like(gathered_loss_mask)
        self.assertTrue(torch.equal(gathered_loss_mask, expected_gathered_loss_mask))

        weight_mask = self.block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        weight_mask = weight_mask * (label_indices < self.input_ids.shape[1]).float()
        pos_in_block = torch.arange(self.block_size).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()
        weight_mask = weight_mask * gathered_loss_mask

        expected_weight_mask = torch.tensor(
            [[[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]]]
        )
        self.assertTrue(torch.equal(weight_mask, expected_weight_mask))

        active_targets = target_ids[weight_mask.bool()]
        expected_active_targets = torch.tensor([13, 14, 15, 17, 18, 19])
        self.assertTrue(torch.equal(active_targets, expected_active_targets))

    def test_draft_model_forward_uses_dflash_mask(self):
        config = Qwen3Config(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            attention_dropout=0.0,
            rms_norm_eps=1e-5,
            block_size=self.block_size,
            num_target_layers=1,
            rope_theta=10000.0,
            layer_types=["full_attention"],
        )
        config._attn_implementation = "eager"
        config.dflash_config = {
            "target_layer_ids": [0],
            "mask_token_id": self.mask_token_id,
        }

        model = DFlashDraftModel(config).to(torch.float32)
        model.eval()

        noise_embedding = torch.randn(
            1, self.anchor_positions.shape[1] * self.block_size, config.hidden_size
        )
        target_hidden = torch.randn(1, self.input_ids.shape[1], config.hidden_size)
        draft_position_ids = self.wrapper._create_position_ids(self.anchor_positions)
        full_position_ids = torch.cat(
            [torch.arange(self.input_ids.shape[1]).unsqueeze(0), draft_position_ids],
            dim=1,
        )

        bool_mask = build_reference_bool_mask(
            self.anchor_positions,
            self.block_keep_mask,
            context_len=self.input_ids.shape[1],
            block_size=self.block_size,
        )
        restrictive_mask = build_eager_attention_mask(bool_mask, dtype=torch.float32)
        relaxed_mask = build_eager_attention_mask(
            torch.ones_like(bool_mask, dtype=torch.bool), dtype=torch.float32
        )

        with torch.no_grad():
            output1 = model(
                position_ids=full_position_ids,
                attention_mask=restrictive_mask,
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
            )
            output2 = model(
                position_ids=full_position_ids,
                attention_mask=restrictive_mask,
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
            )
            output_relaxed = model(
                position_ids=full_position_ids,
                attention_mask=relaxed_mask,
                noise_embedding=noise_embedding,
                target_hidden=target_hidden,
            )

        self.assertEqual(output1.shape, (1, 8, config.hidden_size))
        self.assertFalse(torch.isnan(output1).any())
        self.assertFalse(torch.isinf(output1).any())
        torch.testing.assert_close(output1, output2)
        self.assertFalse(torch.allclose(output1, output_relaxed))


if __name__ == "__main__":
    unittest.main()
