import unittest

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from specforge.core.dflash import OnlineDFlashModel, create_dflash_block_mask
from specforge.modeling.draft.dflash import DFlashDraftModel


class TokenIdEmbedding(torch.nn.Module):
    """Deterministic token-id embedding for testing noise block construction."""

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(torch.float32).unsqueeze(-1)


class DebugDFlashHarness(OnlineDFlashModel):
    def __init__(
        self,
        mask_token_id: int,
        block_size: int,
        anchor_positions: torch.Tensor,
        global_anchor_positions: torch.Tensor | None = None,
        block_keep_mask: torch.Tensor | None = None,
    ):
        super().__init__(
            draft_model=torch.nn.Identity(),
            target_lm_head=torch.nn.Identity(),
            target_embed_tokens=TokenIdEmbedding(),
            mask_token_id=mask_token_id,
            block_size=block_size,
            attention_backend="eager",
            num_anchors=anchor_positions.shape[1],
        )
        self.manual_anchor_positions = anchor_positions
        self.manual_global_anchor_positions = (
            global_anchor_positions
            if global_anchor_positions is not None
            else anchor_positions
        )
        self.manual_block_keep_mask = (
            block_keep_mask
            if block_keep_mask is not None
            else torch.ones_like(anchor_positions, dtype=torch.bool)
        )

    def _sample_anchor_positions(self, seq_len, loss_mask, device):
        return (
            self.manual_anchor_positions.to(device),
            self.manual_block_keep_mask.to(device),
        )

    def _sample_anchor_positions_usp(self, position_ids, loss_mask, device):
        return (
            self.manual_global_anchor_positions.to(device),
            self.manual_anchor_positions.to(device),
            self.manual_block_keep_mask.to(device),
        )

    def collect_forward_state(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        use_usp_sampling: bool = False,
    ) -> dict[str, torch.Tensor]:
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        if use_usp_sampling:
            global_anchor_positions, anchor_positions, block_keep_mask = (
                self._sample_anchor_positions_usp(position_ids, loss_mask, device)
            )
        else:
            anchor_positions, block_keep_mask = self._sample_anchor_positions(
                seq_len, loss_mask, device
            )
            global_anchor_positions = (
                position_ids.gather(1, anchor_positions)
                if position_ids is not None
                else anchor_positions
            )

        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )
        noise_ids = noise_embedding.squeeze(-1).to(torch.long)
        if position_ids is not None:
            context_position_ids = position_ids[:, :seq_len].to(device=device)
        else:
            context_position_ids = (
                torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
            )
        draft_position_ids = self._create_position_ids(global_anchor_positions)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        label_offsets = torch.arange(0, self.block_size, device=device).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        gathered_loss_mask = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()
        pos_in_block = torch.arange(self.block_size, device=device).view(1, 1, -1)
        weight_mask = weight_mask * (pos_in_block > 0).float()
        weight_mask = weight_mask * gathered_loss_mask

        return {
            "anchor_positions": anchor_positions,
            "global_anchor_positions": global_anchor_positions,
            "block_keep_mask": block_keep_mask,
            "noise_ids": noise_ids,
            "context_position_ids": context_position_ids,
            "draft_position_ids": draft_position_ids,
            "full_position_ids": full_position_ids,
            "label_indices": label_indices,
            "target_ids": target_ids,
            "gathered_loss_mask": gathered_loss_mask,
            "weight_mask": weight_mask,
        }


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


def build_tiny_dflash_model_and_heads(
    block_size: int, mask_token_id: int
) -> tuple[DFlashDraftModel, torch.nn.Embedding, torch.nn.Linear]:
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
        block_size=block_size,
        num_target_layers=1,
        rope_theta=10000.0,
        layer_types=["full_attention"],
    )
    config._attn_implementation = "eager"
    config.dflash_config = {
        "target_layer_ids": [0],
        "mask_token_id": mask_token_id,
    }
    model = DFlashDraftModel(config).to(torch.float32)
    model.eval()
    embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
    lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    return model, embed_tokens, lm_head


def run_dflash_state(
    model: DFlashDraftModel,
    lm_head: torch.nn.Module,
    noise_embedding: torch.Tensor,
    target_hidden: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        output_hidden = model(
            position_ids=position_ids,
            attention_mask=attention_mask,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )
        logits = lm_head(output_hidden)
    return output_hidden, logits


def compute_weighted_loss_from_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    weight_mask: torch.Tensor,
) -> torch.Tensor:
    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = target_ids.view(-1)
    flat_weights = weight_mask.view(-1)
    loss_per_token = torch.nn.functional.cross_entropy(
        flat_logits, flat_targets, reduction="none"
    )
    return (loss_per_token * flat_weights).sum() / (flat_weights.sum() + 1e-6)


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
        self.debug_wrapper = DebugDFlashHarness(
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            anchor_positions=self.anchor_positions,
            block_keep_mask=self.block_keep_mask,
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
        model, _, _ = build_tiny_dflash_model_and_heads(
            self.block_size, self.mask_token_id
        )

        noise_embedding = torch.randn(
            1, self.anchor_positions.shape[1] * self.block_size, model.config.hidden_size
        )
        target_hidden = torch.randn(1, self.input_ids.shape[1], model.config.hidden_size)
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

        self.assertEqual(output1.shape, (1, 8, model.config.hidden_size))
        self.assertFalse(torch.isnan(output1).any())
        self.assertFalse(torch.isinf(output1).any())
        torch.testing.assert_close(output1, output2)
        self.assertFalse(torch.allclose(output1, output_relaxed))

    def test_current_usp_core_state_differs_from_full_sequence_baseline(self):
        baseline_state = self.debug_wrapper.collect_forward_state(
            input_ids=self.input_ids,
            loss_mask=self.loss_mask,
            position_ids=None,
            use_usp_sampling=False,
        )

        global_anchor_positions = self.anchor_positions.clone()
        rank0_wrapper = DebugDFlashHarness(
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            anchor_positions=torch.tensor([[2, 0]], dtype=torch.long),
            global_anchor_positions=global_anchor_positions,
            block_keep_mask=torch.tensor([[True, False]]),
        )
        rank1_wrapper = DebugDFlashHarness(
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            anchor_positions=torch.tensor([[0, 1]], dtype=torch.long),
            global_anchor_positions=global_anchor_positions,
            block_keep_mask=torch.tensor([[False, True]]),
        )

        rank0_state = rank0_wrapper.collect_forward_state(
            input_ids=self.input_ids[:, :5],
            loss_mask=self.loss_mask[:, :5],
            position_ids=torch.arange(0, 5).unsqueeze(0),
            use_usp_sampling=True,
        )
        rank1_state = rank1_wrapper.collect_forward_state(
            input_ids=self.input_ids[:, 5:],
            loss_mask=self.loss_mask[:, 5:],
            position_ids=torch.arange(5, 10).unsqueeze(0),
            use_usp_sampling=True,
        )

        self.assertTrue(
            torch.equal(
                baseline_state["global_anchor_positions"], global_anchor_positions
            )
        )
        self.assertTrue(
            torch.equal(rank0_state["global_anchor_positions"], global_anchor_positions)
        )
        self.assertTrue(
            torch.equal(rank1_state["global_anchor_positions"], global_anchor_positions)
        )

        self.assertFalse(
            torch.equal(baseline_state["noise_ids"], rank0_state["noise_ids"])
        )
        self.assertFalse(
            torch.equal(baseline_state["noise_ids"], rank1_state["noise_ids"])
        )
        self.assertTrue(
            torch.equal(
                baseline_state["full_position_ids"][:, :5],
                rank0_state["context_position_ids"],
            )
        )
        self.assertFalse(
            torch.equal(
                baseline_state["anchor_positions"],
                rank0_state["anchor_positions"],
            )
        )
        self.assertFalse(
            torch.equal(
                baseline_state["block_keep_mask"],
                rank0_state["block_keep_mask"],
            )
        )
        self.assertFalse(
            torch.equal(
                baseline_state["target_ids"],
                torch.cat([rank0_state["target_ids"], rank1_state["target_ids"]], dim=1),
            )
        )

        baseline_active_targets = baseline_state["target_ids"][
            baseline_state["weight_mask"].bool()
        ]
        usp_active_targets = torch.cat(
            [
                rank0_state["target_ids"][rank0_state["weight_mask"].bool()],
                rank1_state["target_ids"][rank1_state["weight_mask"].bool()],
            ]
        )
        self.assertFalse(torch.equal(baseline_active_targets, usp_active_targets))

    def test_current_usp_forward_output_differs_from_baseline(self):
        model, embed_tokens, lm_head = build_tiny_dflash_model_and_heads(
            self.block_size, self.mask_token_id
        )
        baseline_wrapper = DebugDFlashHarness(
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            anchor_positions=self.anchor_positions,
            block_keep_mask=self.block_keep_mask,
        )
        baseline_wrapper.embed_tokens = embed_tokens
        baseline_state = baseline_wrapper.collect_forward_state(
            input_ids=self.input_ids,
            loss_mask=self.loss_mask,
            position_ids=None,
            use_usp_sampling=False,
        )
        baseline_target_hidden = torch.randn(
            1, self.input_ids.shape[1], model.config.hidden_size
        )
        baseline_mask = build_eager_attention_mask(
            build_reference_bool_mask(
                self.anchor_positions,
                self.block_keep_mask,
                context_len=self.input_ids.shape[1],
                block_size=self.block_size,
            ),
            dtype=torch.float32,
        )
        baseline_output, baseline_logits = run_dflash_state(
            model=model,
            lm_head=lm_head,
            noise_embedding=embed_tokens(baseline_state["noise_ids"]),
            target_hidden=baseline_target_hidden,
            position_ids=baseline_state["full_position_ids"],
            attention_mask=baseline_mask,
        )

        global_anchor_positions = self.anchor_positions.clone()
        rank0_wrapper = DebugDFlashHarness(
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            anchor_positions=torch.tensor([[2, 0]], dtype=torch.long),
            global_anchor_positions=global_anchor_positions,
            block_keep_mask=torch.tensor([[True, False]]),
        )
        rank1_wrapper = DebugDFlashHarness(
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            anchor_positions=torch.tensor([[0, 1]], dtype=torch.long),
            global_anchor_positions=global_anchor_positions,
            block_keep_mask=torch.tensor([[False, True]]),
        )
        rank0_wrapper.embed_tokens = embed_tokens
        rank1_wrapper.embed_tokens = embed_tokens
        rank0_state = rank0_wrapper.collect_forward_state(
            input_ids=self.input_ids[:, :5],
            loss_mask=self.loss_mask[:, :5],
            position_ids=torch.arange(0, 5).unsqueeze(0),
            use_usp_sampling=True,
        )
        rank1_state = rank1_wrapper.collect_forward_state(
            input_ids=self.input_ids[:, 5:],
            loss_mask=self.loss_mask[:, 5:],
            position_ids=torch.arange(5, 10).unsqueeze(0),
            use_usp_sampling=True,
        )
        rank0_mask = build_eager_attention_mask(
            build_reference_bool_mask(
                rank0_state["anchor_positions"],
                rank0_state["block_keep_mask"],
                context_len=5,
                block_size=self.block_size,
            ),
            dtype=torch.float32,
        )
        rank1_mask = build_eager_attention_mask(
            build_reference_bool_mask(
                rank1_state["anchor_positions"],
                rank1_state["block_keep_mask"],
                context_len=5,
                block_size=self.block_size,
            ),
            dtype=torch.float32,
        )
        rank0_target_hidden = baseline_target_hidden[:, :5, :].clone()
        rank1_target_hidden = baseline_target_hidden[:, 5:, :].clone()
        rank0_output, rank0_logits = run_dflash_state(
            model=model,
            lm_head=lm_head,
            noise_embedding=embed_tokens(rank0_state["noise_ids"]),
            target_hidden=rank0_target_hidden,
            position_ids=rank0_state["full_position_ids"],
            attention_mask=rank0_mask,
        )
        rank1_output, rank1_logits = run_dflash_state(
            model=model,
            lm_head=lm_head,
            noise_embedding=embed_tokens(rank1_state["noise_ids"]),
            target_hidden=rank1_target_hidden,
            position_ids=rank1_state["full_position_ids"],
            attention_mask=rank1_mask,
        )

        usp_output = torch.cat([rank0_output, rank1_output], dim=1)
        usp_logits = torch.cat([rank0_logits, rank1_logits], dim=1)
        self.assertFalse(torch.allclose(baseline_output, usp_output))
        self.assertFalse(torch.allclose(baseline_logits, usp_logits))

    def test_current_usp_loss_differs_from_baseline(self):
        model, embed_tokens, lm_head = build_tiny_dflash_model_and_heads(
            self.block_size, self.mask_token_id
        )
        baseline_wrapper = DebugDFlashHarness(
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            anchor_positions=self.anchor_positions,
            block_keep_mask=self.block_keep_mask,
        )
        baseline_wrapper.embed_tokens = embed_tokens
        baseline_state = baseline_wrapper.collect_forward_state(
            input_ids=self.input_ids,
            loss_mask=self.loss_mask,
            position_ids=None,
            use_usp_sampling=False,
        )
        shared_target_hidden = torch.randn(
            1, self.input_ids.shape[1], model.config.hidden_size
        )
        baseline_mask = build_eager_attention_mask(
            build_reference_bool_mask(
                self.anchor_positions,
                self.block_keep_mask,
                context_len=self.input_ids.shape[1],
                block_size=self.block_size,
            ),
            dtype=torch.float32,
        )
        _, baseline_logits = run_dflash_state(
            model=model,
            lm_head=lm_head,
            noise_embedding=embed_tokens(baseline_state["noise_ids"]),
            target_hidden=shared_target_hidden,
            position_ids=baseline_state["full_position_ids"],
            attention_mask=baseline_mask,
        )
        baseline_loss = compute_weighted_loss_from_logits(
            baseline_logits,
            baseline_state["target_ids"],
            baseline_state["weight_mask"],
        )

        global_anchor_positions = self.anchor_positions.clone()
        rank0_wrapper = DebugDFlashHarness(
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            anchor_positions=torch.tensor([[2, 0]], dtype=torch.long),
            global_anchor_positions=global_anchor_positions,
            block_keep_mask=torch.tensor([[True, False]]),
        )
        rank1_wrapper = DebugDFlashHarness(
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            anchor_positions=torch.tensor([[0, 1]], dtype=torch.long),
            global_anchor_positions=global_anchor_positions,
            block_keep_mask=torch.tensor([[False, True]]),
        )
        rank0_wrapper.embed_tokens = embed_tokens
        rank1_wrapper.embed_tokens = embed_tokens
        rank0_state = rank0_wrapper.collect_forward_state(
            input_ids=self.input_ids[:, :5],
            loss_mask=self.loss_mask[:, :5],
            position_ids=torch.arange(0, 5).unsqueeze(0),
            use_usp_sampling=True,
        )
        rank1_state = rank1_wrapper.collect_forward_state(
            input_ids=self.input_ids[:, 5:],
            loss_mask=self.loss_mask[:, 5:],
            position_ids=torch.arange(5, 10).unsqueeze(0),
            use_usp_sampling=True,
        )
        rank0_mask = build_eager_attention_mask(
            build_reference_bool_mask(
                rank0_state["anchor_positions"],
                rank0_state["block_keep_mask"],
                context_len=5,
                block_size=self.block_size,
            ),
            dtype=torch.float32,
        )
        rank1_mask = build_eager_attention_mask(
            build_reference_bool_mask(
                rank1_state["anchor_positions"],
                rank1_state["block_keep_mask"],
                context_len=5,
                block_size=self.block_size,
            ),
            dtype=torch.float32,
        )
        _, rank0_logits = run_dflash_state(
            model=model,
            lm_head=lm_head,
            noise_embedding=embed_tokens(rank0_state["noise_ids"]),
            target_hidden=shared_target_hidden[:, :5, :],
            position_ids=rank0_state["full_position_ids"],
            attention_mask=rank0_mask,
        )
        _, rank1_logits = run_dflash_state(
            model=model,
            lm_head=lm_head,
            noise_embedding=embed_tokens(rank1_state["noise_ids"]),
            target_hidden=shared_target_hidden[:, 5:, :],
            position_ids=rank1_state["full_position_ids"],
            attention_mask=rank1_mask,
        )
        rank0_loss_num = torch.nn.functional.cross_entropy(
            rank0_logits.view(-1, rank0_logits.size(-1)),
            rank0_state["target_ids"].view(-1),
            reduction="none",
        )
        rank0_loss_num = (rank0_loss_num * rank0_state["weight_mask"].view(-1)).sum()
        rank0_weight = rank0_state["weight_mask"].sum()
        rank1_loss_num = torch.nn.functional.cross_entropy(
            rank1_logits.view(-1, rank1_logits.size(-1)),
            rank1_state["target_ids"].view(-1),
            reduction="none",
        )
        rank1_loss_num = (rank1_loss_num * rank1_state["weight_mask"].view(-1)).sum()
        rank1_weight = rank1_state["weight_mask"].sum()
        usp_loss = (rank0_loss_num + rank1_loss_num) / (
            rank0_weight + rank1_weight + 1e-6
        )

        self.assertFalse(torch.allclose(baseline_loss, usp_loss))


if __name__ == "__main__":
    unittest.main()
