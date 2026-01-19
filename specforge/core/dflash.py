# coding=utf-8
"""DFlash Training Wrapper."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.modeling.draft.dflash import DFlashDraftModel


class OnlineDFlashModel(nn.Module):
    """DFlash online training wrapper with block-wise CE loss."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id

    def prepare_noise_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Prepare noise input: first token of each block is real, rest are MASK."""
        # input_ids: [bsz, seq_len]
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # Create mask in a vectorized way - more efficient than full_like + indexing
        is_block_start = torch.arange(seq_len, device=device) % self.block_size == 0

        # Use where for single-pass creation
        noise_input_ids = torch.where(
            is_block_start.unsqueeze(0), input_ids, self.mask_token_id
        )

        return noise_input_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parallel Block-wise training forward pass.

        Uses a global attention mask to process all blocks in parallel without
        modifying the underlying modeling code.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # Truncate to multiple of block_size
        n_blocks = seq_len // self.block_size
        effective_len = n_blocks * self.block_size
        input_ids = input_ids[:, :effective_len]
        # hidden_states here is the RAW target hidden states (before projection)
        hidden_states = hidden_states[:, :effective_len, :]
        loss_mask = loss_mask[:, :effective_len]
        # Original attention mask is typically just 1s for valid tokens
        attention_mask = attention_mask[:, :effective_len]

        # 2. Prepare Inputs
        noise_input_ids = self.prepare_noise_input(input_ids)
        noise_embedding = self.embed_tokens(noise_input_ids)

        # 3. Construct Parallel Training Position IDs
        # We need Position IDs for K which has length 2*L (Context + Noise)
        # Context part: 0..L-1
        # Noise part:   0..L-1
        # This ensures that Noise at pos i uses the same RoPE embedding as Context at pos i
        pos_seq = torch.arange(effective_len, device=device)
        # Optimized: use repeat instead of cat + expand
        position_ids = pos_seq.unsqueeze(0).repeat(bsz, 2)

        # 4. Construct Parallel Attention Mask
        # The modeling code will internally concat K = [K_ctx, K_noise]
        # So K has length 2*L. Q has length L (from Noise).
        # We need a mask of shape [L, 2*L]
        dflash_attn_mask = self._create_parallel_attention_mask(effective_len, device)
        dflash_attn_mask = dflash_attn_mask.to(dtype=hidden_states.dtype)
        # Expand to batch size: [bsz, 1, L, 2*L]
        # Note: transformers usually expects [bsz, 1, Q, K]
        dflash_attn_mask = (
            dflash_attn_mask.unsqueeze(0).unsqueeze(0).expand(bsz, -1, -1, -1)
        )

        # 5. Parallel Forward Pass
        # efficient: one single forward pass for the whole sequence
        hidden = self.draft_model(
            position_ids=position_ids,  # [bsz, 2*L] (used for RoPE)
            noise_embedding=noise_embedding,  # [bsz, L, H] (Query source)
            target_hidden=hidden_states,  # [bsz, L, H] (Context source)
            attention_mask=dflash_attn_mask,  # [bsz, 1, L, 2*L]
        )

        # 6. Compute Loss
        # Create mask for valid loss positions (skip block 0, skip block starts)
        dflash_loss_mask_base = create_dflash_loss_mask(
            effective_len, self.block_size, device
        )
        # Use in-place multiplication to save memory
        combined_mask = loss_mask * dflash_loss_mask_base.unsqueeze(0)

        logits = self.lm_head(hidden)

        # Calculate Loss - optimized version
        # Get active indices before flattening to reduce operations
        active_mask = combined_mask > 0.5

        # Use view instead of reshape where possible (view is faster)
        vocab_size = logits.size(-1)
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = input_ids.view(-1)
        active_mask_flat = active_mask.view(-1)

        # Index once for both loss and accuracy
        active_logits = logits_flat[active_mask_flat]
        active_labels = labels_flat[active_mask_flat]

        loss = F.cross_entropy(active_logits, active_labels)

        # Combine accuracy computation with argmax to reduce operations
        with torch.no_grad():
            preds = active_logits.argmax(dim=-1)
            accuracy = (preds == active_labels).float().mean()

        return loss, accuracy

    def _create_parallel_attention_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Creates the [L, 2L] mask for parallel training.
        Optimized version with reduced tensor operations.
        """
        # Use direct indexing instead of multiple tensor operations
        indices = torch.arange(seq_len, device=device)
        block_ids = indices % self.block_size

        # Create masks more efficiently using broadcasting
        # ctx_mask: [L, L] - causal mask for context
        ctx_mask = indices.unsqueeze(0) < indices.unsqueeze(1)

        # noise_mask: [L, L] - same block mask for noise
        noise_mask = block_ids.unsqueeze(0) == block_ids.unsqueeze(1)

        # Combine masks: [L, 2L]
        full_mask_bool = torch.cat([ctx_mask, noise_mask], dim=1)

        # Convert to float mask in one operation
        # 0.0 for keep, -inf for mask
        full_mask = torch.where(full_mask_bool, 0.0, torch.finfo(torch.float32).min)

        return full_mask


def create_dflash_loss_mask(
    seq_len: int, block_size: int, device: torch.device
) -> torch.Tensor:
    """
    Create DFlash-specific loss mask.
    Excludes Block 0 and first position of each block.
    Optimized version with reduced operations.
    """
    positions = torch.arange(seq_len, device=device)

    # Combine conditions in one pass: exclude block 0 and block starts
    # positions < block_size -> block 0
    # positions % block_size == 0 -> block start
    valid_mask = (positions >= block_size) & (positions % block_size != 0)

    return valid_mask.float()
