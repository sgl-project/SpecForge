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
        block_size: int = 16,
        mask_token_id: int = 151666,
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
        seq_len = input_ids.shape[1]
        device = input_ids.device

        positions = torch.arange(seq_len, device=device)
        is_block_start = (positions % self.block_size) == 0

        noise_input_ids = torch.full_like(input_ids, self.mask_token_id)
        noise_input_ids[:, is_block_start] = input_ids[:, is_block_start]

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
        # shape: [1, 2*L] -> [bsz, 2*L]
        position_ids = torch.cat([pos_seq, pos_seq], dim=0).unsqueeze(0).expand(bsz, -1)

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
        combined_mask = loss_mask * dflash_loss_mask_base.unsqueeze(0)

        # hidden[i] predicts input_ids[i] (based on DFlash design where noise[i] is input to predict target[i])
        # However, check design:
        # "hidden[i] predicts token[i] (directly corresponding), not token[i+1]"
        # "noise_input[i] is MASK, we want to predict input_ids[i]"
        # So logits at index i should be compared to labels at index i.

        logits = self.lm_head(hidden)

        # Calculate Loss
        # Flatten
        logits_flat = logits.reshape(-1, logits.size(-1))
        labels_flat = input_ids.reshape(-1)
        mask_flat = combined_mask.reshape(-1)

        # Optimization: only compute CE on valid tokens
        active_indices = mask_flat > 0.5
        active_logits = logits_flat[active_indices]
        active_labels = labels_flat[active_indices]

        loss = F.cross_entropy(active_logits, active_labels)

        with torch.no_grad():
            preds = active_logits.argmax(dim=-1)
            correct = (preds == active_labels).float().sum()
            total = active_labels.numel()
            accuracy = correct / total

        return loss, accuracy

    def _create_parallel_attention_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Creates the [L, 2L] mask for parallel training.
        Rows: Query (Noise) indices 0..L-1
        Cols: Key indices 0..2L-1 (First L are Context, Next L are Noise)

        Logic for Query at index i (belonging to block B = i // block_size):
        1. Can see Context (Cols 0..L-1):
           - Can see context of PREVIOUS blocks.
           - Range: [0, B * block_size)
        2. Can see Noise (Cols L..2L-1):
           - Can see noise of CURRENT block up to self.
           - Range: [L + B * block_size, L + i]
           - (Inclusive of i because causal mask usually allows seeing self)
        """
        # Block indices for each position [0, 0, ..., 1, 1, ...]
        indices = torch.arange(seq_len, device=device)
        block_ids = indices // self.block_size

        # 1. Context Mask (L x L) - Left half of K
        # Q[i] can see K_ctx[j] if Block(Q[i]) > Block(K_ctx[j])
        # Actually, Block(Q[i]) can see Context of all previous blocks.
        # It implies Block(K_ctx[j]) < Block(Q[i])
        # Wait, strictly: Block B needs context from 0..(B*16).
        # So it sees all K_ctx where index < B * 16.
        # Which is equivalent to block_ids[j] < block_ids[i].

        # Broadcast logic
        # shape [L, 1]
        q_block_ids = block_ids.unsqueeze(1)
        # shape [1, L]
        k_block_ids = block_ids.unsqueeze(0)

        # Mask: 1 if K's block is strictly less than Q's block
        # This gives access to all PREVIOUS blocks' context.
        ctx_mask = k_block_ids < q_block_ids

        # 2. Noise Mask (L x L) - Right half of K
        # Standard Causal Mask WITHIN the same block.
        # Q[i] can see K_noise[j] if:
        #   a) Same Block: Block(Q[i]) == Block(K_noise[j])
        #   b) Causal: j <= i
        # Different blocks cannot see each other's noise.

        same_block = q_block_ids == k_block_ids

        noise_mask = same_block

        # Combine [Ctx_Mask, Noise_Mask]
        # Shape [L, 2L]
        # We need float mask for attention: 0.0 for allow, -inf for mask
        # Transformers usually handles boolean masks by converting them,
        # but explicit MinValue is safer if passing to generic attention.
        # However, most HF models accept boolean [batch, 1, Q, K].

        full_mask_bool = torch.cat([ctx_mask, noise_mask], dim=1)

        # Check standard HF format: usually 1 for keep, 0 for mask, or boolean.
        # Qwen3 implementation likely uses typical attention_mask handling.
        # Let's return boolean for now, the model wrapper usually handles conversion
        # or we check Qwen3DFlashAttention source usage.
        # Looking at Qwen3DFlashAttention: `attn_output = attn_fn(..., attention_mask, ...)`
        # If using SDPA, it expects boolean or float mask.
        # If we look at `modeling_qwen3.py` (standard), it usually employs `_prepare_4d_causal_attention_mask`.
        # But here we pass it explicitly.
        # To be safe with `eager_attention_forward` and `SDPA`, we typically want:
        # 0.0 for unmasked, -inf for masked.

        dtype = (
            torch.bfloat16
        )  # or get from device/model, but we return a tensor builder
        # We will cast later or return boolean and let logic handle it?
        # Safe bet: Return typical extended attention mask format: 0.0 for keep, min_dtype for remove.

        # But wait, Qwen3DFlashAttention passes attention_mask directly to attn_fn.
        # If attn_fn is SDPA, it handles boolean.
        # Let's return a float mask: 0.0 for True, -inf for False.

        full_mask = torch.zeros_like(full_mask_bool, dtype=torch.float32)
        full_mask.masked_fill_(~full_mask_bool, torch.finfo(torch.float32).min)

        return full_mask


def create_dflash_loss_mask(
    seq_len: int, block_size: int, device: torch.device
) -> torch.Tensor:
    """
    Create DFlash-specific loss mask.
    Excludes Block 0 and first position of each block.
    """
    positions = torch.arange(seq_len, device=device)
    block_ids = positions // block_size

    is_block_0 = block_ids == 0
    is_block_start = (positions % block_size) == 0

    valid_mask = ~is_block_0 & ~is_block_start
    return valid_mask.float()
