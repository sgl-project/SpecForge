# coding=utf-8
"""DFlash Training Wrapper."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.modeling.draft.dflash import DFlashDraftModel

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask

    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    BlockMask = None
    create_block_mask = None


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,  # [b, n_blocks]
    block_keep_mask: torch.Tensor,  # [b, n_blocks]
    S: int,  # Context length (KV sequence length before draft blocks)
    block_size: int,  # Draft block size
    device: torch.device,
):
    """
    Construct Flex Attention Block Mask for DFlash.

    KV Structure: [Context (S tokens) | Block_0 | Block_1 | ... | Block_{n-1}]
    Q Structure:  [Block_0 | Block_1 | ... | Block_{n-1}]

    Attention Rules:
    1. Each draft block can see all context up to its anchor position (inclusive).
    2. Intra-block attention is bidirectional (non-causal).
    3. Different draft blocks are invisible to each other.
    4. Invalid blocks (block_keep_mask=False) cannot see anything.

    Args:
        anchor_positions: [b, n_blocks] - anchor position of each block in the original sequence
        block_keep_mask: [b, n_blocks] - mask indicating which blocks are valid
        S: Context length (length of target model cache)
        block_size: length of each draft block
        device: torch device

    Returns:
        BlockMask object for flex_attention
    """

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        """
        Mask function for flex attention.

        Args:
            b: batch index
            h: head index (unused, broadcast across heads)
            q_idx: query position in [0, n_blocks * block_size)
            kv_idx: key/value position in [0, S + n_blocks * block_size)
        """
        # ====== Step 1: Determine which block the query belongs to ======
        q_block_id = q_idx // block_size

        # ====== Step 2: Get anchor position for the current block ======
        anchor_pos = anchor_positions[b, q_block_id]

        # ====== Step 3: Determine if KV is in the context region ======
        is_context = kv_idx < S

        # Context visibility: can only see context up to the anchor position
        # Note: anchor_pos is the position in the original sequence, corresponding to the index in the context
        mask_context = is_context & (kv_idx <= anchor_pos)

        # ====== Step 4: Determine if KV is in the draft blocks region ======
        is_draft = kv_idx >= S

        # Calculate which draft block the KV belongs to
        # Note: This calculation produces garbage values when is_draft=False, but they will be masked out
        kv_block_id = (kv_idx - S) // block_size

        # Intra-block bidirectional attention: visible within the same block
        mask_draft = is_draft & (q_block_id == kv_block_id)

        # ====== Step 5: Apply block validity mask ======
        # Only valid blocks can see content
        is_valid_block = block_keep_mask[b, q_block_id]

        # ====== Final Mask ======
        # (See context OR see own block) AND block is valid
        return (mask_context | mask_draft) & is_valid_block

    # Create block mask
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    block_mask = create_block_mask(
        dflash_mask_mod,
        B=B,
        H=None,  # Broadcast across all heads
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
    )

    return block_mask


class OnlineDFlashModel(nn.Module):
    """DFlash online training wrapper with block-wise CE loss."""

    def __init__(
        self,
        draft_model: DFlashDraftModel,
        target_lm_head: nn.Module,
        target_embed_tokens: nn.Module,
        mask_token_id: int,
        block_size: int = 16,
        attention_backend: str = "flex_attention",
        num_anchors: int = 512,
        loss_decay_gamma: Optional[float] = None,
    ):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()) - 1)

        if max_n <= 0:
            raise ValueError("should preprocess the data.")

        indices = (
            torch.arange(max_anchor + 1, device=device).unsqueeze(0).expand(bsz, -1)
        )
        masked_indices = torch.where(
            valid, indices, torch.tensor(seq_len + 1, device=device)
        )

        random_vals = torch.rand(bsz, max_anchor + 1, device=device)
        random_vals = torch.where(valid, random_vals, torch.tensor(2.0, device=device))

        _, sorted_idx = random_vals.sort(dim=1)
        gathered = torch.gather(masked_indices, 1, sorted_idx)
        anchors = gathered[:, :max_n].sort(dim=1).values

        keep_mask = torch.arange(max_n, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1).clamp(max=max_n)
        anchors = torch.where(
            keep_mask, anchors, torch.tensor(0, dtype=torch.long, device=device)
        )

        return anchors, keep_mask

    def _build_target_context_feature_from_anchors(
        self,
        hidden_states: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Gather fixed-size blocks; padding blocks get block_id=-1 and loss=0."""
        n = anchor_positions.shape[1]

        # hidden states b, s, d (s: prompt + response) 1, 787, 12800 -> 1, 787*n, 12800
        hidden_repeated = hidden_states.unsqueeze(1).repeat(1, n, 1, 1)

        return hidden_repeated

    def prepare_noise_input(
        self, input_ids: torch.Tensor, block_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Prepare noise input: first token of each block is real, rest are MASK."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        if block_ids is not None:
            is_block_start = torch.ones(bsz, seq_len, dtype=torch.bool, device=device)
            is_block_start[:, 1:] = block_ids[:, 1:] != block_ids[:, :-1]
        else:
            positions = torch.arange(seq_len, device=device)
            is_block_start = (positions % self.block_size) == 0
            is_block_start = is_block_start.unsqueeze(0).expand(bsz, -1)

        noise_input_ids = torch.full_like(input_ids, self.mask_token_id)
        noise_input_ids[is_block_start] = input_ids[is_block_start]
        return noise_input_ids

    def _create_position_ids(self, anchor_positions: torch.Tensor) -> torch.Tensor:
        """
        Create Position IDs for parallel draft blocks.

        Args:
            anchor_positions: [bsz, n_blocks] starting position of each block

        Returns:
            position_ids: [bsz, n_blocks * block_size] flattened absolute position indices
        """
        bsz, n_blocks = anchor_positions.shape
        block_size = self.block_size
        device = anchor_positions.device

        # 1. Create intra-block offsets: [0, 1, ..., block_size-1]
        # Shape: (1, 1, block_size)
        offsets = torch.arange(block_size, device=device).view(1, 1, -1)

        # 2. Expand anchor positions for broadcasting
        # Shape: (bsz, n_blocks, 1)
        anchors = anchor_positions.unsqueeze(-1)

        # 3. Calculate absolute position: anchor + offset
        # Shape: (bsz, n_blocks, block_size)
        pos_ids = anchors + offsets

        # 4. Flatten to match the dimensions of hidden_states (bsz, n * bs)
        # Shape: (bsz, n_blocks * block_size)
        position_ids = pos_ids.view(bsz, -1)

        return position_ids

    def _create_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
        # Get dimensions
        bsz, seq_len = input_ids.shape
        n = anchor_positions.shape[1]
        bs = self.block_size
        device = input_ids.device

        # Initialize noise_ids with mask tokens
        noise_ids = torch.full(
            (bsz, n * bs), self.mask_token_id, dtype=torch.long, device=device
        )

        # Vectorized gathering of anchor tokens
        # Create block start indices: (bsz, n) where each position is j*block_size
        block_starts = torch.arange(n, device=device) * bs  # (n,)
        block_starts = block_starts.unsqueeze(0).expand(bsz, -1)  # (bsz, n)

        # Gather anchor tokens from input_ids at anchor_positions
        # Clamp anchor_positions to valid range
        valid_anchor_positions = anchor_positions.clamp(0, seq_len - 1)

        # Gather the tokens at anchor positions
        anchor_tokens = torch.gather(input_ids, 1, valid_anchor_positions)  # (bsz, n)

        # Create flat indices for where to place anchor tokens in noise_ids
        flat_batch_idx = (
            torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, n)
        )  # (bsz, n)
        flat_seq_idx = block_starts  # (bsz, n)

        # Apply block_keep_mask and scatter anchor tokens
        # Only set anchor tokens where block_keep_mask is True
        noise_ids[flat_batch_idx, flat_seq_idx] = torch.where(
            block_keep_mask,
            anchor_tokens,
            torch.tensor(self.mask_token_id, dtype=torch.long, device=device),
        )

        # Get embeddings
        return self.embed_tokens(noise_ids)  # (bsz, n * bs, embed_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel block-wise training forward pass."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Sample Anchor positions
        # anchor_positions: [bsz, n_blocks]
        # block_keep_mask: [bsz, n_blocks]
        anchor_positions, block_keep_mask = self._sample_anchor_positions(
            seq_len, loss_mask, device
        )
        # anchor_positions = torch.tensor([[13]],device=device,dtype=torch.long)
        # block_keep_mask = torch.tensor([[True]],device=device,dtype=torch.bool)

        # 2. Prepare input Embedding (Noise) and Position IDs
        # noise_embedding: [bsz, n_blocks * block_size, hidden_dim]
        noise_embedding = self._create_noise_embed(
            input_ids, anchor_positions, block_keep_mask
        )

        # A. Generate Context IDs: [0, 1, 2, ..., seq_len-1]
        # Shape: (1, seq_len) -> (bsz, seq_len)
        context_position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        )

        # B. Generate Draft IDs: [anchor, anchor+1, ...]
        draft_position_ids = self._create_position_ids(anchor_positions)

        # C. Concatenate: [Context IDs | Draft IDs]
        # Shape: (bsz, seq_len + n_blocks * block_size)
        # Corresponds to the 787 + 8192 = 8979 mentioned in the previous error message
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        # 3. Create Attention Mask
        # S (Context Length) is set to seq_len here because the draft model cross-attends to the entire target hidden states
        dflash_attn_mask = create_dflash_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            S=seq_len,
            block_size=self.block_size,
            device=device,
        )

        # 4. Draft Model Forward
        # output_hidden: [bsz, n_blocks * block_size, hidden_dim]
        output_hidden = self.draft_model(
            position_ids=full_position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )

        # 5. Compute Logits
        # logits: [bsz, n_blocks * block_size, vocab_size]
        print(output_hidden)
        logits = self.lm_head(output_hidden)

        # =================================================================
        # Loss & Accuracy Calculation Logic
        # =================================================================

        # 6. Construct Labels (Ground Truth)
        # We need to predict block_size tokens after the anchor
        # Label offset: 1, 2, ..., block_size
        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(
            1, 1, -1
        )  # [1, 1, bs]

        # Calculate absolute coordinates of labels in input_ids
        # label_indices: [bsz, n_blocks, block_size]
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets

        # Boundary check: ensure indices do not exceed seq_len - 1
        valid_label_mask = label_indices < seq_len
        safe_label_indices = label_indices.clamp(max=seq_len - 1)

        # Gather labels: [bsz, n_blocks, block_size]
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )

        # 7. Construct Comprehensive Loss Weight Mask
        # A. Block validity (block_keep_mask)
        # B. Indices within bounds (valid_label_mask)
        # C. Original loss_mask (corresponding positions are not padding)

        # [bsz, n_blocks, 1] -> [bsz, n_blocks, block_size]
        weight_mask = (
            block_keep_mask.unsqueeze(-1).expand(-1, -1, self.block_size).float()
        )
        weight_mask = weight_mask * valid_label_mask.float()

        # Gather loss mask of original data
        original_loss_mask_gathered = torch.gather(
            loss_mask.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        weight_mask = weight_mask * original_loss_mask_gathered


        binary_eval_mask = weight_mask.view(-1)

        # 8. Apply Loss Decay (if configured)
        # Here k corresponds to 0 to block_size-1 (i.e., the 1st to Nth predicted tokens)
        # Earlier tokens have higher weights
        if self.loss_decay_gamma is not None and self.loss_decay_gamma > 0:
            k = torch.arange(self.block_size, device=device).view(
                1, 1, -1
            )  # [0, 1, ..., bs-1]
            decay_weights = torch.exp(
                -k / self.loss_decay_gamma
            )  # exp(-(k)/gamma) vs paper k-1 logic
            weight_mask = weight_mask * decay_weights
        # 9. Compute Cross Entropy
        # Flatten for loss computation
        # logits: [N_total, vocab], targets: [N_total]
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = target_ids.view(-1)
        flat_weights = weight_mask.view(-1)

        # Reduction='none' to apply custom weights
        loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        # Weighted average
        valid_token_count = flat_weights.sum() + 1e-6
        loss = (loss_per_token * flat_weights).sum() / valid_token_count

        # 10. Compute Accuracy (only count where mask > 0.5)
        with torch.no_grad():
            pred_ids = torch.argmax(flat_logits, dim=-1)
            correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
            
            actual_token_count = binary_eval_mask.sum() + 1e-6
            accuracy = correct.sum().float() / actual_token_count

        return loss, accuracy
