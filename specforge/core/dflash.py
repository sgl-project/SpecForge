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
    ):
        super().__init__()
        self.draft_model = draft_model
        self.lm_head = target_lm_head
        self.embed_tokens = target_embed_tokens
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.attention_backend = attention_backend

        # Cache for BlockMask
        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None
        self._cached_num_heads: Optional[int] = None

    def prepare_noise_input(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Prepare noise input: first token of each block is real, rest are MASK."""
        seq_len = input_ids.shape[1]
        device = input_ids.device

        positions = torch.arange(seq_len, device=device)
        is_block_start = (positions % self.block_size) == 0

        noise_input_ids = torch.full_like(input_ids, self.mask_token_id)
        noise_input_ids[:, is_block_start] = input_ids[:, is_block_start]

        return noise_input_ids

    def _get_or_create_block_mask(
        self, bsz: int, num_heads: int, q_len: int, kv_len: int, device: torch.device
    ) -> "BlockMask":
        """Get cached BlockMask or create a new one."""
        if (
            self._cached_block_mask is not None
            and self._cached_seq_len == q_len
            and self._cached_bsz == bsz
            and self._cached_num_heads == num_heads
        ):
            return self._cached_block_mask

        block_size = self.block_size

        def dflash_mask_fn(b, h, q_idx, kv_idx):
            L = q_len
            is_ctx = kv_idx < L
            q_block = q_idx // block_size
            k_block_ctx = kv_idx // block_size
            k_block_noise = (kv_idx - L) // block_size
            ctx_visible = is_ctx & (k_block_ctx < q_block)
            noise_visible = (~is_ctx) & (k_block_noise == q_block)
            return ctx_visible | noise_visible

        block_mask = create_block_mask(
            dflash_mask_fn,
            B=bsz,
            H=num_heads,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

        self._cached_block_mask = block_mask
        self._cached_seq_len = q_len
        self._cached_bsz = bsz
        self._cached_num_heads = num_heads

        return block_mask

    def _create_parallel_attention_mask(
        self, seq_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Create [L, 2L] attention mask for parallel training.
        - Left half (ctx): Q can see K_ctx if K's block < Q's block
        - Right half (noise): Q can see K_noise if same block (bidirectional)
        """
        indices = torch.arange(seq_len, device=device)
        block_ids = indices // self.block_size

        q_block_ids = block_ids.unsqueeze(1)
        k_block_ids = block_ids.unsqueeze(0)

        ctx_mask = k_block_ids < q_block_ids
        noise_mask = q_block_ids == k_block_ids

        full_mask_bool = torch.cat([ctx_mask, noise_mask], dim=1)
        full_mask = torch.zeros_like(full_mask_bool, dtype=torch.float32)
        full_mask.masked_fill_(~full_mask_bool, torch.finfo(torch.float32).min)

        return full_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Parallel block-wise training forward pass."""
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # Truncate to multiple of block_size
        n_blocks = seq_len // self.block_size
        effective_len = n_blocks * self.block_size
        input_ids = input_ids[:, :effective_len]
        hidden_states = hidden_states[:, :effective_len, :]
        loss_mask = loss_mask[:, :effective_len]
        attention_mask = attention_mask[:, :effective_len]

        # Prepare inputs
        noise_input_ids = self.prepare_noise_input(input_ids)
        noise_embedding = self.embed_tokens(noise_input_ids)

        # Position IDs: [ctx_pos, noise_pos] both 0..L-1
        pos_seq = torch.arange(effective_len, device=device)
        position_ids = torch.cat([pos_seq, pos_seq], dim=0).unsqueeze(0).expand(bsz, -1)

        # Construct attention mask
        if (
            self.attention_backend == "flex_attention"
            and FLEX_ATTENTION_AVAILABLE
            and create_block_mask is not None
        ):
            num_heads = self.draft_model.config.num_attention_heads
            dflash_attn_mask = self._get_or_create_block_mask(
                bsz=bsz,
                num_heads=num_heads,
                q_len=effective_len,
                kv_len=effective_len * 2,
                device=device,
            )
        else:
            dflash_attn_mask = self._create_parallel_attention_mask(
                effective_len, device
            )
            dflash_attn_mask = dflash_attn_mask.to(dtype=hidden_states.dtype)
            dflash_attn_mask = (
                dflash_attn_mask.unsqueeze(0).unsqueeze(0).expand(bsz, -1, -1, -1)
            )

        # Forward pass
        hidden = self.draft_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )

        # Compute loss (skip block 0 and block starts)
        dflash_loss_mask_base = create_dflash_loss_mask(
            effective_len, self.block_size, device
        )
        combined_mask = loss_mask * dflash_loss_mask_base.unsqueeze(0)

        logits = self.lm_head(hidden)

        logits_flat = logits.reshape(-1, logits.size(-1))
        labels_flat = input_ids.reshape(-1)
        mask_flat = combined_mask.reshape(-1)

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


def create_dflash_loss_mask(
    seq_len: int, block_size: int, device: torch.device
) -> torch.Tensor:
    """Create DFlash loss mask: excludes block 0 and first position of each block."""
    positions = torch.arange(seq_len, device=device)
    block_ids = positions // block_size

    is_block_0 = block_ids == 0
    is_block_start = (positions % block_size) == 0

    valid_mask = ~is_block_0 & ~is_block_start
    return valid_mask.float()
