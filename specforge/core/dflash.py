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
        random_anchor: bool = False,
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
        self.random_anchor = random_anchor
        self.num_anchors = num_anchors
        self.loss_decay_gamma = loss_decay_gamma

        self._cached_block_mask: Optional[BlockMask] = None
        self._cached_seq_len: Optional[int] = None
        self._cached_bsz: Optional[int] = None

    def _sample_anchor_positions(
        self, seq_len: int, loss_mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Randomly sample anchor positions from the response region."""
        bs = self.block_size
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[0, : max_anchor + 1] > 0.5
        valid_indices = valid.nonzero(as_tuple=False).squeeze(-1)

        n = min(self.num_anchors, valid_indices.numel())
        if n == 0:
            return torch.arange(0, seq_len, bs, device=device)

        perm = torch.randperm(valid_indices.numel(), device=device)[:n]
        return valid_indices[perm].sort().values

    def _build_blocks_from_anchors(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather fixed-size blocks starting at each anchor position."""
        bs = self.block_size
        device = input_ids.device
        n = anchor_positions.shape[0]

        offsets = torch.arange(bs, device=device).unsqueeze(0)
        gather_idx = (anchor_positions.unsqueeze(1) + offsets).reshape(-1)

        block_input_ids = input_ids[:, gather_idx]
        block_hidden = hidden_states[:, gather_idx, :]
        block_loss_mask = loss_mask[:, gather_idx]
        block_ids = torch.arange(n, device=device).repeat_interleave(bs)

        return block_input_ids, block_hidden, block_loss_mask, block_ids

    def prepare_noise_input(
        self, input_ids: torch.Tensor, block_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Prepare noise input: first token of each block is real, rest are MASK."""
        seq_len = input_ids.shape[1]
        device = input_ids.device

        if block_ids is not None:
            is_block_start = torch.ones(seq_len, dtype=torch.bool, device=device)
            is_block_start[1:] = block_ids[1:] != block_ids[:-1]
        else:
            positions = torch.arange(seq_len, device=device)
            is_block_start = (positions % self.block_size) == 0

        noise_input_ids = torch.full_like(input_ids, self.mask_token_id)
        noise_input_ids[:, is_block_start] = input_ids[:, is_block_start]
        return noise_input_ids

    def _get_or_create_block_mask(
        self,
        bsz: int,
        q_len: int,
        kv_len: int,
        device: torch.device,
        block_ids: Optional[torch.Tensor] = None,
    ) -> "BlockMask":
        """Get cached BlockMask or create a new one."""
        if block_ids is None:
            if (
                self._cached_block_mask is not None
                and self._cached_seq_len == q_len
                and self._cached_bsz == bsz
            ):
                return self._cached_block_mask

        block_size = self.block_size

        if block_ids is not None:
            _block_ids = block_ids

            def dflash_mask_fn(b, h, q_idx, kv_idx):
                L = q_len
                is_ctx = kv_idx < L
                q_block = _block_ids[q_idx]
                k_block_ctx = _block_ids[kv_idx.clamp(max=L - 1)]
                k_block_noise = _block_ids[(kv_idx - L).clamp(min=0, max=L - 1)]
                ctx_visible = is_ctx & (k_block_ctx < q_block)
                noise_visible = (~is_ctx) & (k_block_noise == q_block)
                return ctx_visible | noise_visible

        else:

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
            H=1,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
        )

        if block_ids is None:
            self._cached_block_mask = block_mask
            self._cached_seq_len = q_len
            self._cached_bsz = bsz

        return block_mask

    def _create_parallel_attention_mask(
        self,
        seq_len: int,
        device: torch.device,
        block_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Create [L, 2L] attention mask for parallel training."""
        if block_ids is None:
            block_ids = torch.arange(seq_len, device=device) // self.block_size

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
        block_ids = None

        if self.random_anchor and self.training:
            anchor_positions = self._sample_anchor_positions(seq_len, loss_mask, device)
            (input_ids, hidden_states, loss_mask, block_ids) = (
                self._build_blocks_from_anchors(
                    input_ids, hidden_states, loss_mask, anchor_positions
                )
            )
            effective_len = input_ids.shape[1]
        else:
            n_blocks = seq_len // self.block_size
            effective_len = n_blocks * self.block_size
            input_ids = input_ids[:, :effective_len]
            hidden_states = hidden_states[:, :effective_len, :]
            loss_mask = loss_mask[:, :effective_len]
            attention_mask = attention_mask[:, :effective_len]

        noise_input_ids = self.prepare_noise_input(input_ids, block_ids)
        noise_embedding = self.embed_tokens(noise_input_ids)

        pos_seq = torch.arange(effective_len, device=device)
        position_ids = torch.cat([pos_seq, pos_seq], dim=0).unsqueeze(0).expand(bsz, -1)

        if (
            self.attention_backend == "flex_attention"
            and FLEX_ATTENTION_AVAILABLE
            and create_block_mask is not None
        ):
            dflash_attn_mask = self._get_or_create_block_mask(
                bsz=bsz,
                q_len=effective_len,
                kv_len=effective_len * 2,
                device=device,
                block_ids=block_ids,
            )
        else:
            dflash_attn_mask = self._create_parallel_attention_mask(
                effective_len, device, block_ids
            )
            dflash_attn_mask = dflash_attn_mask.to(dtype=hidden_states.dtype)
            dflash_attn_mask = (
                dflash_attn_mask.unsqueeze(0).unsqueeze(0).expand(bsz, -1, -1, -1)
            )

        hidden = self.draft_model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=hidden_states,
            attention_mask=dflash_attn_mask,
        )

        dflash_loss_weights = create_dflash_loss_mask(
            effective_len,
            self.block_size,
            device,
            gamma=self.loss_decay_gamma,
            block_ids=block_ids,
        )
        combined_mask = loss_mask * dflash_loss_weights.unsqueeze(0)

        logits = self.lm_head(hidden)
        logits_flat = logits.reshape(-1, logits.size(-1))
        labels_flat = input_ids.reshape(-1)
        mask_flat = combined_mask.reshape(-1)

        active_indices = mask_flat > 1e-6
        active_logits = logits_flat[active_indices]
        active_labels = labels_flat[active_indices]
        active_weights = mask_flat[active_indices]

        if self.loss_decay_gamma is not None:
            per_token_loss = F.cross_entropy(
                active_logits, active_labels, reduction="none"
            )
            loss = (per_token_loss * active_weights).sum() / active_weights.sum()
        else:
            loss = F.cross_entropy(active_logits, active_labels)

        with torch.no_grad():
            preds = active_logits.argmax(dim=-1)
            correct = (preds == active_labels).float().sum()
            total = active_labels.numel()
            accuracy = correct / total

        return loss, accuracy


def create_dflash_loss_mask(
    seq_len: int,
    block_size: int,
    device: torch.device,
    gamma: Optional[float] = None,
    block_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Create DFlash loss mask: excludes block 0 and block starts, with optional decay."""
    positions = torch.arange(seq_len, device=device)

    if block_ids is not None:
        is_block_start = torch.ones(seq_len, dtype=torch.bool, device=device)
        is_block_start[1:] = block_ids[1:] != block_ids[:-1]
        is_first_block = block_ids == block_ids[0]
        pos_in_block = positions % block_size
    else:
        is_block_start = (positions % block_size) == 0
        is_first_block = (positions // block_size) == 0
        pos_in_block = positions % block_size

    valid_mask = ~is_first_block & ~is_block_start

    if gamma is not None:
        decay = torch.exp(-(pos_in_block.float() - 1.0) / gamma)
        return valid_mask.float() * decay
    else:
        return valid_mask.float()
