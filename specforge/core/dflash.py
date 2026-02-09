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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Randomly sample anchor positions per sample; returns (anchors, keep_mask)."""
        bs = self.block_size
        bsz = loss_mask.shape[0]
        max_anchor = max(seq_len - bs, 0)

        valid = loss_mask[:, : max_anchor + 1] > 0.5
        valid_counts = valid.sum(dim=1)
        max_n = min(self.num_anchors, int(valid_counts.max().item()))

        if max_n == 0:
            anchors = torch.arange(0, seq_len, bs, device=device)
            anchors = anchors.unsqueeze(0).expand(bsz, -1)
            return anchors, torch.ones(
                bsz, anchors.shape[1], dtype=torch.bool, device=device
            )

        anchor_list = []
        keep_list = []
        for i in range(bsz):
            valid_indices = valid[i].nonzero(as_tuple=False).squeeze(-1)
            n_i = min(self.num_anchors, valid_indices.numel())
            if n_i == 0:
                anchors_i = torch.zeros(max_n, dtype=torch.long, device=device)
                keep_i = torch.zeros(max_n, dtype=torch.bool, device=device)
            else:
                perm = torch.randperm(valid_indices.numel(), device=device)[:n_i]
                anchors_i = valid_indices[perm].sort().values
                if n_i < max_n:
                    anchors_i = torch.cat(
                        [anchors_i, anchors_i[-1:].expand(max_n - n_i)], dim=0
                    )
                keep_i = torch.zeros(max_n, dtype=torch.bool, device=device)
                keep_i[:n_i] = True
            anchor_list.append(anchors_i)
            keep_list.append(keep_i)
        return torch.stack(anchor_list, dim=0), torch.stack(keep_list, dim=0)

    def _build_blocks_from_anchors(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather fixed-size blocks; padding blocks get block_id=-1 and loss=0."""
        bs = self.block_size
        device = input_ids.device
        bsz = input_ids.shape[0]
        n = anchor_positions.shape[1]

        offsets = torch.arange(bs, device=device).unsqueeze(0)
        gather_idx = anchor_positions.unsqueeze(-1) + offsets
        gather_idx = gather_idx.reshape(bsz, -1)

        block_input_ids = torch.gather(input_ids, 1, gather_idx)
        block_hidden = torch.gather(
            hidden_states,
            1,
            gather_idx.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)),
        )
        block_loss_mask = torch.gather(loss_mask, 1, gather_idx)

        token_keep = block_keep_mask.repeat_interleave(bs, dim=1)
        block_loss_mask = block_loss_mask * token_keep.to(block_loss_mask.dtype)

        block_ids = torch.arange(n, device=device).repeat_interleave(bs)
        pad_token_mask = (~block_keep_mask).repeat_interleave(bs, dim=1)
        block_ids = block_ids.unsqueeze(0).expand(bsz, -1).clone()
        block_ids[pad_token_mask] = -1

        return block_input_ids, block_hidden, block_loss_mask, block_ids, gather_idx

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
                q_b = _block_ids[b, q_idx]
                k_ctx = _block_ids[b, kv_idx.clamp(max=L - 1)]
                k_noise = _block_ids[b, (kv_idx - L).clamp(min=0, max=L - 1)]
                q_valid = q_b >= 0
                k_ctx_valid = k_ctx >= 0
                k_noise_valid = k_noise >= 0
                ctx_visible = is_ctx & q_valid & k_ctx_valid & (k_ctx < q_b)
                noise_visible = (~is_ctx) & q_valid & k_noise_valid & (k_noise == q_b)
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
        bsz: int,
        seq_len: int,
        device: torch.device,
        block_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Create [bsz, L, 2L] attention mask for parallel training."""
        if block_ids is None:
            ids = torch.arange(seq_len, device=device) // self.block_size
            q_ids = ids.unsqueeze(1)
            k_ids = ids.unsqueeze(0)
            ctx_mask = k_ids < q_ids
            noise_mask = q_ids == k_ids
            full_mask_bool = torch.cat([ctx_mask, noise_mask], dim=1)
            full_mask = torch.zeros_like(full_mask_bool, dtype=torch.float32)
            full_mask.masked_fill_(~full_mask_bool, torch.finfo(torch.float32).min)
            return full_mask.unsqueeze(0).expand(bsz, -1, -1)

        q_ids = block_ids.unsqueeze(2)
        k_ids = block_ids.unsqueeze(1)
        q_valid = q_ids >= 0
        k_valid = k_ids >= 0
        ctx_mask = q_valid & k_valid & (k_ids < q_ids)
        noise_mask = q_valid & k_valid & (k_ids == q_ids)
        full_mask_bool = torch.cat([ctx_mask, noise_mask], dim=2)
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
            anchor_positions, block_keep_mask = self._sample_anchor_positions(
                seq_len, loss_mask, device
            )
            (input_ids, hidden_states, loss_mask, block_ids, block_positions) = (
                self._build_blocks_from_anchors(
                    input_ids,
                    hidden_states,
                    loss_mask,
                    anchor_positions,
                    block_keep_mask,
                )
            )
            effective_len = input_ids.shape[1]
            base_positions = block_positions
        else:
            n_blocks = seq_len // self.block_size
            effective_len = n_blocks * self.block_size
            input_ids = input_ids[:, :effective_len]
            hidden_states = hidden_states[:, :effective_len, :]
            loss_mask = loss_mask[:, :effective_len]
            attention_mask = attention_mask[:, :effective_len]
            base_positions = (
                torch.arange(effective_len, device=device).unsqueeze(0).expand(bsz, -1)
            )

        noise_input_ids = self.prepare_noise_input(input_ids, block_ids)
        noise_embedding = self.embed_tokens(noise_input_ids)

        position_ids = torch.cat([base_positions, base_positions], dim=1)

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
                bsz, effective_len, device, block_ids
            )
            dflash_attn_mask = dflash_attn_mask.to(dtype=hidden_states.dtype)
            dflash_attn_mask = dflash_attn_mask.unsqueeze(1)

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
        if block_ids is None:
            dflash_loss_weights = dflash_loss_weights.unsqueeze(0)
        combined_mask = loss_mask * dflash_loss_weights

        logits = self.lm_head(hidden)

        # Compute inference-aligned acceptance rate (completion tokens only)
        with torch.no_grad():
            preds_all = logits.argmax(dim=-1)
            correct_all = (preds_all == input_ids).float()

            bs = self.block_size
            n_blocks = effective_len // bs

            try:
                if block_ids is not None:
                    correct_blocks = correct_all.reshape(bsz, n_blocks, bs)
                    loss_mask_blocks = loss_mask.reshape(bsz, n_blocks, bs)
                else:
                    if n_blocks > 1:
                        correct_blocks = correct_all[:, bs:].reshape(
                            bsz, n_blocks - 1, bs
                        )
                        loss_mask_blocks = loss_mask[:, bs:].reshape(
                            bsz, n_blocks - 1, bs
                        )
                    else:
                        raise ValueError("Only one block")

                correct_pred = correct_blocks[:, :, 1:]
                loss_mask_pred = loss_mask_blocks[:, :, 1:]

                block_valid = (loss_mask_pred.sum(dim=2) == (bs - 1)).float()
                correct_pred = correct_pred * loss_mask_pred
                cumulative_correct = correct_pred.cumprod(dim=2)

                acceptance_lengths = cumulative_correct.sum(dim=2)
                acceptance_lengths = (acceptance_lengths * block_valid).sum(dim=1)
                total_blocks_sum = block_valid.sum(dim=1).sum().clamp_min(1)
                avg_accept_length = acceptance_lengths.sum() / total_blocks_sum
                accuracy = avg_accept_length / (bs - 1)
            except Exception:
                valid_mask = (loss_mask > 0.5).reshape(-1)
                correct_flat = correct_all.reshape(-1)[valid_mask]
                accuracy = correct_flat.mean() if correct_flat.numel() > 0 else 0.0

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

        return loss, accuracy


def create_dflash_loss_mask(
    seq_len: int,
    block_size: int,
    device: torch.device,
    gamma: Optional[float] = None,
    block_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Create DFlash loss mask: excludes block starts; for non-random, also excludes block 0.

    Returns [seq_len] when block_ids is None, [bsz, seq_len] when block_ids is per-sample.
    """
    positions = torch.arange(seq_len, device=device)
    pos_in_block = positions % block_size

    if block_ids is not None:
        is_block_start = torch.ones_like(block_ids, dtype=torch.bool)
        is_block_start[:, 1:] = block_ids[:, 1:] != block_ids[:, :-1]
        valid_mask = ~is_block_start & (block_ids >= 0)
        pos_in_block = pos_in_block.unsqueeze(0)
    else:
        is_block_start = (positions % block_size) == 0
        is_first_block = (positions // block_size) == 0
        valid_mask = ~is_first_block & ~is_block_start

    if gamma is not None:
        decay = torch.exp(-(pos_in_block.float() - 1.0) / gamma)
        return valid_mask.float() * decay
    else:
        return valid_mask.float()
