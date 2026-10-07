"""DFlash-family attention masks shared by training and inference."""

from typing import Optional

import torch

try:
    from torch.nn.attention.flex_attention import create_block_mask
except ImportError:
    create_block_mask = None


def resolve_dflash_is_causal(is_causal: Optional[bool], layer_type: str) -> bool:
    """Resolve in-block causality the way SGLang's DFlash model does.

    An explicit ``is_causal`` applies to every layer. Unset is a per-layer-type
    default: full layers bidirectional, sliding layers causal.
    """

    if is_causal is not None:
        return bool(is_causal)
    return layer_type == "sliding_attention"


def create_dflash_sdpa_mask(
    anchor_positions,
    block_keep_mask,
    S,
    block_size,
    device,
    sliding_window: Optional[int] = None,
    is_causal: Optional[bool] = None,
):
    """Construct a full or sliding dense boolean DFlash mask."""

    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be > 0")
    is_causal = resolve_dflash_is_causal(
        is_causal,
        "sliding_attention" if sliding_window is not None else "full_attention",
    )
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    q_indices = torch.arange(Q_LEN, device=device).view(1, 1, -1, 1)  # (1, 1, Q_LEN, 1)
    kv_indices = torch.arange(KV_LEN, device=device).view(
        1, 1, 1, -1
    )  # (1, 1, 1, KV_LEN)

    q_block_ids = q_indices // block_size
    q_block_offsets = q_indices % block_size

    anchor_expanded = anchor_positions.view(B, 1, N, 1).repeat_interleave(
        block_size, dim=2
    )

    mask_context = (kv_indices < S) & (kv_indices < anchor_expanded)
    if sliding_window is not None:
        # The current draft token occupies one slot in the window.
        context_lower_bound = anchor_expanded + q_block_offsets - (sliding_window - 1)
        mask_context = mask_context & (kv_indices >= context_lower_bound)

    is_draft = kv_indices >= S
    kv_block_ids = (kv_indices - S) // block_size
    mask_draft = is_draft & (q_block_ids == kv_block_ids)
    kv_block_offsets = (kv_indices - S) % block_size
    if is_causal:
        mask_draft = mask_draft & (kv_block_offsets <= q_block_offsets)
    if sliding_window is not None:
        # Left window bound inside the block. Only bites when sliding_window <
        # block_size; no right bound, since SGLang backends disagree on one for
        # non-causal windows (FA3 symmetric, FlashInfer/Triton left-only).
        mask_draft = mask_draft & (
            kv_block_offsets >= q_block_offsets - (sliding_window - 1)
        )

    valid_block = block_keep_mask.view(B, 1, N, 1).repeat_interleave(block_size, dim=2)

    final_mask = (mask_context | mask_draft) & valid_block
    return final_mask


def create_dflash_block_mask(
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    S: int,
    block_size: int,
    device: torch.device,
    flex_block_size=None,
    sliding_window: Optional[int] = None,
    is_causal: Optional[bool] = None,
):
    """Construct a full or sliding Flex Attention mask for DFlash."""

    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must be > 0")
    is_causal = resolve_dflash_is_causal(
        is_causal,
        "sliding_attention" if sliding_window is not None else "full_attention",
    )

    def dflash_mask_mod(b, h, q_idx, kv_idx):
        q_block_id = q_idx // block_size
        q_block_offset = q_idx % block_size
        safe_q_block_id = q_block_id.clamp(max=N - 1)
        anchor_pos = anchor_positions[b, safe_q_block_id]

        is_context = kv_idx < S
        # Strictly less than: matches inference where target_hidden[anchor_pos]
        # is not available as context.
        mask_context = is_context & (kv_idx < anchor_pos)
        if sliding_window is not None:
            # The current draft token occupies one slot in the window.
            context_lower_bound = anchor_pos + q_block_offset - (sliding_window - 1)
            mask_context = mask_context & (kv_idx >= context_lower_bound)

        is_draft = kv_idx >= S
        kv_block_id = (kv_idx - S) // block_size
        mask_draft = is_draft & (q_block_id == kv_block_id)
        kv_block_offset = (kv_idx - S) % block_size
        if is_causal:
            mask_draft = mask_draft & (kv_block_offset <= q_block_offset)
        if sliding_window is not None:
            mask_draft = mask_draft & (
                kv_block_offset >= q_block_offset - (sliding_window - 1)
            )

        is_valid_block = block_keep_mask[b, safe_q_block_id]
        in_bounds = q_block_id < N
        return (mask_context | mask_draft) & is_valid_block & in_bounds

    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    kwargs = {}
    if flex_block_size is not None:
        kwargs["BLOCK_SIZE"] = flex_block_size
    return create_block_mask(
        dflash_mask_mod,
        B=B,
        H=None,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
        **kwargs,
    )
