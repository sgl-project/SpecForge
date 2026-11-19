"""Pure PyTorch COD sampling and visibility, shared by P-EAGLE training paths."""

from typing import Optional, Tuple

import torch


def generate_cod_sample_indices(
    seq_length: int,
    loss_mask: torch.Tensor,
    num_depths: int = 8,
    down_sample_ratio: float = 0.8,
    down_sample_ratio_min: float = 0.2,
    filter_position_zero: bool = True,
    lengths: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if loss_mask.dim() == 2:
        if loss_mask.shape[0] != 1:
            raise ValueError("COD sampling currently requires batch size 1")
        loss_mask = loss_mask[0]
    elif loss_mask.dim() != 1:
        raise ValueError("loss_mask must have shape [seq] or [1, seq]")
    if loss_mask.numel() != seq_length:
        raise ValueError(
            f"loss_mask has {loss_mask.numel()} positions, expected {seq_length}"
        )

    device = loss_mask.device
    if lengths is None:
        lengths = torch.tensor([seq_length], dtype=torch.long, device=device)
    else:
        lengths = lengths.to(device=device, dtype=torch.long).flatten()
    if lengths.numel() == 0 or torch.any(lengths <= 0):
        raise ValueError("lengths must contain positive document lengths")
    packed_length = int(lengths.sum().item())
    if packed_length > seq_length:
        raise ValueError(
            f"packed document lengths sum to {packed_length}, above {seq_length}"
        )
    document_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), device=device, dtype=torch.long),
        lengths,
    )
    if packed_length < seq_length:
        document_ids = torch.cat(
            [
                document_ids,
                torch.full(
                    (seq_length - packed_length,),
                    -1,
                    device=device,
                    dtype=torch.long,
                ),
            ]
        )

    all_valid_indices = torch.where(loss_mask == 1)[0]

    sample_indices = [torch.arange(seq_length, device=device)]
    n_per_depth = [seq_length]
    prev_indices = all_valid_indices

    for d in range(1, num_depths):
        candidate_targets = prev_indices[prev_indices >= d]
        candidate_anchors = candidate_targets - d
        same_document = (document_ids[candidate_targets] >= 0) & (
            document_ids[candidate_targets] == document_ids[candidate_anchors]
        )
        eligible_indices = candidate_targets[same_document]
        valid_length = max(0, all_valid_indices.shape[0] - d)
        ratio = max(down_sample_ratio**d, down_sample_ratio_min)
        sample_size = min(int(valid_length * ratio), eligible_indices.shape[0])

        if sample_size <= 0:
            break

        if eligible_indices.shape[0] > sample_size:
            random_selection = torch.randperm(eligible_indices.shape[0], device=device)[
                :sample_size
            ]
            sampled_idx = eligible_indices[random_selection]
            sampled_idx = torch.sort(sampled_idx)[0]
        else:
            sampled_idx = eligible_indices

        next_candidates = (sampled_idx + 1) % seq_length
        if filter_position_zero:
            next_candidates = next_candidates[next_candidates != 0]
        mask = torch.isin(next_candidates, all_valid_indices)
        prev_indices = next_candidates[mask]

        sample_indices.append(sampled_idx - d)
        n_per_depth.append(sampled_idx.shape[0])

    anchor_pos = torch.cat(sample_indices)
    depth = torch.cat(
        [
            torch.full((n,), i, device=device, dtype=torch.long)
            for i, n in enumerate(n_per_depth)
        ]
    )
    return anchor_pos, depth


def create_peagle_mask_mod(anchor_pos, depth, lengths, total_seq_len):
    lengths = lengths.to(device=anchor_pos.device, dtype=torch.long).flatten()
    document_ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=lengths.device, dtype=torch.long),
        lengths,
    )
    document_ids = torch.cat(
        [
            document_ids,
            -1
            * torch.ones(
                total_seq_len - document_ids.shape[0],
                device=lengths.device,
                dtype=torch.long,
            ),
        ]
    ).contiguous()

    def peagle_mask_mod(_b, _h, q_idx, kv_idx):
        q_anchor_pos = anchor_pos[q_idx]
        kv_anchor_pos = anchor_pos[kv_idx]
        q_depth = depth[q_idx]
        kv_depth = depth[kv_idx]

        same_document = document_ids[q_anchor_pos] == document_ids[kv_anchor_pos]
        is_not_padding = document_ids[q_anchor_pos] != -1
        same_rollout = q_anchor_pos == kv_anchor_pos
        kv_depth0 = kv_depth == 0
        in_depth_order = q_depth >= kv_depth
        is_anchor_causal = q_anchor_pos >= kv_anchor_pos

        return (
            is_not_padding
            & same_document
            & ((kv_depth0 & is_anchor_causal) | (same_rollout & in_depth_order))
        )

    return peagle_mask_mod
