"""Index metadata for packing DFlash context while preserving sampled blocks."""

from dataclasses import dataclass

import torch

from specforge.modeling.packed_sequence import PackedSequenceLayout


@dataclass(frozen=True)
class PackedDFlashLayout:
    tokens: PackedSequenceLayout
    lengths: tuple[int, ...]
    document_starts: torch.Tensor
    padded_indices: torch.Tensor
    padded_valid: torch.Tensor

    @classmethod
    def from_lengths(cls, lengths, sequence_length, device):
        tokens = PackedSequenceLayout.from_lengths(lengths, sequence_length, device)
        lengths = torch.as_tensor(lengths, dtype=torch.long)
        starts = lengths.cumsum(0) - lengths
        columns = torch.arange(tokens.maximum_length)
        indices = starts[:, None] + columns[None, :]
        return cls(
            tokens=tokens,
            lengths=tuple(lengths.tolist()),
            document_starts=starts.to(device),
            padded_indices=indices.clamp(max=sequence_length - 1).to(device),
            padded_valid=(columns[None, :] < lengths[:, None]).to(device),
        )

    def padded_loss_mask(self, loss_mask):
        """Rebuild only the small sampling mask, preserving baseline RNG shape."""
        return loss_mask[0, self.padded_indices] * self.padded_valid

    def pack_anchors(self, local_anchors):
        return (local_anchors + self.document_starts[:, None]).reshape(1, -1)

    def anchor_starts(self, packed_anchors):
        return packed_anchors - self.tokens.positions[packed_anchors]

    def anchor_ends(self, packed_anchors):
        return (
            self.anchor_starts(packed_anchors)
            + self.tokens.document_lengths[packed_anchors]
        )

    def compact_anchor_indices(self, valid_anchor_counts, width):
        """Indices of valid prefix slots after the unchanged sorted sampler."""
        if len(valid_anchor_counts) != len(self.lengths) or any(
            not isinstance(count, int) or count < 0 for count in valid_anchor_counts
        ):
            raise ValueError(
                "valid_anchor_counts must contain one nonnegative integer per document"
            )
        return torch.tensor(
            [
                document * width + index
                for document, count in enumerate(valid_anchor_counts)
                for index in range(min(count, width))
            ],
            dtype=torch.long,
            device=self.document_starts.device,
        )


def create_packed_dflash_block_mask(
    anchor_positions,
    block_keep_mask,
    context_start_positions,
    context_length,
    proposal_size,
    mask_mod,
    *,
    block_size=128,
    sliding_window=None,
):
    """Construct sparse tiles without materializing a quadratic token mask.

    A tile's context union is bounded by its earliest start and latest anchor;
    its intersection identifies fully allowed context tiles. A conservative
    union may include extra partial tiles, whose exact token predicate remains
    ``mask_mod``. Draft tiles intersect only the proposals present in a Q tile.
    """
    from torch.nn.attention.flex_attention import BlockMask

    q_block, kv_block = (
        (block_size, block_size) if isinstance(block_size, int) else block_size
    )
    batch, anchors = anchor_positions.shape
    q_length = anchors * proposal_size
    kv_length = context_length + q_length
    device = anchor_positions.device
    q = torch.arange(
        ((q_length + q_block - 1) // q_block) * q_block, device=device
    ).reshape(-1, q_block)
    anchor_index = (q // proposal_size).clamp(max=anchors - 1)
    valid = (q < q_length).unsqueeze(0) & block_keep_mask[:, anchor_index]
    anchor = anchor_positions[:, anchor_index]
    lower = context_start_positions[:, anchor_index]
    if sliding_window is not None:
        lower = torch.maximum(
            lower, anchor + q.remainder(proposal_size) - (sliding_window - 1)
        )

    context_min = torch.where(valid, lower, context_length).amin(-1)
    context_max = torch.where(valid, anchor, 0).amax(-1)
    intersection_min = torch.where(valid, lower, 0).amax(-1)
    intersection_max = torch.where(valid, anchor, context_length).amin(-1)
    any_valid = valid.any(-1)
    all_valid = valid.all(-1)

    kv_start = (
        torch.arange((kv_length + kv_block - 1) // kv_block, device=device) * kv_block
    )
    kv_end = kv_start + kv_block
    context_tiles = (
        (kv_start < context_max[..., None])
        & (kv_end > context_min[..., None])
        & (kv_start < context_length)
        & (context_min < context_max)[..., None]
    )
    full_tiles = (
        all_valid[..., None]
        & (kv_start >= intersection_min[..., None])
        & (kv_end <= intersection_max[..., None])
        & (kv_end <= context_length)
    )
    draft_min = context_length + (q[:, 0] // proposal_size) * proposal_size
    draft_max = context_length + torch.minimum(
        (q[:, -1] // proposal_size + 1) * proposal_size,
        q.new_tensor(q_length),
    )
    draft_tiles = (kv_start < draft_max[:, None]) & (kv_end > draft_min[:, None])
    partial_tiles = any_valid[..., None] & (context_tiles | draft_tiles) & ~full_tiles

    def ordered(mask):
        mask = mask.unsqueeze(1)
        counts = mask.sum(-1, dtype=torch.int32)
        indices = (
            mask.to(torch.int32)
            .argsort(dim=-1, descending=True, stable=True)
            .to(torch.int32)
        )
        return counts, indices

    partial_counts, partial_indices = ordered(partial_tiles)
    full_counts, full_indices = ordered(full_tiles)
    return BlockMask.from_kv_blocks(
        partial_counts,
        partial_indices,
        full_counts,
        full_indices,
        BLOCK_SIZE=(q_block, kv_block),
        mask_mod=mask_mod,
        seq_lengths=(q_length, kv_length),
    )
