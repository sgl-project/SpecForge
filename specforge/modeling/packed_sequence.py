"""Document boundaries for packed, single-row EAGLE3 training batches."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PackedSequenceLayout:
    """Token metadata shared by segment shifts and the Flex Attention mask.

    Lengths are small CPU metadata; token metadata lives beside the model's
    tensors. Padding remains local to each document during every TTT shift.
    """

    document_ids: torch.Tensor
    positions: torch.Tensor
    document_lengths: torch.Tensor
    padded_denominator: int
    maximum_length: int

    @classmethod
    def from_lengths(cls, lengths, sequence_length: int, device):
        if not isinstance(lengths, torch.Tensor):
            lengths = torch.tensor(lengths, dtype=torch.long)
        if lengths.device.type != "cpu" or lengths.dtype != torch.long:
            raise ValueError("sequence_lengths must be a CPU int64 tensor")
        if lengths.ndim != 1 or not lengths.numel() or bool((lengths <= 0).any()):
            raise ValueError("sequence_lengths must contain positive document lengths")
        if int(lengths.sum()) != sequence_length:
            raise ValueError("sequence_lengths must sum to the packed sequence length")
        document_ids = torch.repeat_interleave(torch.arange(lengths.numel()), lengths)
        starts = lengths.cumsum(0) - lengths
        positions = torch.arange(sequence_length) - starts[document_ids]
        return cls(
            document_ids=document_ids.to(device),
            positions=positions.to(device),
            document_lengths=lengths[document_ids].to(device),
            padded_denominator=int(lengths.numel() * lengths.max()),
            maximum_length=int(lengths.max()),
        )

    def shift_left(self, tensor: torch.Tensor) -> torch.Tensor:
        """Drop each document's first entry and append one zero to that document."""
        if tensor.shape[:2] != (1, self.positions.numel()):
            raise ValueError(
                "packed tensors must have shape [1, sum(sequence_lengths), ...]"
            )
        shifted = torch.cat((tensor[:, 1:], torch.zeros_like(tensor[:, -1:])), dim=1)
        valid = (self.positions + 1 < self.document_lengths).to(tensor.device)
        return shifted * valid.reshape(1, -1, *([1] * (tensor.ndim - 2)))


def generate_packed_eagle3_mask(
    layout: PackedSequenceLayout, query_length: int, depth: int
):
    """Match independent EAGLE3 causal prefixes and diagonal TTT cache suffixes."""
    document_ids = layout.document_ids
    positions = layout.positions
    lengths = layout.document_lengths

    def mask_mod(_b, _h, q_idx, kv_idx):
        # Flex evaluates complete tiles, including indices outside the actual Q/KV
        # sizes. Clamp metadata reads; the explicit bounds mask removes those cells.
        safe_q = q_idx.clamp(max=query_length - 1)
        kv_row = kv_idx % query_length
        same_document = document_ids[safe_q] == document_ids[kv_row]
        valid_query = (q_idx < query_length) & (
            positions[safe_q] < lengths[safe_q] - depth
        )
        valid_key = positions[kv_row] < lengths[kv_row] - depth
        causal = (kv_idx < query_length) & (q_idx >= kv_idx)
        suffix = (kv_idx >= query_length) & (kv_row == q_idx)
        return same_document & valid_query & valid_key & (causal | suffix)

    return mask_mod
