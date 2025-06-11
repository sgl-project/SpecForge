from typing import Any, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX, PACK_TYPE
from torchtune.modules.attention_utils import packed_block_causal_mask

def padded_collate_sft_with_mask(
    batch: list[dict[str, Any]],
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    pad_to_multiple_of: int = 1,
    stack_on_new_dim: bool = False,
) -> dict[str, torch.Tensor]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors. Preserves the mask from ShareGPTToMessages.

    Args:
        batch (list[dict[str, Any]]): A list of dictionaries containing samples, including tokens, labels and mask.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.
        pad_to_multiple_of (int): If > 1, pad the sequence to a multiple of this number.
            This is useful for proper sharding with e.g. SequenceParallel.
        stack_on_new_dim (bool): If True, stack any encoder tensors on a new dimension. Default is False

    Returns:
        dict[str, torch.Tensor]: Collated input, label and mask tensors.
    """
    input_ids = pad_sequence(
        [torch.tensor(x["tokens"]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    labels = pad_sequence(
        [torch.tensor(x["labels"]) for x in batch],
        batch_first=True,
        padding_value=ignore_idx,
    )
    # Preserve the original mask from ShareGPTToMessages
    mask = pad_sequence(
        [torch.tensor(x["mask"]) for x in batch],
        batch_first=True,
        padding_value=False,
    )

    # Pad to multiple of N
    if pad_to_multiple_of > 1:
        input_ids = F.pad(
            input_ids,
            (
                0,
                pad_to_multiple_of - (input_ids.size(1) % pad_to_multiple_of),
            ),
            value=padding_idx,
        )
        labels = F.pad(
            labels,
            (
                0,
                pad_to_multiple_of - (labels.size(1) % pad_to_multiple_of),
            ),
            value=ignore_idx,
        )
        mask = F.pad(
            mask,
            (
                0,
                pad_to_multiple_of - (mask.size(1) % pad_to_multiple_of),
            ),
            value=False,
        )

    return {
        "tokens": input_ids,
        "labels": labels,
        "mask": mask,
    }