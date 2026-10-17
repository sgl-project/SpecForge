"""Regrouping must never reduce native DFlash per-sample supervision counts."""

from types import SimpleNamespace

import torch

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.algorithms.common.hidden_states_data import build_collator


def test_native_anchor_sampling_preserves_counts_across_padding_and_regrouping():
    lengths = [3, 23, 6, 17]
    samples = []
    for length in lengths:
        mask = torch.ones(1, length)
        mask[:, :2] = 0  # prompts must not acquire anchors through padding
        samples.append(
            {
                "input_ids": torch.arange(length).view(1, -1),
                "loss_mask": mask,
                "hidden_states": torch.zeros(1, length, 8),
            }
        )
    expected = [min(5, max(0, length - 3)) for length in lengths]
    for order in ([0, 1, 2, 3], [0, 2, 3, 1]):
        observed = {}
        for start in (0, 2):
            indices = order[start : start + 2]
            batch = build_collator()([samples[index] for index in indices])
            positions, keep = OnlineDFlashModel._sample_anchor_positions(
                SimpleNamespace(num_anchors=5),
                seq_len=batch["input_ids"].shape[1],
                loss_mask=batch["loss_mask"],
                device=torch.device("cpu"),
            )
            for row, index in enumerate(indices):
                observed[index] = int(keep[row].sum())
                selected = positions[row][keep[row]]
                assert bool((selected >= 2).all())
                assert bool((selected + 1 < lengths[index]).all())
        assert [observed[index] for index in range(len(samples))] == expected
