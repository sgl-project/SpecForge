"""EAGLE3 preprocessing preserves supervision until target alignment."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from specforge.algorithms.eagle3.data import normalize_offline_sample
from specforge.data.preprocessing import (
    OfflineEagle3Dataset,
    process_offline_eagle3_sample,
)
from specforge.modeling.target.target_head import TargetHead
from specforge.training.strategies.base import _prepare_eagle_target
from specforge.utils import padding


class Eagle3LossMaskTest(unittest.TestCase):
    def test_terminal_supervision_survives_normalization_and_usp(self):
        head = TargetHead(SimpleNamespace(hidden_size=2, vocab_size=4))

        def prepare_mask(sample):
            return _prepare_eagle_target(
                target_head=head,
                target_repr="hidden_state",
                input_ids=sample["input_ids"],
                target=sample["target"],
                loss_mask=sample["loss_mask"],
                device=torch.device("cpu"),
            )[2]

        for max_len in (1, 5, 8):
            for terminal in (0, 1):
                raw = {
                    "input_ids": torch.arange(8),
                    "loss_mask": torch.ones(8, dtype=torch.long),
                    "aux_hidden_state": torch.zeros(1, 8, 6),
                    "hidden_state": torch.zeros(1, 8, 2),
                }
                raw["loss_mask"][max_len - 1] = terminal
                original = raw["loss_mask"].clone()
                expected_mask = F.pad(original[1:max_len], (0, 1)).view(1, max_len, 1)

                for normalize in (
                    normalize_offline_sample,
                    process_offline_eagle3_sample,
                    OfflineEagle3Dataset.process_data,
                ):
                    with self.subTest(
                        max_len=max_len,
                        terminal=terminal,
                        normalizer=normalize.__name__,
                    ):
                        sample = normalize(raw, max_len)
                        torch.testing.assert_close(
                            sample["loss_mask"][0], original[:max_len]
                        )
                        torch.testing.assert_close(prepare_mask(sample), expected_mask)
                        torch.testing.assert_close(raw["loss_mask"], original)

                for sp_size in (2, 4):
                    chunk = (max_len + sp_size - 1) // sp_size
                    for rank in range(sp_size):
                        shard = OfflineEagle3Dataset.process_data_usp(
                            raw,
                            max_len=max_len,
                            ttt_length=3,
                            sp_rank=rank,
                            sp_size=sp_size,
                            ring_rank=rank,
                            sp_ring_size=sp_size,
                        )
                        local_mask = prepare_mask(shard)
                        full_mask = expected_mask
                        for step in range(3):
                            with self.subTest(
                                max_len=max_len,
                                terminal=terminal,
                                sp=sp_size,
                                rank=rank,
                                step=step,
                            ):
                                padded = F.pad(
                                    full_mask, (0, 0, 0, chunk * sp_size - max_len)
                                )
                                expected = padded[:, rank * chunk : (rank + 1) * chunk]
                                torch.testing.assert_close(
                                    local_mask[:, :chunk], expected
                                )
                            local_mask = padding(local_mask, left=False)
                            full_mask = padding(full_mask, left=False)
                        torch.testing.assert_close(raw["loss_mask"], original)
