"""RoPE positions describe local token chunks before Ulysses exchanges."""

import unittest

import torch

from specforge.core.eagle3_adapters import UspAdapter
from specforge.data.preprocessing import OfflineEagle3Dataset
from specforge.data.utils import DataCollatorWithPadding


def _raw(length):
    ids = torch.arange(length).unsqueeze(0)
    return {
        "input_ids": ids,
        "loss_mask": torch.ones_like(ids),
        "aux_hidden_state": ids.unsqueeze(-1).expand(-1, -1, 6).float(),
        "hidden_state": ids.unsqueeze(-1).expand(-1, -1, 2).float(),
    }


def _batch_and_view(raw, *, rank, ulysses, ring, max_len, overlap):
    shard = OfflineEagle3Dataset.process_data_usp(
        raw, max_len=max_len, ttt_length=overlap,
        sp_rank=rank, sp_size=ulysses * ring,
        ring_rank=rank // ulysses, sp_ring_size=ring,
    )
    # Only topology sizes are needed for these CPU-only contracts.
    collator = DataCollatorWithPadding.__new__(DataCollatorWithPadding)
    collator.sp_degree = ulysses * ring
    collator.ulysses_degree = ulysses
    batch = collator([shard])
    adapter = UspAdapter.__new__(UspAdapter)
    adapter.sp_ulysses_degree = ulysses
    rows = adapter.backbone_row_count(
        seq_length=batch["hidden_state"].shape[1], ttt_length=overlap
    )
    view = adapter.backbone_view(
        row_count=rows,
        global_input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        hidden_states=batch["hidden_state"],
    )
    return shard, view


class UspPositionIdsTest(unittest.TestCase):
    def test_ulysses_peers_use_their_own_global_positions(self):
        for rank in (0, 1):
            with self.subTest(rank=rank):
                shard, view = _batch_and_view(
                    _raw(8), rank=rank, ulysses=2, ring=1,
                    max_len=8, overlap=1,
                )
                expected = torch.arange(rank * 4, (rank + 1) * 4).unsqueeze(0)
                torch.testing.assert_close(shard["position_ids"], expected)
                torch.testing.assert_close(view.position_ids, expected)

    def test_topologies_lengths_truncation_and_overlap(self):
        for ulysses, ring in ((2, 1), (1, 2), (2, 2), (4, 1)):
            for length, max_len in ((1, 1), (7, 7), (8, 8), (17, 17), (17, 9)):
                for overlap in (1, 3, 7):
                    raw = _raw(length)
                    global_len = min(length, max_len)
                    chunk = (global_len + ulysses * ring - 1) // (ulysses * ring)
                    for rank in range(ulysses * ring):
                        with self.subTest(
                            ulysses=ulysses, ring=ring, length=length,
                            max_len=max_len, overlap=overlap, rank=rank,
                        ):
                            shard, view = _batch_and_view(
                                raw, rank=rank, ulysses=ulysses, ring=ring,
                                max_len=max_len, overlap=overlap,
                            )
                            block_start = rank * chunk
                            expected_positions = torch.arange(
                                block_start, block_start + chunk
                            ).unsqueeze(0)
                            torch.testing.assert_close(
                                shard["position_ids"], expected_positions
                            )
                            torch.testing.assert_close(
                                view.position_ids, expected_positions
                            )
                            self.assertEqual(view.hidden_states.shape[1], chunk)

                            # Changing positions must not change the rank's data,
                            # rollout overlap, valid lengths, or loss masking.
                            local_len = chunk + overlap
                            start = rank * chunk
                            valid = max(0, min(local_len, global_len - start))
                            expected_ids = torch.zeros(1, local_len, dtype=torch.long)
                            expected_ids[0, :valid] = torch.arange(start, start + valid)
                            torch.testing.assert_close(shard["input_ids"], expected_ids)
                            expected_attention = torch.zeros_like(expected_ids)
                            expected_attention[:, :valid] = 1
                            torch.testing.assert_close(
                                shard["attention_mask"], expected_attention
                            )
                            expected_loss = expected_attention.clone()
                            last = global_len - 1 - start
                            if 0 <= last < valid:
                                expected_loss[0, last] = 0
                            torch.testing.assert_close(shard["loss_mask"], expected_loss)
                            for name, width in (("hidden_state", 6), ("target", 2)):
                                torch.testing.assert_close(
                                    shard[name],
                                    expected_ids.unsqueeze(-1).expand(-1, -1, width).float(),
                                )
                    self.assertTrue(raw["loss_mask"].all())


if __name__ == "__main__":
    unittest.main()
