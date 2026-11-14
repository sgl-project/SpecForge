"""Sequence shards preserve teacher lookahead and global positions."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from specforge.application.composition import resolve_run
from specforge.training.canonical_data import collate, shard_sequence
from tests.test_modeling.test_canonical_eagle3 import model_config
from tests.test_runtime.test_canonical_training import run_config, target_directory


class SequenceParallelTest(unittest.TestCase):
    def test_lookahead_and_collation(self):
        sample = {
            "input_ids": torch.arange(7)[None],
            "attention_mask": torch.ones(1, 7, dtype=torch.bool),
            "loss_mask": torch.ones(1, 7),
        }
        left = shard_sequence(sample, rank=0, size=2, overlap=2)
        right = shard_sequence(sample, rank=1, size=2, overlap=2)
        self.assertEqual(left["input_ids"].tolist(), [[0, 1, 2, 3, 4, 5]])
        self.assertEqual(right["input_ids"].tolist(), [[4, 5, 6, 0, 0, 0]])
        self.assertEqual(right["position_ids"].tolist(), [[4, 5, 6, 7, 8, 9]])
        result = collate([left, right], keys=("input_ids", "loss_mask"))
        self.assertEqual(result["position_ids"].shape, (2, 6))
        self.assertEqual(result["attention_mask"][1].sum(), 3)

    def test_canonical_eagle_accepts_usp(self):
        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            cfg = run_config(
                target,
                model_config(),
                training={
                    "attention_backend": "usp",
                    "sp_ulysses_size": 2,
                    "objective_options": {"ttt_length": 3},
                },
            )
            run = resolve_run(cfg)
            self.assertEqual(run.config.training.sp_ulysses_size, 2)


if __name__ == "__main__":
    unittest.main()
