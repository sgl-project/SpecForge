# coding=utf-8
"""Offline EAGLE vocab mapping is derivable from canonical feature files."""

import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

from specforge.config import Config
from specforge.training.assembly import (
    _ensure_offline_vocab_mapping,
    _install_dataset_vocab_mapping,
    _prompt_cache_key,
)
from specforge.training.vocab_mapping import count_effective_feature_tokens


class _DraftWithMapping(torch.nn.Module):
    def __init__(self, target_vocab_size=8, draft_vocab_size=4):
        super().__init__()
        self.register_buffer("t2d", torch.zeros(target_vocab_size, dtype=torch.bool))
        self.register_buffer("d2t", torch.zeros(draft_vocab_size, dtype=torch.long))
        self.vocab_mapping_loaded = False

    def load_vocab_mapping(self, path):
        mapping = torch.load(path, weights_only=True)
        self.t2d.copy_(mapping["t2d"])
        self.d2t.copy_(mapping["d2t"])
        self.vocab_mapping_loaded = True


class OfflineVocabMappingTest(unittest.TestCase):
    def test_vocab_sizes_only_change_the_mapping_cache_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = Config.model_validate(
                {
                    "model": {
                        "target_model_path": "target/model",
                        "target_backend": "hf",
                    },
                    "data": {
                        "train_data_path": "train.jsonl",
                        "cache_dir": directory,
                    },
                }
            )
            prompt_key = _prompt_cache_key(cfg)
            counts = Counter({1: 3, 2: 2, 3: 1})

            for target_vocab_size, draft_vocab_size in ((8, 4), (9, 4), (8, 5)):
                bundle = SimpleNamespace(
                    draft_model=_DraftWithMapping(
                        target_vocab_size=target_vocab_size,
                        draft_vocab_size=draft_vocab_size,
                    ),
                    target_vocab_size=target_vocab_size,
                    draft_vocab_size=draft_vocab_size,
                )
                _install_dataset_vocab_mapping(
                    cfg,
                    bundle,
                    counts=counts,
                    dataset_identity=prompt_key,
                )
                self.assertEqual(_prompt_cache_key(cfg), prompt_key)

            mapping_files = list((Path(directory) / "vocab_mapping").glob("*.pt"))

        self.assertEqual(len(mapping_files), 3)

    def test_counts_only_loss_bearing_tokens_across_feature_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(
                {
                    "input_ids": torch.tensor([1, 2, 2, 3]),
                    "loss_mask": torch.tensor([0, 1, 1, 1]),
                },
                root / "000.ckpt",
            )
            torch.save(
                {
                    "input_ids": torch.tensor([[2, 4, 4, 5]]),
                    "loss_mask": torch.tensor([[1, 1, 0, 1]]),
                },
                root / "001.ckpt",
            )

            counts = count_effective_feature_tokens(
                directory, max_length=3, target_vocab_size=8
            )

        self.assertEqual(counts, {2: 3, 4: 1})

    def test_rejects_out_of_range_token_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.save(
                {
                    "input_ids": torch.tensor([0, 8]),
                    "loss_mask": torch.tensor([1, 1]),
                },
                Path(directory) / "bad.ckpt",
            )
            with self.assertRaisesRegex(ValueError, "outside target vocab"):
                count_effective_feature_tokens(directory, target_vocab_size=8)

    def test_rejects_empty_feature_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "no offline feature files"):
                count_effective_feature_tokens(directory)

    def test_local_offline_assembly_builds_and_reuses_cached_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_dir = root / "features"
            feature_dir.mkdir()
            torch.save(
                {
                    "input_ids": torch.tensor([1, 2, 2, 3, 6]),
                    "loss_mask": torch.tensor([0, 1, 1, 1, 0]),
                },
                feature_dir / "000.ckpt",
            )
            cfg = Config.model_validate(
                {
                    "model": {
                        "target_model_path": "target/model",
                        "target_backend": "hf",
                    },
                    "data": {
                        "hidden_states_path": str(feature_dir),
                        "cache_dir": str(root / "cache"),
                    },
                }
            )
            first_draft = _DraftWithMapping()
            first = SimpleNamespace(
                draft_model=first_draft,
                target_vocab_size=8,
                draft_vocab_size=4,
            )
            _ensure_offline_vocab_mapping(cfg, first)
            cached = list((root / "cache" / "vocab_mapping").glob("*.pt"))
            self.assertEqual(len(cached), 1)
            self.assertTrue(first_draft.vocab_mapping_loaded)

            second_draft = _DraftWithMapping()
            second = SimpleNamespace(
                draft_model=second_draft,
                target_vocab_size=8,
                draft_vocab_size=4,
            )
            _ensure_offline_vocab_mapping(cfg, second)

        self.assertTrue(second_draft.vocab_mapping_loaded)
        self.assertTrue(torch.equal(first_draft.t2d, second_draft.t2d))
        self.assertTrue(torch.equal(first_draft.d2t, second_draft.d2t))


if __name__ == "__main__":
    unittest.main(verbosity=2)
