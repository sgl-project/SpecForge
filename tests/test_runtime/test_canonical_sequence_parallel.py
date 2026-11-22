"""Sequence shards preserve teacher lookahead and global positions."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from specforge.application.composition import resolve_run
from specforge.training.canonical_data import collate, normalize_sample, shard_sequence
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

    def test_saved_positions_mask_and_unpadded_loss_scale(self):
        keys = ("input_ids", "loss_mask")
        raw = {
            "input_ids": torch.arange(9),
            "loss_mask": torch.ones(9),
            "attention_mask": torch.tensor([1] * 7 + [0, 0]),
            "position_ids": torch.arange(9) + 17,
        }
        sample = normalize_sample(raw, keys=keys, max_len=9)
        right = shard_sequence(sample, rank=1, size=2, overlap=3)
        batch = collate([right], keys=keys)
        self.assertEqual(batch["position_ids"][0, :4].tolist(), [22, 23, 24, 25])
        self.assertEqual(batch["attention_mask"].sum(), 2)
        # The eager EAGLE objective divides each depth by the capture width.
        # Rank averaging must recover that width, including when 9 % 2 != 0.
        self.assertEqual(batch["shard_loss_denominator"].item(), 4.5)

    def test_mixed_saved_and_implicit_positions(self):
        keys = ("input_ids", "loss_mask")
        plain = normalize_sample(
            {"input_ids": torch.arange(4), "loss_mask": torch.ones(4)},
            keys=keys,
            max_len=4,
        )
        custom = {**plain, "position_ids": torch.tensor([[13, 14, 15, 16]])}
        for features in ([plain, custom], [custom, plain]):
            positions = collate(features, keys=keys)["position_ids"]
            self.assertEqual(positions[0, 0].item(), 0 if features[0] is plain else 13)
            self.assertEqual(positions[1, 0].item(), 0 if features[1] is plain else 13)

    def test_multi_axis_positions_are_sharded_on_sequence_axis(self):
        keys = ("input_ids", "loss_mask")
        raw = {"input_ids": torch.arange(7), "loss_mask": torch.ones(7)}
        positions = torch.arange(21).reshape(3, 1, 7)
        for saved in (positions, positions.squeeze(1)):
            sample = normalize_sample(
                {**raw, "position_ids": saved},
                keys=keys,
                max_len=6,
            )
            right = shard_sequence(sample, rank=1, size=2, overlap=2)
            batch = collate([right], keys=keys)
            torch.testing.assert_close(
                batch["position_ids"][..., :3], positions[..., 3:6]
            )
            plain = normalize_sample(raw, keys=keys, max_len=4)
            combined = collate([plain, sample], keys=keys)["position_ids"]
            self.assertEqual(combined.shape, (3, 2, 6))
            torch.testing.assert_close(combined[:, 1:2], positions[..., :6])
            torch.testing.assert_close(
                combined[:, 0, :4], torch.arange(4).expand(3, -1)
            )

    def test_usp_rejects_internal_attention_gaps(self):
        sample = {
            "input_ids": torch.arange(4)[None],
            "loss_mask": torch.ones(1, 4),
            "attention_mask": torch.tensor([[1, 0, 1, 1]]),
        }
        with self.assertRaisesRegex(ValueError, "right-padding"):
            shard_sequence(sample, rank=0, size=2, overlap=3)

    def test_legacy_manually_sharded_batch_uses_owned_width(self):
        from types import SimpleNamespace
        from unittest.mock import Mock

        from specforge.algorithms.objectives.protocols.eagle3 import Eagle3Objective

        model = Mock()
        model.fusion.side_effect = lambda value: value
        model.new_rollout.return_value.step.side_effect = (
            lambda tokens, hidden, *_: hidden
        )
        runtime = SimpleNamespace(
            attention_backend="usp",
            draft_model=model,
            predictions=lambda *_, **kwargs: kwargs,
        )
        batch = {
            "input_ids": torch.arange(5)[None],
            "loss_mask": torch.ones(1, 5),
            "hidden_states": torch.zeros(1, 5, 8),
        }
        first_depth = next(Eagle3Objective(ttt_length=2).rollout(runtime, batch))
        self.assertEqual(first_depth["denominator"].item(), 3)

    def test_explicit_metadata_keys_are_normalized_once(self):
        raw = {
            "input_ids": torch.arange(7),
            "loss_mask": torch.ones(7),
            "attention_mask": torch.ones(7),
            "position_ids": torch.arange(7) + 20,
            "lengths": torch.tensor([7]),
        }
        keys = tuple(raw)
        sample = normalize_sample(raw, keys=keys, max_len=5)
        batch = collate([sample], keys=keys)
        self.assertEqual(batch["position_ids"].tolist(), [[20, 21, 22, 23, 24]])
        self.assertEqual(batch["lengths"].tolist(), [5])
        self.assertEqual(batch["attention_mask"].shape, (1, 5))

    def test_direct_sharding_builds_missing_attention_mask(self):
        sample = {"input_ids": torch.arange(7)[None], "loss_mask": torch.ones(1, 7)}
        right = shard_sequence(sample, rank=1, size=2, overlap=2)
        self.assertEqual(right["attention_mask"].tolist(), [[1, 1, 1, 0, 0, 0]])

    def test_cached_offline_backends_validate_saved_padding(self):
        raw = {
            "input_ids": torch.arange(5),
            "loss_mask": torch.ones(5),
            "hidden_states": torch.zeros(5, 144),
            "target_last_hidden_states": torch.zeros(5, 32),
            "attention_mask": torch.tensor([0, 1, 1, 1, 0]),
        }
        with TemporaryDirectory() as temp:
            target = target_directory(Path(temp))
            for backend in ("fa", "flex_attention", "eager", "sdpa"):
                cfg = run_config(
                    target, model_config(), training={"attention_backend": backend}
                )
                provider = resolve_run(cfg).algorithm.providers.offline_for("text")
                normalizer = provider.build_normalizer(5)
                if backend in ("fa", "flex_attention"):
                    with self.assertRaisesRegex(ValueError, "right-padding"):
                        normalizer(raw)
                else:
                    self.assertTrue(
                        torch.equal(
                            normalizer(raw)["attention_mask"][0], raw["attention_mask"]
                        )
                    )

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
            # chain weights are normalized inside each shard
            cfg.training.weighting = {"kind": "dpace"}
            with self.assertRaisesRegex(ValueError, "D-PACE weighting"):
                resolve_run(cfg)


if __name__ == "__main__":
    unittest.main()
