import os
import tempfile
import unittest

import torch

from specforge.export.checkpoint_io import resolve_training_state
from specforge.training.checkpoint import STATE_FILE, consolidate_draft_state


class TestExpertParallelCheckpoint(unittest.TestCase):
    """Consolidating the per-rank draft payloads an expert-parallel run writes.

    The tensor names here are deliberately arbitrary: consolidation unions the
    rank payloads and counts the shards, and knows nothing about how a draft
    architecture names its experts.
    """

    @staticmethod
    def _expert_state(expert, value):
        return {
            f"blocks.0.moe.experts.{expert}.{name}": torch.tensor([value])
            for name in ("w1.weight", "w2.weight")
        }

    def _write_rank(self, directory, rank, payload):
        torch.save(
            {"draft_state_dict": payload},
            os.path.join(directory, f"training_state_rank{rank}.pt"),
        )

    def test_rank_local_expert_shards_are_consolidated_for_export(self):
        with tempfile.TemporaryDirectory() as directory:
            replicated = {"blocks.0.attn.q.weight": torch.ones(2, 2)}
            torch.save(
                {
                    "expert_parallel_size": 2,
                    "draft_state_dict": dict(replicated),
                },
                os.path.join(directory, STATE_FILE),
            )
            self._write_rank(directory, 0, {**replicated, **self._expert_state(0, 0.0)})
            self._write_rank(directory, 1, {**replicated, **self._expert_state(1, 1.0)})

            state = resolve_training_state(directory)

        merged = state["draft_state_dict"]
        self.assertIn("blocks.0.moe.experts.0.w1.weight", merged)
        self.assertIn("blocks.0.moe.experts.1.w1.weight", merged)
        # The replicated tensor appears in every shard and is kept once.
        self.assertIn("blocks.0.attn.q.weight", merged)

    def test_consolidation_rejects_a_missing_rank_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            shared = {
                "expert_parallel_size": 2,
                "draft_state_dict": {"shared": torch.ones(1)},
            }
            self._write_rank(directory, 0, self._expert_state(0, 0.0))
            with self.assertRaisesRegex(ValueError, "incomplete.*found 1"):
                consolidate_draft_state(directory, shared)

    def test_single_rank_checkpoints_are_not_shard_counted(self):
        with tempfile.TemporaryDirectory() as directory:
            shared = {
                "expert_parallel_size": 1,
                "draft_state_dict": {"shared": torch.ones(1)},
            }
            self._write_rank(directory, 0, {"shared": torch.ones(1)})
            merged = consolidate_draft_state(directory, shared)
        self.assertEqual(sorted(merged), ["shared"])

    def test_consolidation_rejects_duplicate_shape_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            shared = {"draft_state_dict": {"shared": torch.ones(2)}}
            self._write_rank(directory, 0, {"shared": torch.ones(3)})
            with self.assertRaisesRegex(ValueError, "conflicting"):
                consolidate_draft_state(directory, shared)


if __name__ == "__main__":
    unittest.main(verbosity=2)
