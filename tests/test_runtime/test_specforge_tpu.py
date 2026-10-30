# coding=utf-8
"""CPU-only contracts of the specforge_tpu plugin (no torch_tpu required)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from specforge.launch_plan import CommandSpec
from specforge_tpu import patches, shapes


class ShapesTest(unittest.TestCase):
    def test_pads_every_aligned_feature_to_the_bucket(self):
        batch = {
            "input_ids": torch.ones(2, 5, dtype=torch.long),
            "hidden_states": torch.ones(2, 5, 3),
        }
        padded = shapes.pad_batch_to_bucket(batch, (4, 8))
        self.assertEqual(tuple(padded["input_ids"].shape), (2, 8))
        self.assertEqual(tuple(padded["hidden_states"].shape), (2, 8, 3))
        self.assertEqual(int(padded["input_ids"][:, 5:].sum()), 0)

    def test_length_past_the_largest_bucket_raises(self):
        with self.assertRaises(ValueError):
            shapes.bucket_for(9, (4, 8))

    def test_anchor_width_depends_on_the_bucket_only(self):
        loss_mask = torch.zeros(1, 8)
        loss_mask[0, :3] = 1
        self.assertEqual(shapes.shape_only_max_valid_anchors(lambda _: 2, loss_mask), 7)
        self.assertEqual(shapes.shape_only_max_valid_anchors(lambda _: 0, loss_mask), 0)


class TrainerCommandTest(unittest.TestCase):
    def test_trainer_ranks_share_slicebuilder_ports(self):
        cfg = SimpleNamespace(
            deployment=SimpleNamespace(
                trainer=SimpleNamespace(nnodes=1, nproc_per_node=4)
            )
        )
        command = patches._trainer_command(
            lambda cfg, **_: CommandSpec("consumer", ("torchrun",), {}), cfg
        )
        self.assertEqual(
            len(command.env["TORCH_TPU_SLICEBUILDER_ADDRESSES"].split(",")), 4
        )
        self.assertEqual(command.env["TORCH_TPU_DEVICES_PER_HOST"], "4")


if __name__ == "__main__":
    unittest.main()
