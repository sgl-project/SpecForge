# coding=utf-8
"""CPU-only contracts of the specforge_tpu plugin (no torch_tpu required)."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from specforge_tpu import patches, shapes

from specforge.launch_plan import CommandSpec


class ShapesTest(unittest.TestCase):
    def test_registration_uses_longest_sequence_by_default(self):
        self.check_registered_shapes(buckets=None, expected_length=5)

    def test_registration_uses_configured_buckets(self):
        self.check_registered_shapes(buckets="4,8", expected_length=8)

    def check_registered_shapes(self, buckets, expected_length):
        import specforge_tpu

        from specforge import distributed, launch_plan, utils
        from specforge.algorithms.common import batch_shapes
        from specforge.algorithms.objectives.inputs import (
            prepare_anchor_capacity,
            sample_anchors,
        )
        from specforge.training.canonical_data import build_collator

        for owner, name in (
            (specforge_tpu, "_registered"),
            (utils, "get_local_device"),
            (distributed, "_bind_local_device"),
            (launch_plan, "_trainer_command"),
            (batch_shapes, "padded_sequence_length"),
            (batch_shapes, "anchor_capacity"),
        ):
            self.addCleanup(setattr, owner, name, getattr(owner, name))
        with (
            mock.patch.dict(
                os.environ,
                {"SPECFORGE_DEVICE": "tpu"},
            ),
            mock.patch.dict(distributed._DISTRIBUTED_BACKENDS),
            mock.patch.object(
                patches, "import_module", side_effect=[ImportError("torch_tpu"), None]
            ) as compiler,
        ):
            os.environ.pop("WORLD_SIZE", None)
            if buckets is None:
                os.environ.pop("SPECFORGE_TPU_SEQ_BUCKETS", None)
            else:
                os.environ["SPECFORGE_TPU_SEQ_BUCKETS"] = buckets
            with self.assertRaises(ImportError):
                specforge_tpu.register()
            specforge_tpu.register()
            specforge_tpu.register()
            self.assertEqual(compiler.call_count, 2)
            self.assertEqual(distributed._distributed_backend("tpu"), "tpu_dist")

            ids = torch.ones(1, 5, dtype=torch.long)
            feature = {"input_ids": ids, "loss_mask": ids, "lengths": torch.tensor([5])}
            batch = build_collator(keys=("input_ids", "loss_mask"))([feature])
            self.assertEqual(tuple(batch["input_ids"].shape), (1, expected_length))
            self.assertEqual(batch["attention_mask"].sum().item(), 5)
            self.assertIs(batch["lengths"], feature["lengths"])
            for valid_count in (4, 0):
                if not valid_count:
                    batch["loss_mask"].zero_()
                capacity = prepare_anchor_capacity(batch, None)[
                    "anchor_capacity"
                ].numel()
                valid = batch["loss_mask"].bool()
                _, keep = sample_anchors(valid[:, :-1] & valid[:, 1:], 8, capacity)
                self.assertEqual(keep.numel(), expected_length - 1)
                self.assertEqual(keep.sum().item(), valid_count)

    def test_length_past_the_largest_bucket_raises(self):
        with self.assertRaises(ValueError):
            shapes.bucket_for(9, (4, 8))

    def test_offline_recipe_selects_dspark(self):
        from specforge.application.composition import resolve_run
        from specforge.config import load_config

        cfg = load_config(
            "examples/configs/offline/colocated/gpt-oss-120b-dspark-tpu.yaml"
        )
        self.assertEqual(resolve_run(cfg).algorithm.spec.name, "dspark")


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
