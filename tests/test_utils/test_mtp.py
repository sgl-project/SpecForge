"""Selective target checkpoint reads remain independent of native draft models."""

import json
import os
import tempfile
import unittest

import torch


class SelectiveCheckpointLoadingTest(unittest.TestCase):
    def test_sharded_selective_loading(self):
        from safetensors.torch import save_file

        from specforge.modeling.target.checkpoint import (
            list_checkpoint_keys,
            load_checkpoint_tensors,
            read_weight_map,
        )

        with tempfile.TemporaryDirectory(prefix="mtp-ckpt-") as tmpdir:
            save_file(
                {
                    "mtp.fc.weight": torch.zeros(4, 4),
                    "model.embed_tokens.weight": torch.zeros(2, 2),
                },
                os.path.join(tmpdir, "model-00001-of-00002.safetensors"),
            )
            save_file(
                {
                    "mtp.norm.weight": torch.ones(4),
                    "lm_head.weight": torch.zeros(2, 2),
                },
                os.path.join(tmpdir, "model-00002-of-00002.safetensors"),
            )
            weight_map = {
                "mtp.fc.weight": "model-00001-of-00002.safetensors",
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "mtp.norm.weight": "model-00002-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
            }
            with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
                json.dump({"weight_map": weight_map}, f)

            self.assertEqual(weight_map, read_weight_map(tmpdir))
            self.assertEqual(4, len(list_checkpoint_keys(tmpdir)))
            selected = load_checkpoint_tensors(
                tmpdir, key_filter=lambda key: key.startswith("mtp.")
            )

        self.assertEqual({"mtp.fc.weight", "mtp.norm.weight"}, set(selected))
        self.assertTrue(torch.equal(selected["mtp.norm.weight"], torch.ones(4)))

    def test_single_file_selective_loading(self):
        from safetensors.torch import save_file

        from specforge.modeling.target.checkpoint import load_checkpoint_tensors

        with tempfile.TemporaryDirectory(prefix="mtp-ckpt-") as tmpdir:
            save_file(
                {"mtp.fc.weight": torch.zeros(4, 4), "other.weight": torch.zeros(1)},
                os.path.join(tmpdir, "model.safetensors"),
            )
            selected = load_checkpoint_tensors(
                tmpdir, key_filter=lambda key: key.startswith("mtp.")
            )
        self.assertEqual({"mtp.fc.weight"}, set(selected))
