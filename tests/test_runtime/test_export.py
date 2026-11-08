# coding=utf-8
"""Legacy import helpers; publication is covered in test_checkpoint/test_export.py."""

import json
import os
import tempfile
import unittest

import torch


class TestRoPEConfigCompatibility(unittest.TestCase):
    def _write_config(self, directory, payload):
        path = os.path.join(directory, "config.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    def test_modern_rope_parameters_are_mirrored_for_legacy_readers(self):
        from specforge.export.checkpoint_io import apply_legacy_rope_scaling

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(
                directory,
                {
                    "rope_parameters": {
                        "rope_type": "yarn",
                        "factor": 128.0,
                        "rope_theta": 8_000_000,
                    }
                },
            )
            self.assertTrue(apply_legacy_rope_scaling(directory))
            with open(path, encoding="utf-8") as handle:
                config = json.load(handle)

        self.assertEqual(
            config["rope_scaling"],
            {"rope_type": "yarn", "factor": 128.0},
        )
        self.assertEqual(config["rope_theta"], 8_000_000)

    def test_legacy_rope_scaling_is_mirrored_for_modern_readers(self):
        from specforge.export.checkpoint_io import apply_legacy_rope_scaling

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(
                directory,
                {
                    "rope_theta": 8_000_000,
                    "rope_scaling": {"type": "yarn", "factor": 128.0},
                },
            )
            self.assertTrue(apply_legacy_rope_scaling(directory))
            with open(path, encoding="utf-8") as handle:
                config = json.load(handle)

        self.assertEqual(
            config["rope_parameters"],
            {"type": "yarn", "factor": 128.0, "rope_theta": 8_000_000},
        )

    def test_default_rope_config_is_not_rewritten(self):
        from specforge.export.checkpoint_io import apply_legacy_rope_scaling

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(
                directory,
                {"rope_parameters": {"rope_type": "default"}},
            )
            with open(path, "rb") as handle:
                before = handle.read()
            self.assertFalse(apply_legacy_rope_scaling(directory))
            with open(path, "rb") as handle:
                after = handle.read()

        self.assertEqual(after, before)

    def test_default_rope_theta_is_mirrored_for_legacy_readers(self):
        from specforge.export.checkpoint_io import apply_legacy_rope_scaling

        with tempfile.TemporaryDirectory() as directory:
            path = self._write_config(
                directory,
                {
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 1_000_000,
                    }
                },
            )
            self.assertTrue(apply_legacy_rope_scaling(directory))
            with open(path, encoding="utf-8") as handle:
                config = json.load(handle)

        self.assertEqual(config["rope_theta"], 1_000_000)
        self.assertNotIn("rope_scaling", config)


class TestLegacyVocabMappingCompatibility(unittest.TestCase):
    def setUp(self):
        from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
        from tests.test_runtime import _fixtures as fx

        self.tempdir = tempfile.TemporaryDirectory(prefix="legacy_export_")
        self.addCleanup(self.tempdir.cleanup)
        self.cfg_path = fx.write_draft_config(
            os.path.join(self.tempdir.name, "draft.json")
        )
        self.vocab_path = fx.write_vocab_mapping(
            os.path.join(self.tempdir.name, "mapping.pt")
        )
        config = AutoDraftModelConfig.from_file(self.cfg_path)
        model = AutoDraftModel.from_config(config, torch_dtype=torch.bfloat16)
        self.legacy_state = {
            key: value
            for key, value in model.state_dict().items()
            if "embed" not in key.lower() and key not in {"t2d", "d2t"}
        }

    def test_mapping_restores_legacy_checkpoint_buffers(self):
        from specforge.export.checkpoint_io import materialize_draft

        model = materialize_draft(
            {"draft_state_dict": self.legacy_state},
            self.cfg_path,
            vocab_mapping_path=self.vocab_path,
        )
        expected = torch.load(self.vocab_path, map_location="cpu", weights_only=True)

        self.assertTrue(torch.equal(model.t2d.cpu(), expected["t2d"]))
        self.assertTrue(torch.equal(model.d2t.cpu(), expected["d2t"]))

    def test_missing_mapping_buffers_remain_strict_without_mapping(self):
        from specforge.export.checkpoint_io import materialize_draft

        with self.assertRaisesRegex(ValueError, "d2t.*t2d"):
            materialize_draft(
                {"draft_state_dict": self.legacy_state},
                self.cfg_path,
            )

    def test_mapping_does_not_tolerate_other_missing_weights(self):
        from specforge.export.checkpoint_io import materialize_draft

        incomplete = dict(self.legacy_state)
        incomplete.pop("fc.weight")

        with self.assertRaisesRegex(ValueError, r"fc\.weight"):
            materialize_draft(
                {"draft_state_dict": incomplete},
                self.cfg_path,
                vocab_mapping_path=self.vocab_path,
            )
