"""Public schema and separate training/serving serialization."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from specforge.config.draft import AutoDraftConfig
from specforge.config.draft.format import encode_format
from specforge.export.checkpoint_io import save_serving_draft
from specforge.modeling.auto import AutoDraftModel
from tests.test_modeling.test_all_draft_configs import architecture_payloads


class DraftFormatTest(unittest.TestCase):
    def test_all_architectures_roundtrip_canonical_without_hf(self):
        for name, payload in architecture_payloads().items():
            with self.subTest(architecture=name):
                config = AutoDraftConfig.from_inline(payload)
                encoded = encode_format(config)
                self.assertEqual(encoded["schema_version"], 1)
                self.assertEqual(
                    AutoDraftConfig.from_inline(encoded).layers, config.layers
                )
                self.assertEqual(
                    AutoDraftConfig.from_dict(encoded).to_dict(), config.to_dict()
                )

    def test_unknown_versions_and_component_fields_fail(self):
        data = encode_format(
            AutoDraftConfig.from_inline(architecture_payloads()["DFlashDraftModel"])
        )
        for value in (0, 2, True, "1"):
            with (
                self.subTest(version=value),
                self.assertRaisesRegex(ValueError, "schema_version"),
            ):
                AutoDraftConfig.from_inline({**data, "schema_version": value})
        for field in ("sliding_window", "is_causal", "attention_config"):
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "legacy structural"),
            ):
                AutoDraftConfig.from_inline({**data, field: None})
        data["layers"][0]["attention"]["typo"] = 1
        with self.assertRaisesRegex(ValueError, "Unknown"):
            AutoDraftConfig.from_inline(data)

    def test_layer_shorthand_and_list_resolve_identically(self):
        data = encode_format(
            AutoDraftConfig.from_inline(architecture_payloads()["DFlashDraftModel"])
        )
        full = AutoDraftConfig.from_inline(data)
        data["layers"] = data["layers"][0]
        self.assertEqual(AutoDraftConfig.from_inline(data).layers, full.layers)

    def test_heterogeneous_checkpoint_reloads_but_serving_export_fails_before_writes(
        self,
    ):
        data = architecture_payloads()["DFlashDraftModel"]
        data.update(
            num_hidden_layers=2,
            layers=[
                {"attention": {"head_dim": 4}},
                {"attention": {"head_dim": 8}, "ffn": {"intermediate_size": 40}},
            ],
        )
        model = AutoDraftModel.from_config(AutoDraftConfig.from_inline(data))
        before = model.config.to_dict()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            payload = json.loads(Path(directory, "config.json").read_text())
            self.assertEqual(payload["layers"][1]["attention"]["head_dim"], 8)
            restored = AutoDraftModel.from_pretrained(directory, local_files_only=True)
            self.assertEqual(restored.config.layers, model.config.layers)
            torch.testing.assert_close(
                restored.state_dict(), model.state_dict(), rtol=0, atol=0
            )
            dest = Path(directory, "serving")
            with self.assertRaisesRegex(ValueError, "legacy serving"):
                save_serving_draft(model, dest, state_dict=model.state_dict())
            self.assertFalse(dest.exists())
        self.assertEqual(model.config.to_dict(), before)

    def test_serving_uses_legacy_config_and_keeps_canonical_sidecar(self):
        model = AutoDraftModel.from_config(
            AutoDraftConfig.from_inline(architecture_payloads()["DFlashDraftModel"])
        )
        with tempfile.TemporaryDirectory() as directory:
            save_serving_draft(model, directory, state_dict=model.state_dict())
            legacy = json.loads(Path(directory, "config.json").read_text())
            canonical = json.loads(Path(directory, "specforge_config.json").read_text())
            self.assertNotIn("schema_version", legacy)
            self.assertNotIn("layers", legacy)
            self.assertEqual(
                legacy["block_size"], legacy["dflash_config"]["block_size"]
            )
            self.assertEqual(canonical["schema_version"], 1)
            self.assertEqual(
                AutoDraftConfig.from_inline(canonical).layers, model.config.layers
            )
