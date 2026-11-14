"""Canonical export keeps checkpoint tensors and rejects structural replacements."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from click.testing import CliRunner
from safetensors.torch import load_file, save_file

from specforge.checkpoint import save_draft
from specforge.cli import cli
from specforge.export.model import export_draft
from specforge.loading import load_draft
from specforge.modeling import build_draft_model
from tests.test_modeling.test_canonical_eagle3 import install_mapping, model_config


class ExportContractTest(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.source = Path(directory.name) / "source"
        self.output = Path(directory.name) / "output"

    def save(self, dtype=torch.float32, *, bound=False):
        self.source = self.source.parent / f"source-{dtype}"
        self.output = self.output.parent / f"output-{dtype}"
        model = build_draft_model(
            model_config(target_width=32, bound=bound), dtype=dtype
        )
        install_mapping(model)
        with torch.no_grad():
            model.fusion.fc.weight.flatten()[0] = 0.1234567
        save_draft(model, self.source)
        return model.state_dict()

    def test_export_preserves_every_tensor_and_dtype(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                expected = self.save(dtype)
                export_draft(self.source, None, self.output)
                torch.testing.assert_close(
                    load_draft(self.output).state_dict(),
                    expected,
                    rtol=0,
                    atol=0,
                )

    def test_serving_export_keeps_mapping_and_precision(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                expected = self.save(dtype, bound=True)
                export_draft(self.source, None, self.output, converter="sglang")
                served = load_file(self.output / "model.safetensors")
                self.assertFalse(any(k.startswith("draft_model.") for k in served))
                for key in ("t2d", "d2t", "lm_head.weight"):
                    torch.testing.assert_close(
                        served[key],
                        expected[f"vocabulary.{key}" if key in ("t2d", "d2t") else key],
                        rtol=0,
                        atol=0,
                    )
                torch.testing.assert_close(
                    served["fc.weight"],
                    expected["fusion.fc.weight"],
                    rtol=0,
                    atol=0,
                )

    def test_export_rejects_mapping_replacement(self):
        self.save()
        with self.assertRaisesRegex(ValueError, "never replaces checkpoint tensors"):
            export_draft(self.source, None, self.output, vocab_mapping_path="new.pt")

    def test_missing_mapping_and_other_weights_remain_strict(self):
        original = self.save()
        for missing in ("vocabulary.t2d", "vocabulary.d2t", "fusion.fc.weight"):
            with self.subTest(missing=missing):
                save_file(
                    {k: v for k, v in original.items() if k != missing},
                    self.source / "model.safetensors",
                )
                with self.assertRaises(ValueError):
                    export_draft(self.source, None, self.output)

    def test_cli_uses_the_same_writer(self):
        expected = self.save()
        result = CliRunner().invoke(
            cli,
            [
                "export",
                "--checkpoint",
                str(self.source),
                "--output-dir",
                str(self.output),
            ],
        )
        self.assertEqual(result.exit_code, 0, result.output)
        torch.testing.assert_close(
            load_draft(self.output).state_dict(),
            expected,
            rtol=0,
            atol=0,
        )
