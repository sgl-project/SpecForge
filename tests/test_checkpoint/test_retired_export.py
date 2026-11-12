"""One native publisher, isolated canonical imports and legacy-file entry."""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.merge_mtp_to_base import main
from specforge.application.composition import resolve_run
from specforge.config import load_config
from specforge.config.draft.codec import decode_legacy_config
from specforge.training.canonical import resolve_config
from tests.test_config.test_draft_legacy_mtp import mtp_payload
from tests.test_runtime.test_canonical_training import run_config, target_directory


class RetiredExportTest(unittest.TestCase):
    def test_old_merge_cli_delegates_to_checked_native_export(self):
        with patch("scripts.merge_mtp_to_base.export_to_sglang") as export:
            main(
                [
                    "--base-model-path",
                    "target",
                    "--mtp-checkpoint-path",
                    "draft",
                    "--output-path",
                    "output",
                    "--copy-target",
                ]
            )
        export.assert_called_once_with(
            "draft", None, "output", target_model="target", copy_target=True
        )

    def test_canonical_import_does_not_load_old_implementations(self):
        helper = Path(__file__).parent / "helpers/check_canonical_imports.py"
        root = Path(__file__).resolve().parents[2]
        # -m preserves the source checkout on sys.path for this import-boundary check.
        subprocess.run(
            [sys.executable, "-m", "tests.test_checkpoint.helpers." + helper.stem],
            cwd=root,
            check=True,
        )

    def test_mtp_recipes_use_canonical_sdpa(self):
        from tests.test_checkpoint.test_native_contracts import FIXTURE

        target = json.loads((FIXTURE / "config.json").read_text())["text_config"]
        target_config = patch(
            "specforge.checkpoint.resolve.read_target_config", return_value=target
        )
        target_config.start()
        self.addCleanup(target_config.stop)
        for recipe in (
            "offline/colocated/qwen3.5-4b-native-mtp.yaml",
            "online/disaggregated/managed-local/qwen3.5-4b-mtp-npu.yaml",
        ):
            with self.subTest(recipe=recipe):
                resolved = resolve_run(load_config("examples/configs/" + recipe))
                self.assertEqual(
                    resolved.config.draft["architectures"], ["MTPDraftModel"]
                )
                self.assertEqual(resolved.config.training.attention_backend, "sdpa")
                self.assertEqual(
                    resolved.config.training.objective_options["objective_chunk_size"],
                    4096,
                )

    def test_legacy_config_file_resolves_through_codec(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            target = target_directory(root)
            payload = mtp_payload() | {"hidden_size": 32, "vocab_size": 64}
            expected = decode_legacy_config(
                payload, target_config={"hidden_size": 32, "vocab_size": 64}
            )
            path = root / "old.json"
            path.write_text(json.dumps(payload))
            cfg = run_config(target, expected).model_copy(update={"draft": str(path)})
            self.assertEqual(resolve_config(cfg), expected)
            self.assertEqual(
                resolve_run(cfg).config.draft["architectures"], ["MTPDraftModel"]
            )
            old_recipe = cfg.model_copy(update={"draft": None})
            old_recipe.training = cfg.training.model_copy(update={"strategy": "mtp"})
            with self.assertRaisesRegex(ValueError, "MTP now requires draft"):
                resolve_run(old_recipe)
            cfg.training.mtp_objective_chunk_size = 128
            self.assertEqual(
                resolve_run(cfg).config.training.objective_options[
                    "objective_chunk_size"
                ],
                128,
            )
            cfg.training.objective_options = {"objective_chunk_size": 64}
            with self.assertRaisesRegex(ValueError, "conflicting MTP"):
                resolve_run(cfg)


if __name__ == "__main__":
    unittest.main()
