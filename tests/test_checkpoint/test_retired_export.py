"""Canonical import boundaries and native recipes."""

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from specforge.application.composition import resolve_run
from specforge.config import load_config


class RetiredExportTest(unittest.TestCase):

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
        from specforge.training.capture_contract import resolve_server_capture_contract
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
                capture = resolve_server_capture_contract(
                    resolved.config, algorithm=resolved.algorithm
                )
                self.assertEqual(capture.method, "dflash")
                self.assertEqual(capture.aux_layer_ids, (0,))
                self.assertLess(
                    max(capture.aux_layer_ids), target["num_hidden_layers"] - 1
                )


if __name__ == "__main__":
    unittest.main()
