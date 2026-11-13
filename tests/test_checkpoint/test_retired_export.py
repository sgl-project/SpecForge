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
from tests.test_config.test_draft_legacy_dflash import CONFIGS
from tests.test_config.test_draft_legacy_mtp import mtp_payload
from tests.test_runtime.test_canonical_training import run_config, target_directory


class RetiredExportTest(unittest.TestCase):
    def test_bundled_legacy_files_resolve_through_training_entry(self):
        for path in sorted(CONFIGS.glob("*.json")):
            with self.subTest(path=path.name):
                payload = json.loads(path.read_text())
                options = (
                    {"mask_token_id": 1}
                    if payload["architectures"] == ["PEagleDraftModel"]
                    else {}
                )
                target = {
                    "hidden_size": payload["hidden_size"],
                    "vocab_size": payload["vocab_size"],
                    "num_hidden_layers": payload.get("num_target_layers", 128),
                }
                expected = decode_legacy_config(
                    payload, **options, target_config=target
                )
                cfg = run_config(path.parent, expected).model_copy(
                    update={"draft": str(path)}
                )
                if options:
                    cfg.model.mask_token_id = options["mask_token_id"]
                self.assertEqual(resolve_config(cfg, target_config=target), expected)

    def test_legacy_peagle_draft_builds_after_resolution(self):
        """The mask token decodes the draft once; later resolves must accept it."""
        import torch

        from specforge.checkpoint import save_draft
        from specforge.config import Config
        from specforge.modeling import build_draft_model
        from specforge.training.assembly import build_model_bundle

        fixture = Path(__file__).resolve().parents[1] / "fixtures" / "legacy_peagle"
        payload = json.loads((fixture / "config.json").read_text())
        with TemporaryDirectory() as temp:
            root = Path(temp)
            target = target_directory(root)
            (root / "features").mkdir()
            old = root / "old.json"
            old.write_text(json.dumps(payload))
            target_config = json.loads((target / "config.json").read_text())
            canonical = root / "canonical"
            save_draft(
                build_draft_model(
                    decode_legacy_config(
                        payload, target_config=target_config, mask_token_id=63
                    ),
                    dtype=torch.float32,
                ),
                canonical,
            )
            for initialization in ({}, {"weights_from": {"checkpoint": str(canonical)}}):
                with self.subTest(initialization=initialization):
                    cfg = Config.model_validate(
                        dict(
                            model={
                                "target_model_path": str(target),
                                "torch_dtype": "float32",
                                "mask_token_id": 63,
                            },
                            data={
                                "hidden_states_path": str(root / "features"),
                                "dataloader_num_workers": 0,
                            },
                            training={"attention_backend": "eager", "max_steps": 1},
                            draft=str(old),
                            initialization=initialization,
                            output_dir=str(root / "output"),
                        )
                    )
                    resolved = resolve_run(cfg)
                    bundle = build_model_bundle(
                        resolved.config, algorithm=resolved.algorithm
                    )
                    self.assertEqual(bundle.draft_config.spec_config.mask_token_id, 63)

    def test_old_merge_cli_delegates_to_checked_native_export(self):
        with patch("scripts.merge_mtp_to_base.export_draft") as export:
            main(
                [
                    "--base-model-path",
                    "target",
                    "--mtp-checkpoint-path",
                    "draft",
                    "--output-path",
                    "output",
                ]
            )
        export.assert_called_once_with(
            "draft",
            None,
            "output",
            target_model="target",
            copy_target=True,
            export_mode="native",
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
