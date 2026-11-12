"""Public export modes do not depend on a serving engine or architecture."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from click.testing import CliRunner

from specforge.checkpoint import save_draft
from specforge.checkpoint.native import _TARGETS, register_native_target
from specforge.cli import cli
from specforge.export import export_draft
from specforge.loading import read_draft
from specforge.modeling import build_draft_model
from tests.test_checkpoint.test_loading import write_target
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_mtp_spec import mtp_config
from tests.test_modeling.test_canonical_dspark import model_config as dspark_config


class ExportModesTest(unittest.TestCase):
    def test_registration_does_not_change_standalone_for_any_architecture(self):
        for config in (mtp_config(), draft_config(), dspark_config(bound=True)):
            with (
                self.subTest(architecture=config.architecture),
                patch.dict(_TARGETS),
                TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                save_draft(build_draft_model(config), root / "draft")
                target = root / "target"
                write_target(target)
                path = target / "config.json"
                path.write_text(
                    json.dumps(
                        json.loads(path.read_text()) | {"model_type": "test_target"}
                    )
                )
                native = Mock()
                register_native_target(
                    "test_target",
                    config.architecture,
                    source_prefix="draft",
                    validate=Mock(),
                    export=native,
                )
                # The default remains canonical even after native registration.
                export_draft(root / "draft", None, root / "standalone")
                self.assertEqual(read_draft(root / "standalone").config, config)
                native.assert_not_called()
                export_draft(
                    root / "draft",
                    None,
                    root / "merged",
                    export_mode="native",
                    target_model=target,
                )
                native.assert_called_once()
                self.assertFalse((root / "merged").exists())  # Only the adapter writes.

    def test_cli_modes_and_invalid_combinations(self):
        args = ["export", "--checkpoint", "source", "--output-dir", "out"]
        for mode in ("standalone", "native"):
            with patch("specforge.export.model.export_draft") as export:
                result = CliRunner().invoke(cli, args + ["--export-mode", mode])
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual(export.call_args.kwargs["export_mode"], mode)
        for options, error in (
            ({"export_mode": "native"}, "target-model"),
            ({"export_mode": "standalone", "copy_target": True}, "copy-target"),
            ({"export_mode": "native", "converter": "sglang"}, "omit --converter"),
        ):
            with self.assertRaisesRegex(ValueError, error):
                export_draft("missing", None, "out", **options)
        result = CliRunner().invoke(
            cli, args + ["--to", "hf", "--export-mode", "native"]
        )
        self.assertNotEqual(result.exit_code, 0)

    def test_explicit_mode_never_falls_back(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = mtp_config()
            save_draft(build_draft_model(config), root / "draft")
            write_target(root / "target")
            with self.assertRaisesRegex(ValueError, "no native target implementation"):
                export_draft(
                    root / "draft",
                    None,
                    root / "native",
                    export_mode="native",
                    target_model=root / "target",
                )
            with self.assertRaisesRegex(ValueError, "no SGLang converter"):
                export_draft(
                    root / "draft",
                    None,
                    root / "serving",
                    export_mode="standalone",
                    converter="sglang",
                )
            self.assertFalse((root / "native").exists())
            self.assertFalse((root / "serving").exists())
