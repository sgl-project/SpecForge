"""Export routing, atomic publication and source-bundling boundaries."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from click.testing import CliRunner

from specforge.checkpoint import save_draft
from specforge.checkpoint.draft_dir import write_artifact
from specforge.checkpoint.formats.hf import bundle_code
from specforge.cli import cli
from specforge.export.canonical import read_export
from specforge.modeling import build_draft_model
from specforge.modeling.draft.pretrained import DraftPreTrainedModel
from specforge.training.checkpoint import CheckpointManager
from tests.test_config.draft_fixtures import draft_config
from tests.test_modeling.test_canonical_dspine import model_config as dspine_config
from tests.test_modeling.test_canonical_eagle3 import model_config as eagle_config


class ExportReviewTest(unittest.TestCase):
    def test_export_delegates_to_components_without_materializing_target(self):
        class Component(torch.nn.Module):
            def validate_export_state(self, state):
                torch.testing.assert_close(state["weight"], torch.ones(1))
                raise ValueError("component rejected state")

        config = draft_config()
        model = build_draft_model(config, device="meta")
        model.add_module("custom", Component())
        with self.assertRaisesRegex(ValueError, "component rejected"):
            model.validate_export_state({"custom.weight": torch.ones(1)})

        validate = DraftPreTrainedModel.validate_export_state

        def check_meta(preview, state):
            self.assertTrue(all(p.is_meta for p in preview.parameters()))
            self.assertNotIn("lm_head.weight", state)
            validate(preview, state)

        with TemporaryDirectory() as temp:
            save_draft(build_draft_model(config), temp)
            with patch.object(
                DraftPreTrainedModel, "validate_export_state", check_meta
            ):
                read_export(temp)

    def test_export_rejects_unready_model_and_component_state(self):
        for config, error in (
            (dspine_config(), "transfer_ready"),
            (eagle_config(), "t2d/d2t"),
        ):
            with (
                self.subTest(architecture=config.architecture),
                TemporaryDirectory() as temp,
            ):
                save_draft(build_draft_model(config), temp)
                with self.assertRaisesRegex(ValueError, error):
                    read_export(temp)

    def test_cli_defaults_to_canonical_and_rejects_legacy_options_early(self):
        args = ["export", "--checkpoint", "missing", "--output-dir", "out"]
        with patch("specforge.export.to_hf.export_to_hf") as export:
            result = CliRunner().invoke(cli, args)
            self.assertEqual(result.exit_code, 0, result.output)
            export.assert_called_once()
            export.reset_mock()
            result = CliRunner().invoke(cli, args + ["--embedding-source", "target"])
            self.assertEqual(result.exit_code, 2)
            self.assertIn("--target-model", result.output)
            result = CliRunner().invoke(cli, args + ["--vocab-mapping", "map.pt"])
            self.assertEqual(result.exit_code, 2)
            self.assertIn("--vocab-mapping belongs to legacy --to", result.output)
            export.assert_not_called()
        with patch("specforge.export.to_sglang.export_to_sglang") as export:
            result = CliRunner().invoke(cli, args + ["--to", "sglang"])
            self.assertEqual(result.exit_code, 2)
            self.assertIn("--draft-config", result.output)
            export.assert_not_called()

    def test_run_root_uses_complete_steps_and_rejects_multiple_runs(self):
        with TemporaryDirectory() as temp:
            manager = CheckpointManager(temp, "run")
            with patch.object(manager, "_point"):
                manager.save({}, 1)
                latest = manager.save({}, 2)
            with patch(
                "specforge.export.canonical.read_draft",
                side_effect=RuntimeError("selected"),
            ) as read:
                with self.assertRaisesRegex(RuntimeError, "selected"):
                    read_export(temp)
                self.assertEqual(read.call_args.args[0], Path(latest))
            other = Path(temp) / "other-step3"
            other.mkdir()
            torch.save({}, other / "training_state.pt")
            with self.assertRaisesRegex(ValueError, "multiple|several"):
                read_export(temp)

    def test_writer_preserves_shards_and_does_not_publish_on_failure(self):
        from safetensors.torch import load_file

        with TemporaryDirectory() as temp:
            output = Path(temp) / "model"
            state = {name: torch.randn(32, dtype=torch.bfloat16) for name in ("a", "b")}
            write_artifact(output, {}, state, max_shard_size=64)
            index = json.loads((output / "model.safetensors.index.json").read_text())
            for name, shard in index["weight_map"].items():
                torch.testing.assert_close(
                    load_file(output / shard)[name], state[name], rtol=0, atol=0
                )
            with self.assertRaises(FileExistsError):
                write_artifact(output, {}, state)
            failed = Path(temp) / "failed"
            with self.assertRaisesRegex(RuntimeError, "bundling failed"):
                write_artifact(
                    failed, {}, state, write_code=lambda _: self.fail_bundle()
                )
            self.assertFalse(failed.exists())

    @staticmethod
    def fail_bundle():
        raise RuntimeError("bundling failed")

    def test_bundler_rejects_external_modules_and_ambiguous_sources(self):
        cases = [
            (
                "extension.modeling.draft.models.dflash",
                "class Model: pass",
                "only bundles",
            ),
            (
                "specforge.a_b",
                "from specforge import X\nclass Model: pass",
                "explicit module",
            ),
            ("specforge.a_b", "class Other: pass", "not defined"),
            (
                "specforge.a_b",
                "from specforge.a.b import X\nclass Model: pass",
                "collision",
            ),
        ]
        for module, source, error in cases:
            with self.subTest(module=module, error=error), TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "modeling/draft").mkdir(parents=True)
                (root / "modeling/draft/hf_config.py").write_text(
                    "class SpecForgeDraftConfig: pass"
                )
                (root / "a_b.py").write_text(source)
                (root / "a").mkdir()
                (root / "a/b.py").write_text("X = 1")
                output = root / "output"
                output.mkdir()
                cls = SimpleNamespace(__module__=module, __name__="Model")
                with patch(
                    "specforge.checkpoint.formats.hf.__file__",
                    str(root / "checkpoint/formats/hf.py"),
                ):
                    with self.assertRaisesRegex(ValueError, error):
                        bundle_code(output, cls)
