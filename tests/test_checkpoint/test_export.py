"""Canonical publication preserves state and runs without installed SpecForge."""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from specforge.checkpoint import save_draft
from specforge.export.to_hf import export_to_hf
from specforge.loading import load_draft, read_draft
from specforge.modeling import build_draft_model
from tests.test_config.draft_fixtures import draft_config
from tests.test_modeling.test_canonical_dflash import inputs_for


class CanonicalExportTest(unittest.TestCase):
    def test_run_paths_agree_for_loading_export_and_initialization(self):
        from specforge.application.composition import resolve_run
        from specforge.export.canonical import read_export
        from specforge.training.canonical import resolve_config
        from specforge.training.checkpoint import CheckpointManager
        from tests.test_runtime.test_canonical_training import (
            run_config,
            target_directory,
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "run"
            step = run / "draft-step2"
            config = draft_config(target_bindings=(), target_logit_transform=None)
            model = build_draft_model(config)
            save_draft(model, step)
            torch.save(
                {"draft_state_dict": model.state_dict()}, step / "training_state.pt"
            )
            target = target_directory(root)
            for linked in (False, True):
                if linked:
                    (run / "draft-latest").symlink_to(step, target_is_directory=True)
                for path in (
                    step,
                    step / "training_state.pt",
                    run,
                    "file://" + str(run),
                ):
                    with self.subTest(linked=linked, path=path):
                        self.assertEqual(
                            Path(CheckpointManager.resolve_resume_dir(path)).resolve(),
                            step.resolve(),
                        )
                        self.assertEqual(read_draft(path).config, config)
                        self.assertEqual(read_export(path)[0].config, config)
                        cfg = run_config(
                            target,
                            config,
                            initialization={"checkpoint": str(path)},
                        ).model_copy(update={"draft": None})
                        self.assertEqual(resolve_config(cfg), config)
                        resolve_run(cfg)

    def test_owned_and_bound_export(self):
        for bound in (False, True):
            with self.subTest(bound=bound), TemporaryDirectory() as tmp:
                root = Path(tmp)
                cfg = (
                    draft_config()
                    if bound
                    else draft_config(target_bindings=(), target_logit_transform=None)
                )
                model = build_draft_model(cfg).eval()
                torch.save(model.state_dict(), root / "expected_state.pt")
                save_draft(model, root / "source")
                export_to_hf(str(root / "source"), str(root / "hf"))
                self.assertEqual(read_draft(root / "hf").config, cfg)
                self.assertFalse((root / "hf" / "artifact.json").exists())
                self.assertEqual(model.config.to_dict().get("auto_map"), None)
                if not bound:
                    restored = load_draft(root / "hf")
                    args = inputs_for(cfg)
                    torch.testing.assert_close(
                        restored(*args).logits, model(*args).logits, rtol=0, atol=0
                    )
                    layout = args[2][0].attention_layout
                    torch.save(
                        dict(
                            ids=args[0],
                            features=args[1].detach(),
                            q=layout.query_positions,
                            k=layout.key_positions,
                            visibility=layout.visibility,
                            logits=model(*args).logits.detach(),
                        ),
                        root / "expected.pt",
                    )
                # Isolated Python excludes cwd/PYTHONPATH, and an import blocker
                # catches editable installations as well. Only bundled code may run.
                script = (
                    Path(__file__).resolve().parent
                    / "helpers"
                    / "check_standalone_export.py"
                )
                result = subprocess.run(
                    [sys.executable, "-I", str(script), str(root / "hf")],
                    capture_output=True,
                    text=True,
                    env={
                        **os.environ,
                        "TORCHDYNAMO_DISABLE": "1",
                        "HF_MODULES_CACHE": str(root / "modules"),
                    },
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_all_architectures_reload_and_resave_without_specforge(self):
        from tests.test_algorithms.test_objectives import configurations
        from tests.test_modeling.test_canonical_eagle3 import install_mapping

        for cfg in configurations():
            with (
                self.subTest(architecture=cfg.architecture),
                TemporaryDirectory() as tmp,
            ):
                root = Path(tmp)
                model = build_draft_model(cfg).eval()
                for name in cfg.target_bindings:
                    module = model.get_submodule(name)
                    module.load_target_weight(torch.randn_like(module.weight))
                if hasattr(model, "vocabulary"):
                    install_mapping(model)
                model.finalize_initialization(loaded_keys=frozenset())
                torch.save(model.state_dict(), root / "expected_state.pt")
                save_draft(model, root / "source", max_shard_size="5KB")
                export_to_hf(str(root / "source"), str(root / "hf"))
                script = (
                    Path(__file__).parent / "helpers" / "check_standalone_export.py"
                )
                result = subprocess.run(
                    [sys.executable, "-I", str(script), str(root / "hf")],
                    capture_output=True,
                    text=True,
                    env={**os.environ, "HF_MODULES_CACHE": str(root / "modules")},
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_training_state_and_strict_publication(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = draft_config(
                target_bindings=(),
                target_logit_transform=None,
                tie_word_embeddings=True,
            )
            model = build_draft_model(cfg)
            save_draft(model, root / "source")
            state = dict(model.state_dict())
            del state["lm_head.weight"]
            torch.save(
                {"draft_state_dict": state}, root / "source" / "training_state.pt"
            )
            (root / "source" / "model.safetensors").unlink()
            export_to_hf(str(root / "source"), str(root / "hf"))
            restored = load_draft(root / "hf")
            self.assertIs(restored.lm_head.weight, restored.embed_tokens.weight)
            with self.assertRaises(FileExistsError):
                export_to_hf(str(root / "source"), str(root / "hf"))
            state.pop("fusion.fc.weight")
            torch.save(
                {"draft_state_dict": state}, root / "source" / "training_state.pt"
            )
            with self.assertRaisesRegex(ValueError, "missing draft tensors"):
                export_to_hf(str(root / "source"), str(root / "bad"))
            self.assertFalse((root / "bad").exists())

    def test_mixed_precision_export_does_not_inherit_a_source_dtype(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = draft_config(target_bindings=(), target_logit_transform=None)
            model = build_draft_model(cfg).to(torch.bfloat16)
            model.backbone.final_norm.float()
            save_draft(model, root / "source")
            payload = json.loads((root / "source" / "config.json").read_text())
            (root / "source" / "config.json").write_text(
                json.dumps(payload | {"dtype": "bfloat16"})
            )
            export_to_hf(str(root / "source"), str(root / "hf"))
            config = json.loads((root / "hf" / "config.json").read_text())
            self.assertIsNone(config.get("dtype"))  # not the stale "bfloat16"

    def test_draft_defects_fail_before_target_tables_are_read(self):
        from safetensors.torch import load_file, save_file

        from tests.test_modeling.test_canonical_eagle3 import (
            install_mapping,
            model_config,
        )

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, source = root / "target", root / "source"
            target.mkdir()
            (target / "config.json").write_text(
                json.dumps(
                    {"hidden_size": 32, "vocab_size": 64, "num_hidden_layers": 128}
                )
            )
            model = build_draft_model(model_config(bound=True, target_width=32))
            install_mapping(model)
            save_draft(model, source)
            state = load_file(source / "model.safetensors")
            del state["fusion.fc.weight"]
            save_file(state, source / "model.safetensors")
            with self.assertRaisesRegex(ValueError, "missing draft tensors"):
                export_to_hf(str(source), str(root / "hf"), target_model=str(target))
