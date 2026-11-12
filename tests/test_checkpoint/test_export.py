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
                export_to_hf(str(root / "source"), None, str(root / "hf"))
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
                export_to_hf(str(root / "source"), None, str(root / "hf"))
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
            export_to_hf(str(root / "source"), None, str(root / "hf"))
            restored = load_draft(root / "hf")
            self.assertIs(restored.lm_head.weight, restored.embed_tokens.weight)
            with self.assertRaises(FileExistsError):
                export_to_hf(str(root / "source"), None, str(root / "hf"))
            state.pop("fusion.fc.weight")
            torch.save(
                {"draft_state_dict": state}, root / "source" / "training_state.pt"
            )
            with self.assertRaisesRegex(ValueError, "missing draft tensors"):
                export_to_hf(str(root / "source"), None, str(root / "bad"))
            self.assertFalse((root / "bad").exists())

    def test_legacy_import_requires_target_and_preserves_precision(self):
        from specforge.modeling.auto import AutoDraftModel
        from specforge.modeling.draft.config import AutoDraftConfig
        from tests.test_checkpoint.test_loading import write_target
        from tests.test_config.test_draft_legacy_dflash import legacy_payload

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), TemporaryDirectory() as tmp:
                root = Path(tmp)
                payload = legacy_payload() | dict(
                    hidden_size=32,
                    intermediate_size=64,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=8,
                    num_hidden_layers=2,
                    vocab_size=64,
                    draft_vocab_size=64,
                    num_target_layers=128,
                    layer_types=["full_attention"] * 2,
                )
                payload["dflash_config"].update(
                    mask_token_id=63, target_layer_ids=[1, 3]
                )
                old = AutoDraftModel.from_config(AutoDraftConfig.from_dict(payload)).to(
                    dtype
                )
                (root / "legacy.json").write_text(json.dumps(payload))
                torch.save(
                    {"draft_state_dict": old.state_dict()}, root / "training_state.pt"
                )
                write_target(root / "target")
                with self.assertRaisesRegex(ValueError, "target configuration"):
                    export_to_hf(
                        str(root), str(root / "legacy.json"), str(root / "bad")
                    )
                export_to_hf(
                    str(root),
                    str(root / "legacy.json"),
                    str(root / "hf"),
                    target_model=str(root / "target"),
                )
                restored = load_draft(root / "hf", target=root / "target")
                self.assertEqual(restored.fusion.fc.weight.dtype, dtype)
                torch.testing.assert_close(
                    restored.fusion.fc.weight, old.fc.weight, rtol=0, atol=0
                )

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
            export_to_hf(str(root / "source"), None, str(root / "hf"))
            config = json.loads((root / "hf" / "config.json").read_text())
            self.assertIsNone(config.get("dtype"))  # not the stale "bfloat16"

    def test_legacy_bound_copy_must_match_the_target(self):
        from safetensors.torch import save_file

        from tests.test_checkpoint.test_loading import write_target
        from tests.test_config.test_draft_legacy_mtp import mtp_payload

        for offset, accepted in ((0, True), (1, False)):
            with self.subTest(accepted=accepted), TemporaryDirectory() as tmp:
                root = Path(tmp)
                target, source = root / "target", root / "source"
                weights = write_target(target)
                source.mkdir()
                payload = mtp_payload() | dict(hidden_size=32, vocab_size=64)
                (source / "config.json").write_text(json.dumps(payload))
                torch.save({}, source / "training_state.pt")
                head = {"mtp.lm_head.weight": weights["lm_head.weight"]}
                save_file(head, source / "model.safetensors")
                model = build_draft_model(read_draft(source, target=target).config)
                state = {
                    "mtp."
                    + key.replace("backbone.final_norm.", "norm.")
                    .removeprefix("backbone.")
                    .removeprefix("fusion."): value
                    for key, value in model.state_dict().items()
                }
                state["mtp.lm_head.weight"] = weights["lm_head.weight"] + offset
                save_file(state, source / "model.safetensors")
                if accepted:
                    export_to_hf(
                        str(source), None, str(root / "hf"), target_model=str(target)
                    )
                else:
                    with self.assertRaisesRegex(
                        ValueError, "differs from bound target"
                    ):
                        export_to_hf(
                            str(source),
                            None,
                            str(root / "hf"),
                            target_model=str(target),
                        )
                    self.assertFalse((root / "hf").exists())

    def test_draft_defects_fail_before_target_tables_are_read(self):
        from safetensors.torch import save_file

        from tests.test_config.test_draft_legacy_eagle3 import eagle_payload
        from tests.test_modeling.test_canonical_eagle3 import install_mapping

        payload = eagle_payload() | dict(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
            draft_vocab_size=32,
        )
        payload.pop("rope_scaling", None)
        renames = (
            ("backbone.layers.0.", "midlayer."),
            ("backbone.final_norm.", "norm."),
            ("fusion.", ""),
            ("vocabulary.", ""),
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            target, source = root / "target", root / "source"
            target.mkdir()
            source.mkdir()
            # config only: reading a target table would fail with FileNotFoundError
            (target / "config.json").write_text(
                json.dumps(
                    {"hidden_size": 32, "vocab_size": 64, "num_hidden_layers": 128}
                )
            )
            (source / "config.json").write_text(json.dumps(payload))
            save_file({"x": torch.zeros(1)}, source / "model.safetensors")
            model = build_draft_model(read_draft(source, target=target).config)
            install_mapping(model)
            state = {}
            for key, value in model.state_dict().items():
                for new, old in renames:
                    if key.startswith(new):
                        key = old + key.removeprefix(new)
                        break
                state[key] = value
            state["embed_tokens.weight"] = torch.randn(64, 32)  # a bound copy
            del state["fc.weight"]
            save_file(state, source / "model.safetensors")
            with self.assertRaisesRegex(ValueError, "missing draft tensors"):
                export_to_hf(
                    str(source), None, str(root / "hf"), target_model=str(target)
                )
