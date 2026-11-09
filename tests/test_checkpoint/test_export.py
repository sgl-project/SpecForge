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
                    env={**os.environ, "TORCHDYNAMO_DISABLE": "1"},
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
        from tests.test_checkpoint.legacy_weights import legacy_state
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
                state = {
                    k: v.to(dtype) if v.is_floating_point() else v
                    for k, v in legacy_state("DFlashDraftModel").items()
                }
                (root / "legacy.json").write_text(json.dumps(payload))
                torch.save({"draft_state_dict": state}, root / "training_state.pt")
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
                    restored.fusion.fc.weight, state["fc.weight"], rtol=0, atol=0
                )
