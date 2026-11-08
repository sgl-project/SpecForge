"""A trained legacy token table survives migration and remains trainable."""

import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import load_file

from specforge.loading import initialize_draft, load_draft
from specforge.modeling.draft.layers.inputs import LayerInputs
from tests.test_checkpoint.test_loading import write_target
from tests.test_modeling.test_draft_gqa import layout_for

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/legacy_peagle"


class PEagleMigrationTest(unittest.TestCase):
    def test_trained_embedding_outputs_padding_and_target_copy(self):
        reference = load_file(FIXTURE / "reference.safetensors")
        trained = load_file(FIXTURE / "model.safetensors")["embed_tokens.weight"]
        with TemporaryDirectory() as tmp:
            target = Path(tmp) / "target"
            target_weights = write_target(target)["model.embed_tokens.weight"]
            source = Path(tmp) / "draft"
            shutil.copytree(FIXTURE, source)
            torch.save({"peagle_mask_token_id": 63}, source / "training_state.pt")
            model = load_draft(source, target=target)
            self.assertTrue(model.embed_tokens.weight.requires_grad)
            self.assertEqual(model.embed_tokens.padding_idx, 0)
            torch.testing.assert_close(
                model.embed_tokens.weight, trained, rtol=0, atol=0
            )
            self.assertFalse(torch.equal(trained, target_weights))
            layout = layout_for(reference["target_features"])
            entries = [
                LayerInputs(attention_layout=layout, position_ids=layout.key_positions)
            ]
            out = model(reference["input_ids"], reference["target_features"], entries)
            for name in ("hidden_states", "logits"):
                torch.testing.assert_close(
                    getattr(out, name), reference[name], rtol=3e-5, atol=2e-6
                )
            out.logits.square().sum().backward()
            grad = model.embed_tokens.weight.grad
            self.assertEqual(grad[0].count_nonzero().item(), 0)
            self.assertGreater(grad[1].count_nonzero().item(), 0)
            fresh = initialize_draft(
                model.config.draft,
                target=target,
                from_target={"embed_tokens": "model.embed_tokens"},
            )
            torch.testing.assert_close(
                fresh.embed_tokens.weight, target_weights, rtol=0, atol=0
            )
            self.assertTrue(fresh.embed_tokens.weight.requires_grad)
