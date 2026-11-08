"""Historical MTP weights/outputs survive removal of its model implementation."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import load_file, save_file

from specforge.algorithms.objectives.inputs import sequence_inputs
from specforge.loading import load_draft, read_draft

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/legacy_mtp"


class MTPCodecParityTest(unittest.TestCase):
    def test_saved_legacy_outputs_and_gradients(self):
        reference = load_file(FIXTURE / "reference.safetensors")
        weights = load_file(FIXTURE / "model.safetensors")
        with TemporaryDirectory() as temp:
            target = Path(temp)
            (target / "config.json").write_text(
                json.dumps({"hidden_size": 32, "vocab_size": 64})
            )
            save_file(
                {"model.embed_tokens.weight": weights["embed_tokens.weight"]},
                target / "model.safetensors",
            )
            artifact = read_draft(FIXTURE, target=target)
            for backend in ("eager", "sdpa"):
                model = load_draft(FIXTURE, target=target, backend=backend).train()
                self.assertEqual(
                    model.config.draft.rope_parameters.rope_theta, 10_000_000.0
                )
                features = reference["target_features"].clone().requires_grad_()
                positions = reference["position_ids"]
                entries = sequence_inputs(
                    artifact.config, positions, torch.ones(2, 4, 4, dtype=torch.bool)
                )
                out = model(reference["input_ids"], features, entries)
                out.logits.square().mean().backward()
                for name in ("logits", "hidden_states"):
                    torch.testing.assert_close(
                        getattr(out, name), reference[name], rtol=3e-5, atol=2e-6
                    )
                torch.testing.assert_close(
                    features.grad, reference["feature_grad"], rtol=3e-4, atol=2e-6
                )
                expected_grads = {
                    name: (
                        reference["grad." + artifact.weights[name].key]
                        if param.requires_grad
                        else None
                    )
                    for name, param in model.named_parameters()
                }
                torch.testing.assert_close(
                    {name: param.grad for name, param in model.named_parameters()},
                    expected_grads,
                    rtol=3e-4,
                    atol=2e-6,
                )


if __name__ == "__main__":
    unittest.main()
