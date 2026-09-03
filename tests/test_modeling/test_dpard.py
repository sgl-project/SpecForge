"""D-PARD backward through the native DSpark model and provider."""

import tempfile
import unittest
from pathlib import Path

import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestDPardModel(unittest.TestCase):
    def test_b16_bf16_backward(self):
        from specforge.algorithms.model_providers import build_dspark_model
        from specforge.config import Config
        from tests.test_runtime import _fixtures as fx

        with tempfile.TemporaryDirectory(prefix="dpard_test_") as workdir:
            base, width = fx.build_dspark(
                workdir,
                hidden=64,
                draft_layers=3,
                block_size=16,
                num_anchors=3,
                attention_backend="sdpa",
            )
            cfg = Config.model_validate(
                {
                    "model": {
                        "target_model_path": str(Path(workdir) / "dspark_target"),
                        "mask_token_id": 0,
                    },
                    "data": {"hidden_states_path": workdir},
                    "training": {
                        "strategy": "dspark",
                        "loss_type": "dpard",
                        "dpard_alpha": 0.5,
                        "dspark_ce_loss_alpha": 0.0,
                        "dspark_l1_loss_alpha": 0.0,
                        "attention_backend": "sdpa",
                        "num_anchors": 3,
                        "objective_chunk_blocks": 1,
                    },
                }
            )
            model = build_dspark_model(cfg, base.draft_model, None, None, None).model
            self.assertEqual(model.loss_type, "dpard")
            ids = torch.randint(1, 20, (2, 24), device="cuda")
            mask = torch.ones_like(ids, dtype=torch.float32)
            mask[1, 17:] = 0
            loss, _, metrics = model(
                input_ids=ids,
                loss_mask=mask,
                hidden_states=torch.randn(
                    2, 24, width, device="cuda", dtype=torch.bfloat16
                ),
                target_last_hidden_states=torch.randn(
                    2, 24, 64, device="cuda", dtype=torch.bfloat16
                ),
            )
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            gradients = {
                name: p.grad
                for name, p in model.draft_model.named_parameters()
                if p.grad is not None
            }
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients.values()))
            for component in ["markov", "confidence", "layers"]:
                self.assertTrue(
                    any(
                        component in name and g.norm() > 0
                        for name, g in gradients.items()
                    ),
                    component,
                )
            credit, count = metrics["ratio_metrics"]["dpard_credit_position"]
            self.assertEqual(tuple(credit.shape), (16,))
            self.assertTrue(torch.isfinite(credit).all())
            self.assertTrue((credit >= 0).all())
            self.assertTrue((count >= 0).all())
            num, den = metrics["ratio_metrics"]["dpard_loss"]
            self.assertGreater(float(den), 0.0)
            _, confidence_den = metrics["ratio_metrics"]["confidence_loss"]
            torch.testing.assert_close(den, confidence_den)
            self.assertTrue(torch.isfinite(num))
