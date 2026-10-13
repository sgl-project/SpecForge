# coding=utf-8
"""DSpark objective normalization: ``round`` (historical) versus ``step``."""

import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "the DSpark fixture builds on CUDA")
class TestDSparkLossNormalization(unittest.TestCase):
    def test_round_and_step_agree_per_micro_batch_but_only_step_emits_terms(self):
        from tests.test_runtime import _fixtures as fx

        hidden, seq, batch = 64, 12, 2
        workdir = tempfile.mkdtemp(prefix="dspark_norm_")
        model, width = fx.build_dspark(
            workdir,
            hidden=hidden,
            block_size=4,
            num_anchors=2,
            attention_backend="sdpa",
        )
        self.assertEqual(model.loss_normalization, "round")
        generator = torch.Generator().manual_seed(0)
        input_ids = torch.randint(0, fx.V, (batch, seq), generator=generator).cuda()
        loss_mask = torch.ones(batch, seq, dtype=torch.long)
        loss_mask[0, :3] = 0  # different supervised-token weights per sample
        loss_mask = loss_mask.cuda()
        hidden_states = (
            torch.randn(batch, seq, width, generator=generator)
            .to(torch.bfloat16)
            .cuda()
        )
        target_last = (
            torch.randn(batch, seq, hidden, generator=generator)
            .to(torch.bfloat16)
            .cuda()
        )

        def run():
            torch.manual_seed(1)  # same anchor sample for both normalizations
            with torch.no_grad():
                return model(
                    input_ids=input_ids,
                    hidden_states=hidden_states,
                    loss_mask=loss_mask,
                    target_last_hidden_states=target_last,
                    collect_detailed_metrics=False,
                )

        loss_round, acc_round, metrics_round = run()
        self.assertNotIn("loss_terms", metrics_round)
        model.loss_normalization = "step"
        loss_step, acc_step, metrics_step = run()
        numerator, denominator = metrics_step["loss_terms"]
        self.assertEqual(numerator.shape, ())
        self.assertEqual(denominator.shape, ())
        self.assertFalse(denominator.requires_grad)
        torch.testing.assert_close(loss_step, numerator / denominator)
        # A single rank with one micro-batch per step is where both objectives
        # coincide: the per-round ratio equals the global ratio.
        torch.testing.assert_close(loss_round, loss_step)
        torch.testing.assert_close(acc_round, acc_step)
        self.assertGreater(float(denominator), 0.0)


class DSparkLossNormalizationResolutionTest(unittest.TestCase):
    def test_follows_cost_aware_scheduling_and_is_recorded(self):
        from types import SimpleNamespace

        from specforge.algorithms.dspark.providers import resume_contract
        from specforge.algorithms.model_providers import (
            resolve_dspark_loss_normalization,
        )

        self.assertEqual(resolve_dspark_loss_normalization(SimpleNamespace()), "round")
        self.assertEqual(
            resolve_dspark_loss_normalization(
                SimpleNamespace(cost_aware_scheduling=False)
            ),
            "round",
        )
        self.assertEqual(
            resolve_dspark_loss_normalization(
                SimpleNamespace(cost_aware_scheduling=True)
            ),
            "step",
        )
        draft = SimpleNamespace(
            config=SimpleNamespace(num_hidden_layers=5), target_layer_ids=[1, 9]
        )
        model = SimpleNamespace(
            block_size=16,
            mask_token_id=7,
            attention_backend="flex_attention",
            num_anchors=512,
            loss_decay_gamma=4.0,
            dspark_ce_loss_alpha=0.1,
            dspark_l1_loss_alpha=0.9,
            dspark_confidence_head_alpha=1.0,
            loss_normalization="step",
        )
        contract = resume_contract(None, draft, model)
        self.assertEqual(contract["dspark_loss_normalization"], "step")


if __name__ == "__main__":
    unittest.main()
