"""Projection precision and the DSpark correction boundary must match inference."""

import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from specforge.modeling.precision import NativeDtypeLinear
from specforge.modeling.target.target_head import TargetHead


class LMHeadPrecisionTest(unittest.TestCase):
    def test_target_logits_project_before_upcasting(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            for dtype in (torch.bfloat16, torch.float16, torch.float32):
                with self.subTest(device=device, dtype=dtype):
                    torch.manual_seed(17)
                    head = TargetHead.__new__(TargetHead)
                    torch.nn.Module.__init__(head)
                    head.fc = NativeDtypeLinear(16, 11, bias=False,
                                               device=device, dtype=dtype)
                    head.freeze_weights()
                    saved = head.fc.weight.detach().clone()
                    hidden = torch.randn(3, 16, device=device, requires_grad=True)
                    logits = head(hidden)
                    expected = F.linear(hidden.to(dtype), saved).float()
                    self.assertEqual(logits.dtype, torch.float32)
                    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
                    torch.testing.assert_close(head.fc.weight, saved, rtol=0, atol=0)
                    actual_grad = torch.autograd.grad(logits.square().sum(), hidden)[0]
                    expected_grad = torch.autograd.grad(expected.square().sum(), hidden)[0]
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
                    if dtype != torch.float32:
                        # A float output does not mean the projection used FP32.
                        fp32_gemm = F.linear(hidden, saved.float())
                        self.assertFalse(torch.equal(logits, fp32_gemm))

    def test_dspark_ce_promotes_after_native_markov_addition(self):
        from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
        from specforge.modeling.draft.dspark import VanillaMarkovHead

        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                head = NativeDtypeLinear(4, 3, bias=False, device=device,
                                         dtype=torch.bfloat16).requires_grad_(False)
                markov = VanillaMarkovHead(vocab_size=3, markov_rank=1).to(
                    device=device, dtype=torch.bfloat16)
                with torch.no_grad():
                    head.weight.copy_(torch.tensor([[1., 0., 0., 0.],
                                                    [0., 1., 0., 0.],
                                                    [-1., 0., 0., 0.]], device=device))
                    markov.markov_w1.weight.fill_(1)
                    markov.markov_w2.weight.copy_(torch.tensor(
                        [[.001], [.01], [.1]], device=device))
                hidden = torch.tensor([[[[1., 0., 0., 0.], [1., 0., 0., 0.]]]],
                                      device=device, requires_grad=True)
                prev_ids = torch.zeros((1, 1, 2), device=device, dtype=torch.long)
                targets = torch.tensor([[[0, 1]]], device=device)
                observed = []

                def apply_logits(base, *, prev_token_ids, hidden_states):
                    observed.append(base.dtype)
                    return markov.apply_block_logits(
                        base, token_ids=prev_token_ids, hidden_states=hidden_states)

                model = SimpleNamespace(
                    _project_dspark_logits=head, lm_head=head,
                    draft_model=SimpleNamespace(apply_logits_head=apply_logits,
                                                predict_confidence=lambda *a, **k: None),
                    loss_type='dflash', dspark_objective='default',
                    dspark_l1_loss_alpha=0., dspark_confidence_head_alpha=0.,
                )
                terms = OnlineDSparkModel._dspark_objective_chunk_terms(
                    model, hidden, prev_ids, targets, torch.ones_like(targets).float(),
                    torch.ones_like(targets).bool(), None)
                base = F.linear(hidden.to(head.weight.dtype), head.weight)
                bias = markov.compute_step_bias(prev_ids, hidden)
                corrected = (base + bias).float()
                self.assertFalse(torch.equal(corrected, base.float() + bias.float()))
                expected = F.cross_entropy(corrected.flatten(0, 2), targets.flatten(),
                                           reduction='sum')
                self.assertEqual(observed, [torch.bfloat16])
                self.assertEqual(terms[0].dtype, torch.float32)
                torch.testing.assert_close(terms[0], expected, rtol=0, atol=0)
                params = (hidden, markov.markov_w2.weight)
                actual_grad = torch.autograd.grad(terms[0], params)
                expected_grad = torch.autograd.grad(expected, params)
                for actual, reference in zip(actual_grad, expected_grad):
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
                    self.assertTrue(torch.isfinite(actual).all())


if __name__ == '__main__':
    unittest.main()
