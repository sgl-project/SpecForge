import unittest

import torch

from specforge.core.loss import LogSoftmaxLoss, _compute_loss

from .utils import norm_tensor


class TestLogSoftmaxLoss(unittest.TestCase):

    TTT_LENGTH = 7

    def _test_loss_and_gradient_calculation(self, B, T, V):
        if not torch.cuda.is_available():
            device = "cpu"
        else:
            device = "cuda"

        logits = norm_tensor((B, T, V), device, torch.float32)
        logits2 = logits.clone().detach().requires_grad_(True)
        target = norm_tensor((B, T, V), device, torch.float32)
        position_mask = torch.randint(0, 2, (B, T, 1), dtype=torch.bool, device=device)

        output1 = LogSoftmaxLoss.apply(logits, target, position_mask)
        output2 = _compute_loss(logits2, target, position_mask)
        torch.testing.assert_close(output1, output2, rtol=1e-4, atol=1e-4)

        output1.backward()
        output2.backward()
        torch.testing.assert_close(logits.grad, logits2.grad, rtol=1e-4, atol=1e-4)

    def test_loss(self):
        B = [1, 2, 4]
        T = [1024, 2048, 4096, 6000]
        V = [4096, 8192, 10000]
        for b in B:
            for t in T:
                for v in V:
                    self._test_loss_and_gradient_calculation(b, t, v)

    def test_ttt_loss_accumulation(self):
        if not torch.cuda.is_available():
            device = "cpu"
        else:
            device = "cuda"

        B, T, V = 1, 1024, 3200
        plosses = []
        plosses_compare = []
        logits_list = [
            norm_tensor((B, T, V), device, torch.float32)
            for _ in range(self.TTT_LENGTH)
        ]
        logits_list_copy = [
            logits.clone().detach().requires_grad_(True) for logits in logits_list
        ]
        for i in range(self.TTT_LENGTH):
            logits = logits_list[i]
            logits2 = logits_list_copy[i]
            target = norm_tensor((B, T, V), device, torch.float32)
            position_mask = torch.randint(
                0, 2, (B, T, 1), dtype=torch.bool, device=device
            )

            output1 = LogSoftmaxLoss.apply(logits, target, position_mask)
            output2 = _compute_loss(logits2, target, position_mask)
            torch.testing.assert_close(output1, output2, rtol=1e-4, atol=1e-4)
            plosses.append(output1)
            plosses_compare.append(output2)

        ploss_weight = [0.8**i for i in range(len(plosses))]
        ploss = (
            sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
            / self.TTT_LENGTH
        )
        ploss_compare = (
            sum([ploss_weight[i] * plosses_compare[i] for i in range(len(plosses))])
            / self.TTT_LENGTH
        )
        torch.testing.assert_close(ploss, ploss_compare, rtol=1e-4, atol=1e-4)
        ploss.backward()
        ploss_compare.backward()
        for i in range(self.TTT_LENGTH):
            torch.testing.assert_close(
                logits_list[i].grad, logits_list_copy[i].grad, rtol=1e-4, atol=1e-4
            )
