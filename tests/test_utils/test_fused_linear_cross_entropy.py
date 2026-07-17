import unittest
from importlib.metadata import PackageNotFoundError
from unittest import mock

import torch
import torch.nn.functional as F

from specforge.ops import fused_linear_cross_entropy as fused_ce


def _reference_forward(*, _input, weight, target, bias=None, **_kwargs):
    logits = _input @ weight.t()
    if bias is not None:
        logits = logits + bias
    grad_logits = torch.softmax(logits, dim=-1)
    grad_logits[torch.arange(target.numel()), target] -= 1
    grad_input = grad_logits @ weight
    loss = F.cross_entropy(logits, target, reduction="none")
    accuracy = (logits.argmax(dim=-1) == target).float()
    return loss, None, accuracy, None, grad_input, None, None


class TestFrozenLinearCrossEntropy(unittest.TestCase):
    def test_arbitrary_per_token_gradient_matches_torch(self):
        torch.manual_seed(7)
        hidden = torch.randn(9, 6, dtype=torch.double, requires_grad=True)
        reference_hidden = hidden.detach().clone().requires_grad_(True)
        weight = torch.randn(13, 6, dtype=torch.double)
        bias = torch.randn(13, dtype=torch.double)
        target = torch.randint(0, 13, (9,))
        token_weights = torch.tensor(
            [0.0, 0.3, 1.5, 0.0, 2.1, 0.7, 0.2, 1.0, 0.4], dtype=torch.double
        )

        with mock.patch.object(
            fused_ce, "_load_liger_forward", return_value=_reference_forward
        ):
            loss, accuracy = fused_ce.frozen_linear_cross_entropy(
                hidden, weight, target, bias
            )
            (loss * token_weights).sum().backward()

        reference_logits = reference_hidden @ weight.t() + bias
        reference_loss = F.cross_entropy(reference_logits, target, reduction="none")
        (reference_loss * token_weights).sum().backward()

        torch.testing.assert_close(loss, reference_loss)
        torch.testing.assert_close(
            accuracy, (reference_logits.argmax(dim=-1) == target).float()
        )
        torch.testing.assert_close(hidden.grad, reference_hidden.grad)

    def test_scales_low_precision_gradient_in_fp32_before_cast(self):
        saved_gradient = torch.tensor(
            [[0.0001001358, -0.00331], [0.00091, 0.02111]], dtype=torch.bfloat16
        )

        def saved_gradient_forward(**kwargs):
            count = kwargs["target"].numel()
            return (
                torch.ones(count),
                None,
                torch.ones(count),
                None,
                saved_gradient,
                None,
                None,
            )

        hidden = torch.zeros(2, 2, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.zeros(3, 2, dtype=torch.bfloat16)
        target = torch.tensor([0, 1])
        token_weights = torch.tensor([0.008334385, 1.33791])
        with mock.patch.object(
            fused_ce, "_load_liger_forward", return_value=saved_gradient_forward
        ):
            loss, _ = fused_ce.frozen_linear_cross_entropy(hidden, weight, target)
            (loss * token_weights).sum().backward()

        expected = (saved_gradient.float() * token_weights[:, None]).to(torch.bfloat16)
        early_cast = saved_gradient * token_weights.to(torch.bfloat16)[:, None]
        torch.testing.assert_close(hidden.grad, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(expected, early_cast))

    def test_rejects_invalid_inputs_and_trainable_head(self):
        hidden = torch.randn(2, 3, requires_grad=True)
        weight = torch.randn(5, 3, requires_grad=True)
        target = torch.tensor([1, 2])
        with self.assertRaisesRegex(ValueError, "frozen LM head"):
            fused_ce.frozen_linear_cross_entropy(hidden, weight, target)
        with self.assertRaisesRegex(ValueError, "torch.long"):
            fused_ce.frozen_linear_cross_entropy(
                hidden, weight.detach(), target.float()
            )

    def test_missing_dependency_and_unsupported_version_are_rejected(self):
        fused_ce._load_liger_forward.cache_clear()
        with mock.patch.object(
            fused_ce, "version", side_effect=PackageNotFoundError("liger-kernel")
        ):
            with self.assertRaisesRegex(RuntimeError, r"specforge\[liger\]"):
                fused_ce._load_liger_forward()

        fused_ce._load_liger_forward.cache_clear()
        with mock.patch.object(fused_ce, "version", return_value="0.7.0"):
            with self.assertRaisesRegex(RuntimeError, "requires liger-kernel==0.8.0"):
                fused_ce._load_liger_forward()
        fused_ce._load_liger_forward.cache_clear()


if __name__ == "__main__":
    unittest.main()
