import os
import unittest

import torch
import torch.nn.functional as F

from specforge.ops.fused_linear_cross_entropy import frozen_linear_cross_entropy


@unittest.skipUnless(
    os.environ.get("SPECFORGE_RUN_LIGER_TESTS") == "1" and torch.cuda.is_available(),
    "set SPECFORGE_RUN_LIGER_TESTS=1 on an exclusive CUDA GPU",
)
class TestLigerFusedLinearCrossEntropyCuda(unittest.TestCase):
    def test_bf16_loss_accuracy_and_weighted_hidden_gradient(self):
        for use_bias in (False, True):
            with self.subTest(use_bias=use_bias):
                torch.manual_seed(123)
                token_count, hidden_size, vocab_size = 67, 128, 1024
                hidden = torch.randn(
                    token_count,
                    hidden_size,
                    device="cuda",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                weight = torch.randn(
                    vocab_size, hidden_size, device="cuda", dtype=torch.bfloat16
                )
                bias = (
                    torch.randn(vocab_size, device="cuda", dtype=torch.bfloat16)
                    if use_bias
                    else None
                )
                target = torch.randint(vocab_size, (token_count,), device="cuda")
                token_weights = torch.linspace(0.0, 1.75, token_count, device="cuda")

                loss, accuracy = frozen_linear_cross_entropy(
                    hidden, weight, target, bias
                )
                (loss * token_weights).sum().backward()
                fused_grad = hidden.grad.float().clone()

                reference_hidden = hidden.detach().clone().requires_grad_(True)
                # Liger applies bias after the BF16 matmul rather than through a
                # fused linear epilogue, so preserve that rounding boundary.
                reference_logits = reference_hidden @ weight.t()
                if bias is not None:
                    reference_logits = reference_logits + bias
                reference_loss = F.cross_entropy(
                    reference_logits, target, reduction="none"
                )
                (reference_loss * token_weights).sum().backward()
                reference_grad = reference_hidden.grad.float()

                self.assertLess(
                    (loss - reference_loss.float()).abs().mean().item(), 0.1
                )
                self.assertTrue(
                    torch.equal(
                        accuracy,
                        (reference_logits.argmax(dim=-1) == target).float(),
                    )
                )
                cosine = F.cosine_similarity(
                    fused_grad.flatten(), reference_grad.flatten(), dim=0
                )
                relative_l2_error = (
                    fused_grad - reference_grad
                ).norm() / reference_grad.norm()
                self.assertGreater(cosine.item(), 0.999)
                self.assertLess(relative_l2_error.item(), 0.01)
                self.assertTrue(torch.isfinite(fused_grad).all())


if __name__ == "__main__":
    unittest.main()
