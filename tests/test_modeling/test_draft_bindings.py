"""Target modules stay frozen, omit target tensors and preserve muP numerics."""

import unittest

import torch
import torch.nn.functional as F
from torch import nn

from specforge.config.draft.logits import MuPLogitTransformSpec, NoLogitTransformSpec
from specforge.modeling.draft.bindings import BoundEmbedding, BoundLMHead
from tests.test_modeling.test_draft_pretrained import DFlashDraftModel, model_config


def owner():
    module = nn.Module()
    module.embed_tokens = BoundEmbedding(16, 8)
    module.lm_head = BoundLMHead(NoLogitTransformSpec(), vocab_size=16, hidden_size=8)
    module.proj = nn.Linear(8, 8)
    return module


class BindingTest(unittest.TestCase):
    def _parity(self, device, dtype):
        for config in (NoLogitTransformSpec(), MuPLogitTransformSpec(multiplier=3)):
            with self.subTest(device=device, dtype=dtype, config=config):
                source = torch.randn(16, 8, device=device, requires_grad=True)
                original = source.detach().clone()
                embedding = BoundEmbedding(16, 8, device=device, dtype=dtype)
                head = BoundLMHead(
                    config, vocab_size=16, hidden_size=8, device=device, dtype=dtype
                )
                parameter = head.weight
                embedding.load_target_weight(source)
                head.load_target_weight(source)
                self.assertIs(head.weight, parameter)
                self.assertNotEqual(head.weight.data_ptr(), source.data_ptr())
                expected_weight = source.detach().to(dtype).clone()
                if type(config) is MuPLogitTransformSpec:
                    expected_weight.div_(config.multiplier)
                torch.testing.assert_close(head.weight, expected_weight, rtol=0, atol=0)
                ids = torch.tensor([[1, 4, 7]], device=device)
                torch.testing.assert_close(
                    embedding(ids),
                    F.embedding(ids, source.detach().to(dtype)),
                    rtol=0,
                    atol=0,
                )
                hidden = torch.randn(
                    2, 4, 8, device=device, dtype=dtype, requires_grad=True
                )
                other = hidden.detach().clone().requires_grad_()
                logits, expected = head(hidden), F.linear(other, expected_weight)
                torch.testing.assert_close(logits, expected, rtol=0, atol=0)
                upstream = torch.randn_like(logits)
                logits.backward(upstream)
                expected.backward(upstream)
                torch.testing.assert_close(hidden.grad, other.grad, rtol=0, atol=0)
                self.assertFalse(head.weight.requires_grad)
                self.assertIsNone(head.weight.grad)
                self.assertIsNone(source.grad)
                torch.testing.assert_close(source, original, rtol=0, atol=0)
                head.load_target_weight(source)
                torch.testing.assert_close(head.weight, expected_weight, rtol=0, atol=0)

    def test_cpu_binding_and_logits(self):
        for dtype in (torch.float32, torch.bfloat16):
            self._parity("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_binding_and_logits(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._parity("cuda", dtype)

    def test_draft_state_omits_bound_weights_and_strictly_restores_trainable_weights(
        self,
    ):
        module = owner()
        for bound in (module.embed_tokens, module.lm_head):
            bound.load_target_weight(torch.randn(16, 8))
        self.assertIn("lm_head.weight", dict(module.named_parameters()))
        state = module.state_dict()
        self.assertEqual(set(state), {"proj.weight", "proj.bias"})
        restored = owner()
        restored.load_state_dict(state, strict=True)
        torch.testing.assert_close(restored.proj.weight, module.proj.weight)
        with self.assertRaisesRegex(RuntimeError, "not bound"):
            restored.lm_head(torch.randn(1, 8))
        with self.assertRaisesRegex(RuntimeError, "Unexpected key"):
            restored.load_state_dict(
                state | {"embed_tokens.weight": torch.randn(16, 8)}, strict=True
            )
        # Draft state restoration does not replace an already bound target.
        previous = module.lm_head.weight.detach().clone()
        module.load_state_dict(state, strict=True)
        torch.testing.assert_close(module.lm_head.weight, previous, rtol=0, atol=0)

    def test_meta_materialization_and_hf_initialization_never_synthesize_target_weights(
        self,
    ):
        head = BoundLMHead(
            NoLogitTransformSpec(), vocab_size=16, hidden_size=8, device="meta"
        )
        with self.assertRaisesRegex(ValueError, "materialize"):
            head.load_target_weight(torch.ones(16, 8))
        head.to_empty(device="cpu")
        initialize = DFlashDraftModel(
            model_config(initializer_range=0)
        )._initialize_weights
        head.apply(initialize)
        with self.assertRaisesRegex(RuntimeError, "not bound"):
            head(torch.randn(1, 8))
        head.load_target_weight(torch.ones(16, 8))
        head.reset_draft_parameters()
        torch.testing.assert_close(
            head.weight, torch.ones_like(head.weight), rtol=0, atol=0
        )

    def test_invalid_binding_does_not_change_installed_weight(self):
        embedding = BoundEmbedding(16, 8)
        with self.assertRaisesRegex(RuntimeError, "not bound"):
            embedding(torch.tensor([1]))
        embedding.load_target_weight(torch.ones(16, 8))
        for source, error in (
            (torch.ones(8, 8), ValueError),
            (torch.ones(16, 8, dtype=torch.int64), TypeError),
            (torch.empty(16, 8, device="meta"), ValueError),
        ):
            with self.assertRaises(error):
                embedding.load_target_weight(source)
            torch.testing.assert_close(
                embedding.weight, torch.ones_like(embedding.weight), rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()
