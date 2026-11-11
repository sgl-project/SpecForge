"""Target modules stay frozen, omit target tensors and preserve muP numerics."""

import unittest
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import ClassVar
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch import nn

from specforge.config.draft.codec.components import decode_component
from specforge.config.draft.logits import (
    MuPLogitTransformSpec,
    NoLogitTransformSpec,
    TargetLogitSpec,
)
from specforge.config.draft.registry import _SCHEMAS, register_component_schema
from specforge.modeling.draft.bindings import BoundEmbedding, BoundLMHead
from specforge.modeling.draft.layers.logits import TargetLogitTransform
from specforge.modeling.draft.layers.registry import (
    _BUILDERS,
    register_component_builder,
)
from specforge.modeling.draft.pretrained import DraftPreTrainedModel
from tests.test_modeling.test_draft_pretrained import model_config


class DFlashDraftModel(DraftPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.tokens = owner()
        self.post_init()


def owner():
    module = nn.Module()
    module.embed_tokens = BoundEmbedding(16, 8)
    module.lm_head = BoundLMHead(NoLogitTransformSpec(), vocab_size=16, hidden_size=8)
    module.proj = nn.Linear(8, 8)
    return module


@dataclass(frozen=True, kw_only=True)
class RowNormalizedSpec(TargetLogitSpec):
    type: ClassVar[str] = "test_row_normalized_head"


class RowNormalizedHead(TargetLogitTransform):
    def transform_weight_(self, weight):
        weight.div_(weight.norm(dim=-1, keepdim=True))


class BindingTest(unittest.TestCase):
    def test_registered_transform_preserves_full_and_weight_based_scoring(self):
        self.enterContext(patch.dict(_SCHEMAS))
        self.enterContext(patch.dict(_BUILDERS))
        register_component_schema("logits", RowNormalizedSpec)
        config = decode_component("logits", {"type": RowNormalizedSpec.type})
        with self.assertRaisesRegex(ValueError, "no modeling builder"):
            BoundLMHead(config, vocab_size=4, hidden_size=2)
        register_component_builder("logits", RowNormalizedSpec, RowNormalizedHead)
        head = BoundLMHead(config, vocab_size=4, hidden_size=2)
        source = torch.arange(1, 9, dtype=torch.float32).reshape(4, 2)
        expected = F.normalize(source, dim=-1)
        hidden = torch.randn(2, 2)
        for _ in range(2):
            head.load_target_weight(source)
            torch.testing.assert_close(head(hidden), F.linear(hidden, expected))
            torch.testing.assert_close(
                F.embedding(torch.tensor([0, 3]), head.bound_weight()), expected[[0, 3]]
            )
        self.assertIs(head.logit_transform.config, config)
        self.assertEqual(head.state_dict(), {})
        self.assertFalse(head.weight.requires_grad)
        torch.testing.assert_close(source, torch.arange(1, 9).reshape(4, 2).float())
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_component_builder(
                "logits", RowNormalizedSpec, TargetLogitTransform
            )

        transform = RowNormalizedHead(config)
        transform.register_buffer("scale", torch.ones(1))
        with patch(
            "specforge.modeling.draft.bindings.build_component", return_value=transform
        ):
            with self.assertRaisesRegex(ValueError, "parameters or buffers"):
                BoundLMHead(config, vocab_size=4, hidden_size=2)

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

    def test_restore_materializes_bound_modules_without_binding(self):
        model = DFlashDraftModel(model_config())
        with TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = DFlashDraftModel.from_pretrained(directory)
        restored.to("cpu")
        torch.testing.assert_close(
            restored.tokens.proj.weight, model.tokens.proj.weight
        )
        for module in (restored.tokens.embed_tokens, restored.tokens.lm_head):
            self.assertFalse(module.weight.is_meta)
            with self.assertRaisesRegex(RuntimeError, "not bound"):
                module.bound_weight()
            module.load_target_weight(torch.ones(16, 8))
            torch.testing.assert_close(module.bound_weight(), torch.ones(16, 8))

    def test_dtype_and_storage_changes_require_rebinding(self):
        source = torch.full((4, 2), 0.9)
        head = BoundLMHead(
            MuPLogitTransformSpec(multiplier=3), vocab_size=4, hidden_size=2
        )
        head.load_target_weight(source)
        head.float()  # A no-op preserves the binding.
        self.assertIs(head.bound_weight(), head.weight)
        for move in (
            lambda: head.bfloat16(),
            lambda: head.to_empty(device="cpu"),
            lambda: head.to("meta").to_empty(device="cpu"),
        ):
            move()
            with self.assertRaisesRegex(RuntimeError, "not bound"):
                head.bound_weight()
            with self.assertRaisesRegex(RuntimeError, "not bound"):
                head(torch.ones(1, 2, dtype=head.weight.dtype))
            head.load_target_weight(source)
            torch.testing.assert_close(
                head.bound_weight(), source.to(head.weight.dtype) / 3, rtol=0, atol=0
            )

    def test_binding_rejects_its_own_storage_without_folding_again(self):
        head = BoundLMHead(
            MuPLogitTransformSpec(multiplier=3), vocab_size=4, hidden_size=2
        )
        head.load_target_weight(torch.ones(4, 2))
        before = head.bound_weight().clone()
        for source in (
            head.weight,
            head.weight.detach(),
            head.weight.view_as(head.weight),
        ):
            with self.assertRaisesRegex(ValueError, "share storage"):
                head.load_target_weight(source)
            torch.testing.assert_close(head.bound_weight(), before, rtol=0, atol=0)

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
