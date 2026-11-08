"""Candidate transitions reuse learned refinement and the configured token head."""

import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from specforge.config.draft.logits import MuPLogitTransformSpec
from specforge.modeling import build_draft_model
from specforge.modeling.draft.dspine import DSpineDraftModel
from tests.test_modeling.test_canonical_dspine import model_config


def initialized_model(device="cpu", dtype=torch.float32, *, binding="owned"):
    config = model_config(bound=binding != "owned")
    if binding == "mup":
        config = replace(
            config, target_logit_transform=MuPLogitTransformSpec(multiplier=3)
        )
    model = build_draft_model(config, device=device, dtype=dtype)
    source = torch.randn(64, 32, device=device, requires_grad=True)
    model.initialize_transfer_space(source)
    if binding != "owned":
        model.lm_head.load_target_weight(source)
    with torch.no_grad():
        for write in model.injection.writes:
            write.weight.normal_(std=0.1)
    return model, source


class DSpineTransitionTest(unittest.TestCase):
    def _parity(self, device, dtype):
        for binding in ("owned", "none", "mup"):
            with self.subTest(device=device, dtype=dtype, binding=binding):
                model, source = initialized_model(device, dtype, binding=binding)
                reference = copy.deepcopy(model)
                hidden = torch.randn(
                    2, 2, 3, 32, device=device, dtype=dtype, requires_grad=True
                )
                other = hidden.detach().clone().requires_grad_()
                candidates = torch.randint(64, (2, 2, 3, 4), device=device)
                anchors = torch.randint(64, (2, 2), device=device)
                output = model.transition_scores(hidden, candidates, anchors)
                expected = DSpineDraftModel.transition_scores(
                    reference, other, candidates, anchors, reference.lm_head.weight
                )
                self.assertEqual(output.shape, (2, 2, 3, 4, 4))
                self.assertEqual(output.dtype, torch.float32)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                upstream = torch.randn_like(output)
                output.backward(upstream)
                expected.backward(upstream)
                torch.testing.assert_close(hidden.grad, other.grad, rtol=0, atol=0)
                for name, parameter in model.named_parameters():
                    ref = dict(reference.named_parameters())[name]
                    if ref.grad is None:
                        self.assertIsNone(parameter.grad, name)
                    else:
                        torch.testing.assert_close(
                            parameter.grad, ref.grad, rtol=0, atol=0
                        )
                self.assertIsNotNone(model.injection.writes[-1].weight.grad)
                self.assertEqual(
                    model.lm_head.weight.grad is not None, binding == "owned"
                )
                if binding != "owned":
                    weights = source.detach().to(dtype).clone()
                    if binding == "mup":
                        weights.div_(3)
                    torch.testing.assert_close(
                        model.lm_head.candidate_weights(candidates),
                        torch.nn.functional.embedding(candidates, weights),
                        rtol=0,
                        atol=0,
                    )
                self.assertIsNone(source.grad)
                self.assertIsNone(model.transfer_codes.grad)

    def test_cpu_legacy_scores_and_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            self._parity("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_legacy_scores_and_gradients(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._parity("cuda", dtype)

    def test_candidate_axes_and_anchor_alignment(self):
        model, _ = initialized_model()
        # Scalar dot products encode predecessor and successor IDs exactly.
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.lm_head.weight[:, 0] = torch.arange(64)
        hidden = torch.zeros(3, 32)
        candidates = torch.tensor([[2, 4], [5, 7], [8, 9]])
        anchor = torch.tensor(3)

        def refine(states, ids):
            result = torch.zeros_like(states)
            result[..., 0] = ids
            return result

        with patch.object(model, "refine", side_effect=refine):
            output = model.transition_scores(hidden, candidates, anchor)
        expected = torch.tensor(
            [[[6, 12], [6, 12]], [[10, 14], [20, 28]], [[40, 45], [56, 63]]],
            dtype=torch.float32,
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_unbound_head_and_invalid_inputs(self):
        model = build_draft_model(model_config(bound=True))
        model.initialize_transfer_space(torch.randn(64, 32))
        hidden, ids, anchors = (
            torch.randn(2, 3, 32),
            torch.ones(2, 3, 4, dtype=torch.long),
            torch.ones(2, dtype=torch.long),
        )
        with self.assertRaisesRegex(RuntimeError, "not bound"):
            model.transition_scores(hidden, ids, anchors)
        for candidates, first, error in (
            (ids[..., :0], anchors, ValueError),
            (ids[:, :2], anchors, ValueError),
            (ids, anchors[:1], ValueError),
            (ids.float(), anchors, TypeError),
            (ids, anchors.float(), TypeError),
        ):
            with self.assertRaises(error):
                model.transition_scores(hidden, candidates, first)


if __name__ == "__main__":
    unittest.main()
