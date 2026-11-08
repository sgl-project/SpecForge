"""Checkpointing preserves model-owned fusion/injection and nested input graphs."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from specforge.modeling.draft.backbone import LayerInputs
from tests.test_modeling.test_draft_backbone import make_backbone


class BackboneCheckpointTest(unittest.TestCase):
    def _assert_autocast_parity(self, device):
        baseline = make_backbone().to(device)
        hidden = torch.randn(1, 3, 4, device=device)
        inputs = [
            LayerInputs(position_ids=torch.arange(3, device=device).unsqueeze(0))
        ] * 2
        results = []
        for enabled in (False, True):
            backbone = copy.deepcopy(baseline)
            backbone.gradient_checkpointing = enabled
            with torch.autocast(device, dtype=torch.bfloat16):
                output = backbone(hidden, inputs)
                loss = output.float().square().sum()
            loss.backward()
            results.append((output, [p.grad for p in backbone.parameters()]))
        torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)

    def test_cpu_autocast_recomputation_matches(self):
        self._assert_autocast_parity("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_autocast_recomputation_matches(self):
        self._assert_autocast_parity("cuda")

    def test_fusion_context_rope_and_injection_gradients_match(self):
        baseline = make_backbone()
        # Exercise RNG restoration on recomputation as well as parameter gradients.
        baseline.layers[0].proj = nn.Sequential(
            baseline.layers[0].proj, nn.Dropout(0.25)
        )
        fusion = nn.Linear(8, 4)
        injection = nn.Linear(4, 4)
        seed = torch.randn(1, 3, 8)
        context_seed = torch.randn(1, 3, 4)
        position_seed = torch.randn(1, 3, 1)
        results = []
        for enabled, manual in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            backbone, fuse, inject = copy.deepcopy((baseline, fusion, injection))
            backbone.gradient_checkpointing = enabled
            raw = seed.clone().requires_grad_()
            context = context_seed.clone().requires_grad_()
            angles = position_seed.clone().requires_grad_()
            inputs = [
                LayerInputs(
                    target_context=context,
                    position_embeddings=(angles.cos(), angles.sin()),
                )
            ] * backbone.num_layers
            hidden = fuse(raw)
            torch.manual_seed(123)
            if manual:
                state = backbone.begin(hidden, inputs)
                features = []
                for _ in range(backbone.num_layers):
                    state = backbone.run_next_layer(state)
                    pre_injection = state.hidden
                    state = state.with_hidden(inject(state.hidden))
                    features.append(state.hidden)
                output = backbone.finalize(state)
                auxiliary = pre_injection.sum() + sum(x.sum() for x in features)
            else:
                output = backbone(hidden, inputs)
                auxiliary = 0
            (output.square().sum() + auxiliary).backward()
            parameters = list(backbone.parameters()) + list(fuse.parameters())
            if manual:
                parameters += list(inject.parameters())
            gradients = [p.grad for p in parameters] + [
                raw.grad,
                context.grad,
                angles.grad,
            ]
            self.assertTrue(all(g is not None for g in gradients))
            results.append((output, gradients, torch.rand(4)))
        for plain, checked in ((results[0], results[1]), (results[2], results[3])):
            torch.testing.assert_close(plain, checked, rtol=0, atol=0)

    def test_frozen_inputs_still_train_layers_and_module_hooks_run(self):
        backbone = make_backbone()
        backbone.gradient_checkpointing = True
        calls = []
        handle = backbone.layers[0].register_forward_pre_hook(
            lambda *args: calls.append(1)
        )
        inputs = [LayerInputs(position_ids=torch.arange(3).unsqueeze(0))] * 2
        output = backbone(torch.randn(1, 3, 4), inputs)
        output.square().sum().backward()
        handle.remove()
        self.assertGreaterEqual(len(calls), 2)
        self.assertTrue(all(p.grad is not None for p in backbone.parameters()))

    def test_eval_and_no_grad_bypass_checkpointing(self):
        backbone = make_backbone()
        backbone.gradient_checkpointing = True
        inputs = [LayerInputs(position_ids=torch.arange(3).unsqueeze(0))] * 2
        with patch(
            "specforge.modeling.draft.backbone.checkpoint", wraps=checkpoint
        ) as call:
            backbone.eval()
            backbone(torch.randn(1, 3, 4), inputs)
            backbone.train()
            with torch.no_grad():
                backbone(torch.randn(1, 3, 4), inputs)
            call.assert_not_called()
            backbone(torch.randn(1, 3, 4), inputs)
            self.assertEqual(call.call_count, 2)
            self.assertIs(call.call_args.kwargs["use_reentrant"], False)


if __name__ == "__main__":
    unittest.main()
