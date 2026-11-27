"""Cached recurrent execution matches the general model forward and gradients."""

import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint

from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.attention.masks import AttentionLayout
from specforge.modeling.draft.layers.inputs import LayerInputs
from tests.test_modeling.test_canonical_eagle3 import model_config


class _FullForwardRollout:
    """Uncached model.forward() reference used to verify EagleRollout parity."""

    def __init__(self, model):
        self.model = model
        self.tokens, self.features, self.positions = [], [], []

    def step(self, input_ids, features, padding, positions):
        depth, length = len(self.tokens), input_ids.shape[1]
        self.tokens.append(input_ids)
        self.features.append(features)
        self.positions.append(positions + depth)
        all_positions = torch.cat(self.positions, dim=-1)
        logical = all_positions[0] if all_positions.ndim == 3 else all_positions
        valid_length = padding.sum(-1)

        def visible(b, h, q, k):
            qa, ka = q % length, k % length
            qd, kd = q // length, k // length
            return (
                (((kd == 0) & (ka <= qa)) | ((ka == qa) & (kd <= qd)))
                & (qa + qd < valid_length[b])
                & (ka + kd < valid_length[b])
            )

        indices = torch.arange(all_positions.shape[-1], device=padding.device)[None]
        layout = AttentionLayout(
            query_positions=logical,
            key_positions=logical,
            visibility=visible,
            active_queries=(
                indices % length + indices // length < valid_length[:, None]
            ),
        )
        output = self.model(
            torch.cat(self.tokens, dim=1),
            None,
            (LayerInputs(position_ids=all_positions, attention_layout=layout),),
            recurrent_hidden=torch.cat(self.features, dim=1),
            output_logits=False,
        )
        return output.hidden_states[:, -length:]


class EagleRolloutTest(unittest.TestCase):
    def test_cache_matches_full_forward_with_checkpoint_recomputation(self):
        self.check_backends(
            torch.float32,
            (
                [None, "eager", "sdpa", "flex_attention"]
                if torch.cuda.is_available()
                else [None, "eager", "sdpa"]
            ),
            3e-5,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA FlashAttention")
    def test_flash_cached_rollout_matches_sdpa(self):
        self.check_backends(torch.bfloat16, ["sdpa", "fa"], 0.025)

    def check_backends(self, dtype, backends, tolerance):
        for norm_output in (False, True):
            with self.subTest(norm_output=norm_output):
                self._check_backends(dtype, backends, tolerance, norm_output)

    def _check_backends(self, dtype, backends, tolerance, norm_output):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        config = model_config(norm_output=norm_output)
        config = replace(
            config,
            attention_configs={
                name: replace(node, head_dim=16)
                for name, node in config.attention_configs.items()
            },
        )
        original = build_draft_model(config, device=device, dtype=dtype)
        ids = torch.arange(8, device=device)[None]
        features = torch.randn(
            1,
            8,
            original.config.draft.spec_config.target_features.input_width,
            device=device,
            dtype=dtype,
        )
        padding = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]], device=device)
        results = []
        for backend in backends:
            model = copy.deepcopy(original)
            model.gradient_checkpointing_enable()
            rollout = (
                _FullForwardRollout(model)
                if backend is None
                else model.new_rollout(backend)
            )
            hidden = model.fusion(features)
            outputs = []
            for depth in range(3):
                hidden = rollout.step(ids + depth, hidden, padding, ids)
                outputs.append(hidden[:, :4])
            output = torch.stack(outputs)
            output.square().mean().backward()
            results.append(
                (
                    output,
                    {
                        name: p.grad
                        for name, p in model.named_parameters()
                        if p.grad is not None
                    },
                )
            )
        reference, gradients = results[0]
        for output, grads in results[1:]:
            torch.testing.assert_close(
                output, reference, atol=tolerance, rtol=tolerance
            )
            self.assertEqual(grads.keys(), gradients.keys())
            for name in grads:
                torch.testing.assert_close(
                    grads[name], gradients[name], atol=tolerance, rtol=tolerance
                )

    def test_cached_step_normalizes_features_once_when_norm_before_residual(self):
        model = build_draft_model(model_config())
        stage = model.backbone.layers[0]
        stage.norm_before_residual = True
        ids = torch.arange(6)[None]
        features = torch.randn(
            1, 6, model.config.draft.spec_config.target_features.input_width
        )
        padding = torch.tensor([[1, 1, 1, 1, 0, 0]])
        hidden = model.fusion(features)
        rollout = model.new_rollout("eager")
        reference = _FullForwardRollout(model)
        with patch.object(
            stage.hidden_norm, "forward", wraps=stage.hidden_norm.forward
        ) as hidden_norm:
            actual = rollout.step(ids, hidden, padding, ids)
        self.assertEqual(hidden_norm.call_count, 1)
        expected = reference.step(ids, hidden, padding, ids)
        torch.testing.assert_close(actual[:, :4], expected[:, :4], atol=3e-5, rtol=3e-5)

    def test_eval_and_no_grad_bypass_checkpointing(self):
        model = build_draft_model(model_config())
        model.gradient_checkpointing_enable()
        ids = torch.arange(6)[None]
        features = torch.randn(
            1, 6, model.config.draft.spec_config.target_features.input_width
        )
        padding = torch.ones(1, 6, dtype=torch.long)
        hidden = model.fusion(features)
        with patch(
            "specforge.modeling.draft.models.eagle_rollout.checkpoint",
            wraps=checkpoint,
        ) as call:
            model.eval()
            model.new_rollout("eager").step(ids, hidden, padding, ids)
            model.train()
            with torch.no_grad():
                model.new_rollout("eager").step(ids, hidden, padding, ids)
            call.assert_not_called()
            model.new_rollout("eager").step(ids, hidden, padding, ids)
            self.assertEqual(call.call_count, 1)

    def test_new_rollout_rejects_non_gqa_attention(self):
        model = build_draft_model(model_config())
        model.backbone.layers[0].self_attn = torch.nn.Identity()
        for backend in ("eager", "sdpa", "fa"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "GQA/MHA"):
                    model.new_rollout(backend)


if __name__ == "__main__":
    unittest.main()
