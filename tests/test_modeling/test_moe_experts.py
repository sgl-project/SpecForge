# coding=utf-8
"""MoE experts: Qwen3.5-MoE layout, grouped_mm == fallback loop == per-token math."""

import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeExperts,
    Qwen3_5MoeMLP,
)

from specforge.modeling.draft.moe import experts as experts_module
from specforge.modeling.draft.moe.experts import RoutedExperts, SharedExpert, swiglu

E, H, I, T, K = 4, 64, 32, 10, 2
UNUSED_EXPERT = 1


def _clamped_swiglu(gate, up, limit=1.0):
    """A DeepSeek-V4-style activation: fp32, gate clamped above, up both ways."""
    gate, up = gate.float().clamp(max=limit), up.float().clamp(-limit, limit)
    return F.silu(gate) * up


def _routing(device="cpu"):
    """Distinct top-k experts per token; ``UNUSED_EXPERT`` is never picked."""
    generator = torch.Generator().manual_seed(0)
    choices = torch.tensor([e for e in range(E) if e != UNUSED_EXPERT])
    indices = torch.stack(
        [
            choices[torch.randperm(len(choices), generator=generator)[:K]]
            for _ in range(T)
        ]
    )
    weights = torch.rand(T, K, generator=generator)
    return indices.to(device), weights.to(device)


def _per_token_reference(x, indices, weights, gate_up_proj, down_proj, activation):
    rows = []
    for t in range(x.shape[0]):
        row = torch.zeros_like(x[t])
        for slot in range(indices.shape[1]):
            expert = indices[t, slot]
            gate, up = (gate_up_proj[expert] @ x[t]).chunk(2)
            row = row + weights[t, slot] * (down_proj[expert] @ activation(gate, up))
        rows.append(row)
    return torch.stack(rows)


def _forward_and_grads(experts, x, indices, weights, reference=False):
    x = x.clone().requires_grad_()
    weights = weights.clone().requires_grad_()
    params = (experts.gate_up_proj, experts.down_proj)
    if reference:
        out = _per_token_reference(x, indices, weights, *params, experts.activation)
    else:
        out = experts(x, indices, weights)
    grads = torch.autograd.grad(out.square().sum(), (x, weights, *params))
    return [out, *grads]


def _assert_all_close(got, want, **tolerance):
    for g, w in zip(got, want):
        torch.testing.assert_close(g.float(), w.float(), **tolerance)


def _qwen3_5_config():
    return Qwen3_5MoeTextConfig(
        num_experts=E, hidden_size=H, moe_intermediate_size=I, hidden_act="silu"
    )


class RoutedExpertsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.experts = RoutedExperts(E, H, I, initializer_range=0.02)
        self.x = torch.randn(T, H)
        self.indices, self.weights = _routing()

    def test_layout_and_math_match_transformers_qwen3_5_moe(self):
        reference = Qwen3_5MoeExperts(_qwen3_5_config())
        reference.load_state_dict(self.experts.state_dict())  # strict: same names
        torch.testing.assert_close(
            self.experts(self.x, self.indices, self.weights),
            reference(self.x, self.indices, self.weights),
        )

    def test_grouped_mm_matches_the_per_token_reference_with_gradients(self):
        args = (self.experts, self.x, self.indices, self.weights)
        got = _forward_and_grads(*args)
        reference = _forward_and_grads(*args, reference=True)
        _assert_all_close(got, reference, rtol=1e-4, atol=1e-6)
        self.assertTrue(torch.all(got[3][UNUSED_EXPERT] == 0))

    def test_fallback_loop_matches_grouped_mm(self):
        args = (self.experts, self.x, self.indices, self.weights)
        grouped = _forward_and_grads(*args)
        with mock.patch.object(experts_module, "_GROUPED_MM_DEVICES", ()):
            looped = _forward_and_grads(*args)
        _assert_all_close(looped, grouped, rtol=1e-4, atol=1e-6)

    def test_unaligned_sizes_take_the_fallback_loop(self):
        for hidden, intermediate in ((63, I), (H, 31)):
            with self.subTest(hidden=hidden, intermediate=intermediate):
                experts = RoutedExperts(E, hidden, intermediate, initializer_range=0.02)
                args = (experts, torch.randn(T, hidden), self.indices, self.weights)
                _assert_all_close(
                    _forward_and_grads(*args),
                    _forward_and_grads(*args, reference=True),
                    rtol=1e-4,
                    atol=1e-6,
                )

    def test_activation_seam_runs_a_clamped_fp32_activation(self):
        experts = RoutedExperts(
            E, H, I, initializer_range=0.5, activation=_clamped_swiglu
        )
        args = (experts, self.x, self.indices, self.weights)
        _assert_all_close(
            _forward_and_grads(*args),
            _forward_and_grads(*args, reference=True),
            rtol=1e-4,
            atol=1e-5,
        )

    def test_bf16_experts_return_fp32_on_both_paths(self):
        experts = RoutedExperts(E, H, I, initializer_range=0.02).bfloat16()
        args = (self.x.bfloat16(), self.indices, self.weights)
        for activation in (swiglu, _clamped_swiglu):
            experts.activation = activation
            self.assertEqual(experts(*args).dtype, torch.float32)
            with mock.patch.object(experts_module, "_GROUPED_MM_DEVICES", ()):
                self.assertEqual(experts(*args).dtype, torch.float32)

    def test_no_tokens(self):
        out = self.experts(
            torch.zeros(0, H), torch.zeros(0, K, dtype=torch.long), torch.zeros(0, K)
        )
        self.assertEqual(tuple(out.shape), (0, H))

    def test_reset_parameters_reinitializes_with_std(self):
        experts = RoutedExperts(64, 256, 128, initializer_range=0.02)
        with torch.no_grad():
            experts.gate_up_proj.fill_(1e30)
        experts.reset_parameters(0.02)
        for param in (experts.gate_up_proj, experts.down_proj):
            self.assertAlmostEqual(param.std().item(), 0.02, delta=0.002)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_bf16_matches_cpu_fp32(self):
        cpu = _forward_and_grads(self.experts, self.x, self.indices, self.weights)
        experts = RoutedExperts(E, H, I, initializer_range=0.02)
        experts.load_state_dict(self.experts.state_dict())
        experts = experts.to(device="cuda", dtype=torch.bfloat16)
        cuda = _forward_and_grads(
            experts,
            self.x.to("cuda", torch.bfloat16),
            self.indices.cuda(),
            self.weights.cuda(),
        )
        _assert_all_close([t.cpu() for t in cuda], cpu, rtol=2e-2, atol=2e-3)


class SharedExpertTest(unittest.TestCase):
    def test_layout_and_math_match_transformers_qwen3_5_moe(self):
        torch.manual_seed(0)
        shared = SharedExpert(H, I)
        reference = Qwen3_5MoeMLP(_qwen3_5_config(), intermediate_size=I)
        reference.load_state_dict(shared.state_dict())  # strict: same names
        x = torch.randn(3, H)
        torch.testing.assert_close(shared(x), reference(x))

    def test_fp32_activation_casts_back_before_the_down_projection(self):
        shared = SharedExpert(H, I, activation=_clamped_swiglu).bfloat16()
        out = shared(torch.randn(3, H, dtype=torch.bfloat16))
        self.assertEqual(out.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
