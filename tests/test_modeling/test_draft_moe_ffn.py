"""MoE FFN: parity with Transformers, kernel agreement, the router contract, the kernel choice."""

import unittest

import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock,
)

from specforge.config.draft.mlp import MoEExpertSpec, MoEFFNSpec, MoERouterSpec
from specforge.modeling.draft.layers.inputs import MoEOutput
from specforge.modeling.draft.layers.kernels import configure_kernels
from specforge.modeling.draft.layers.mlp import EXPERT_KERNELS, MoEFFN
from specforge.modeling.draft.layers.mlp.moe import RoutedExperts, Router, TopKRouter

# Widths are multiples of 8: F.grouped_mm needs 16-byte strides in every dtype.
H = 32
CUDA = torch.cuda.is_available()


def router_spec(**overrides):
    values = {
        "scoring": "softmax",
        "normalize_top_k": True,
        "route_scale": 1.0,
        "expert_bias": False,
    }
    return MoERouterSpec(**{**values, **overrides})


def expert_spec(intermediate_size=32, hidden_act="silu", gated=True):
    return MoEExpertSpec(
        intermediate_size=intermediate_size, hidden_act=hidden_act, gated=gated
    )


def moe_spec(**overrides):
    values = {
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "experts": expert_spec(),
        "shared_expert": None,
        "shared_expert_gate": False,
        "router": router_spec(),
    }
    return MoEFFNSpec(**{**values, **overrides})


def qwen_spec(**overrides):
    """Qwen3.5-MoE's block: renormalized softmax top-k, a sigmoid-gated shared expert."""
    return moe_spec(
        shared_expert=expert_spec(),
        shared_expert_gate=True,
        **overrides,
    )


class MoEFFNTest(unittest.TestCase):
    @staticmethod
    def _run(module, x, upstream):
        inputs = x.detach().clone().requires_grad_()
        output = module(inputs)
        if isinstance(output, MoEOutput):
            output = output.update
        output.backward(upstream)
        grads = {name: p.grad for name, p in module.named_parameters()}
        return output.detach(), inputs.grad, grads

    def _assert_qwen_parity(self, device, dtypes):
        config = Qwen3_5MoeTextConfig(
            hidden_size=H,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            num_experts=4,
            num_experts_per_tok=2,
            hidden_act="silu",
        )
        torch.manual_seed(0)
        reference = Qwen3_5MoeSparseMoeBlock(config).to(device)
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.normal_(std=0.2)  # the router starts at zero
        weights = {k: v.clone() for k, v in reference.state_dict().items()}
        x = torch.randn(2, 5, H, device=device)
        upstream = torch.randn(2, 5, H, device=device)
        truth = self._run(reference, x, upstream)
        for dtype in dtypes:
            for kernel in sorted(EXPERT_KERNELS):
                with self.subTest(dtype=dtype, kernel=kernel):
                    expected = Qwen3_5MoeSparseMoeBlock(config).to(device, dtype)
                    expected.load_state_dict(
                        {k: v.to(dtype) for k, v in weights.items()}
                    )
                    actual = MoEFFN(
                        qwen_spec(), hidden_size=H, device=device, dtype=dtype
                    )
                    actual.experts_kernel = kernel
                    actual.load_state_dict(expected.state_dict(), strict=True)
                    self.assertEqual(
                        set(actual.state_dict()), set(expected.state_dict())
                    )
                    left = self._run(expected, x.to(dtype), upstream.to(dtype))
                    right = self._run(actual, x.to(dtype), upstream.to(dtype))
                    for name, theirs, ours, exact in (
                        ("output", left[0], right[0], truth[0]),
                        ("input grad", left[1], right[1], truth[1]),
                        *(
                            (key, left[2][key], right[2][key], truth[2][key])
                            for key in left[2]
                        ),
                    ):
                        self.assertEqual(ours.dtype, theirs.dtype, name)
                        if dtype == torch.float32:
                            # Measured 7e-7 on outputs and 4e-6 on parameter gradients:
                            # Transformers weights each expert after its down projection
                            # and accumulates per expert; the fused path weights before
                            # it and sums once.
                            torch.testing.assert_close(
                                ours, theirs, rtol=1e-5, atol=1e-5, msg=name
                            )
                            continue
                        # In bf16 the two also differ by where they round, so compare
                        # each to the fp32 computation: the fused path must stay as
                        # close as Transformers' own bf16 result (measured 1.2x at most).
                        gap = (ours.float() - exact).abs().max().item()
                        bound = (theirs.float() - exact).abs().max().item()
                        self.assertLessEqual(gap, 2 * bound + 1e-6, name)

    def test_cpu_parity_with_qwen3_5_moe(self):
        self._assert_qwen_parity("cpu", (torch.float32, torch.bfloat16))

    @unittest.skipUnless(CUDA, "CUDA is required")
    def test_cuda_parity_with_qwen3_5_moe(self):
        self._assert_qwen_parity("cuda", (torch.float32, torch.bfloat16))

    def _assert_kernels_agree(self, device, dtype):
        experts = RoutedExperts(
            expert_spec(intermediate_size=16),
            num_experts=3,
            hidden_size=H,
            device=device,
            dtype=dtype,
        )
        experts.reset_draft_parameters(initializer_range=0.2)
        x = torch.randn(6, H, device=device, dtype=dtype)
        indices = torch.tensor([[0, 2]] * 6, device=device)  # expert 1 is empty
        weights = torch.rand(6, 2, device=device)
        results = {}
        for kernel in EXPERT_KERNELS:
            experts.zero_grad()
            inputs = x.clone().requires_grad_()
            output = experts(inputs, indices, weights, kernel=kernel)
            output.square().sum().backward()
            results[kernel] = (
                output,
                inputs.grad,
                experts.gate_up_proj.grad.clone(),
                experts.down_proj.grad.clone(),
            )
        # Measured 0.0 on CPU fp32 and on H200 bf16 with torch 2.13: the same
        # operations in the same order; the tolerance covers other GEMM backends.
        tolerance = 1e-6 if dtype == torch.float32 else 1e-2
        for left, right in zip(results["grouped_mm"], results["loop"]):
            torch.testing.assert_close(left, right, rtol=tolerance, atol=tolerance)
        self.assertEqual(results["loop"][0].dtype, torch.float32)
        self.assertEqual(experts.gate_up_proj.grad[1].abs().sum().item(), 0.0)
        with self.assertRaisesRegex(ValueError, "unknown experts kernel 'triton'"):
            experts(x, indices, weights, kernel="triton")

    def test_cpu_expert_kernels_agree(self):
        self._assert_kernels_agree("cpu", torch.float32)

    @unittest.skipUnless(CUDA, "CUDA is required")
    def test_cuda_expert_kernels_agree(self):
        self._assert_kernels_agree("cuda", torch.bfloat16)

    def test_router_contract(self):
        spec = router_spec(scoring="sigmoid", route_scale=2.0, expert_bias=True)
        with self.assertRaises(NotImplementedError):
            Router(spec, num_experts=4, top_k=2)(torch.randn(3, H))
        router = TopKRouter(spec, num_experts=4, hidden_size=H, top_k=2)
        router.reset_draft_parameters(initializer_range=0.2)
        self.assertEqual(router.expert_bias.dtype, torch.float32)
        self.assertEqual(torch.count_nonzero(router.expert_bias).item(), 0)
        x = torch.randn(5, H)
        scores, indices, weights = router(x)
        self.assertEqual(tuple(scores.shape), (5, 4))
        self.assertEqual(tuple(indices.shape), (5, 2))
        self.assertEqual((scores.dtype, weights.dtype), (torch.float32, torch.float32))
        # Renormalized then scaled: every row of weights sums to route_scale.
        torch.testing.assert_close(weights.sum(-1), torch.full((5,), 2.0))
        # The bias shifts the choice only; the weights come from the unshifted scores.
        with torch.no_grad():
            router.expert_bias[0] = 10.0
        _, shifted, shifted_weights = router(x)
        self.assertTrue((shifted == 0).any(-1).all())
        torch.testing.assert_close(shifted_weights.sum(-1), torch.full((5,), 2.0))
        self.assertTrue((shifted_weights <= 2.0).all())
        router.validate_draft_state()
        state = dict(router.state_dict())
        self.assertEqual(sorted(state), ["expert_bias", "weight"])
        router.validate_export_state(state)
        for bad, message in (
            ({"weight": state["weight"]}, "router state"),
            ({**state, "expert_bias": state["expert_bias"].bfloat16()}, "fp32"),
            ({**state, "extra": torch.zeros(1)}, "router state"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                router.validate_export_state(bad)
        router.to(torch.bfloat16)
        self.assertEqual(router.weight.dtype, torch.bfloat16)
        self.assertEqual(router.expert_bias.dtype, torch.float32)
        router.validate_draft_state()
        # One DeepSeek-V3 step: under-used experts move up, the delta is zero-mean.
        before = router.expert_bias.clone()
        router.update_expert_bias(torch.tensor([10.0, 0.0, 0.0, 0.0]), rate=0.01)
        delta = router.expert_bias - before
        torch.testing.assert_close(delta, torch.tensor([-0.015, 0.005, 0.005, 0.005]))
        router.update_expert_bias(torch.tensor([10.0, 0.0, 0.0, 0.0]), rate=0.0)
        torch.testing.assert_close(router.expert_bias, before + delta)
        router.expert_bias = None
        with self.assertRaisesRegex(ValueError, "disagrees"):
            router.validate_draft_state()
        plain = TopKRouter(router_spec(), num_experts=4, hidden_size=H, top_k=1)
        self.assertIsNone(plain.expert_bias)
        plain.validate_draft_state()
        plain.validate_export_state(dict(plain.state_dict()))
        self.assertEqual(sorted(plain.state_dict()), ["weight"])

    def test_configure_kernels_selects_and_checks_the_expert_kernel(self):
        aligned = torch.nn.Sequential(
            MoEFFN(moe_spec(), hidden_size=H), MoEFFN(qwen_spec(), hidden_size=H)
        )
        for module in aligned:
            module.experts_kernel = "loop"
        configure_kernels(aligned)  # the default resolves to grouped_mm
        self.assertEqual([m.experts_kernel for m in aligned], ["grouped_mm"] * 2)
        configure_kernels(aligned, moe_experts_kernel="loop")
        self.assertEqual([m.experts_kernel for m in aligned], ["loop"] * 2)
        with self.assertRaisesRegex(ValueError, "unknown moe_experts_kernel 'triton'"):
            configure_kernels(aligned, moe_experts_kernel="triton")
        # torch accepts F.grouped_mm only with 16-byte strides; the probe reports
        # the layer and torch's reason, and points at the loop kernel.
        for dtype, hidden_size, intermediate_size in (
            (torch.float32, 18, 32),
            (torch.float32, 32, 18),
            (torch.bfloat16, 20, 32),
        ):
            with self.subTest(dtype=dtype, widths=(hidden_size, intermediate_size)):
                misaligned = MoEFFN(
                    moe_spec(experts=expert_spec(intermediate_size)),
                    hidden_size=hidden_size,
                    dtype=dtype,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    rf"'grouped_mm' cannot run a MoE layer \(hidden_size {hidden_size}, "
                    rf"intermediate_size {intermediate_size}, {dtype} on cpu\): .*"
                    r"set model.moe_experts_kernel: loop",
                ):
                    configure_kernels(misaligned)
                configure_kernels(misaligned, moe_experts_kernel="loop")
                self.assertEqual(misaligned.experts_kernel, "loop")
        with self.assertRaisesRegex(ValueError, "unknown experts kernel 'nope'"):
            aligned[0].experts.probe_kernel("nope")


if __name__ == "__main__":
    unittest.main()
