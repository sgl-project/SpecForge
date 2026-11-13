"""MoE FFN: parity with Transformers, kernel agreement, the router contract, shared experts."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers import DeepseekV3Config, NemotronHConfig
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3MLP
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMLP
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeMLP,
    Qwen3_5MoeSparseMoeBlock,
)

from specforge.config.draft.mlp import (
    DenseFFNSpec,
    MoEExpertSpec,
    MoEFFNSpec,
    MoERouterSpec,
)
from specforge.modeling.draft.layers import build_component, component_layout
from specforge.modeling.draft.layers.inputs import ExpertRouting, MoEOutput
from specforge.modeling.draft.layers.kernels import configure_kernels
from specforge.modeling.draft.layers.layout import join, split
from specforge.modeling.draft.layers.mlp import EXPERT_KERNELS, DenseFFN, MoEFFN
from specforge.modeling.draft.layers.mlp.moe import (
    MOE_LAYOUT,
    RoutedExperts,
    Router,
    SharedExpert,
    TopKRouter,
    expert_kernel,
)
from tests.test_modeling.test_draft_pretrained import DFlashDraftModel, model_config

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

    def test_kernel_table_rejects_duplicate_names(self):
        self.assertEqual(sorted(EXPERT_KERNELS), ["grouped_mm", "loop"])
        with self.assertRaisesRegex(ValueError, "already registered"):
            expert_kernel("loop")(lambda *args, **kwargs: None)
        self.assertEqual(sorted(EXPERT_KERNELS), ["grouped_mm", "loop"])

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

    def test_shared_expert_settings_match_transformers(self):
        gated = expert_spec(48)
        relu2 = expert_spec(48, "relu2", gated=False)
        deepseek = DeepseekV3Config(
            hidden_size=H,
            hidden_act="silu",
            moe_intermediate_size=24,
            n_shared_experts=2,
        )
        nemotron = NemotronHConfig(
            hidden_size=H, mlp_hidden_act="relu2", mlp_bias=False
        )
        x = torch.randn(5, H)
        for setting, spec, reference in (
            (
                "qwen gated",
                gated,
                Qwen3_5MoeMLP(
                    Qwen3_5MoeTextConfig(hidden_size=H, hidden_act="silu"),
                    intermediate_size=48,
                ),
            ),
            (
                "deepseek merged shared experts",
                gated,
                DeepseekV3MLP(
                    deepseek,
                    intermediate_size=deepseek.moe_intermediate_size
                    * deepseek.n_shared_experts,
                ),
            ),
            ("nemotron relu2", relu2, NemotronHMLP(nemotron, intermediate_size=48)),
        ):
            with self.subTest(setting=setting):
                actual = SharedExpert(spec, hidden_size=H)
                actual.load_state_dict(reference.state_dict(), strict=True)
                torch.testing.assert_close(actual(x), reference(x), rtol=0, atol=0)
        self.assertIsNone(SharedExpert(relu2, hidden_size=H).gate_proj)

    @unittest.skipUnless(CUDA, "Liger kernels require CUDA")
    def test_liger_swiglu_fuses_gated_shared_experts_only(self):
        calls = []

        def fake_swiglu(gate, up):
            calls.append(tuple(gate.shape))
            return F.silu(gate) * up

        def shared(hidden_act, gated):
            spec = expert_spec(48, hidden_act, gated)
            return SharedExpert(spec, hidden_size=H, device="cuda")

        model = torch.nn.ModuleDict(
            {
                "gated": shared("silu", True),
                "plain": shared("silu", False),
                "relu2": shared("relu2", False),
                "dense": DenseFFN(
                    DenseFFNSpec(intermediate_size=48, hidden_act="silu"),
                    hidden_size=H,
                    device="cuda",
                ),
            }
        )
        x = torch.randn(5, H, device="cuda")
        eager = {name: module(x) for name, module in model.items()}
        with patch(
            "specforge.modeling.draft.layers.kernels.load_liger_kernels",
            return_value=(object(), fake_swiglu),
        ):
            configure_kernels(model, use_liger=True)
        self.assertIs(model["gated"].liger_swiglu, fake_swiglu)
        self.assertIs(model["dense"].liger_swiglu, fake_swiglu)
        self.assertIsNone(model["relu2"].liger_swiglu)
        fused = {name: module(x) for name, module in model.items()}
        self.assertEqual(calls, [(5, 48), (5, 48)])
        for name in model:
            torch.testing.assert_close(fused[name], eager[name], rtol=0, atol=0)
        configure_kernels(model, use_liger=False)
        self.assertTrue(all(m.liger_swiglu is None for m in model.values()))

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

    def test_forward_returns_the_typed_update_with_its_routing(self):
        module = MoEFFN(qwen_spec(), hidden_size=H)
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.normal_(std=0.2)
        x = torch.randn(2, 5, H)
        for training in (True, False):
            with self.subTest(training=training):
                module.train(training)
                output = module(x)
                self.assertIsInstance(output, MoEOutput)
                self.assertEqual(tuple(output.update.shape), (2, 5, H))
                self.assertEqual(output.update.dtype, torch.float32)
                self.assertIsInstance(output.routing, ExpertRouting)
                indices = output.routing.expert_ids
                self.assertEqual(tuple(indices.shape), (10, 2))
                self.assertEqual(indices.dtype, torch.long)
                self.assertFalse(indices.requires_grad)
                scores, expected, _ = module.gate(x.reshape(-1, H))
                self.assertTrue(torch.equal(indices, expected))
                # The router scores travel with the routing, differentiable for an aux loss.
                scores = output.routing.scores
                self.assertEqual(tuple(scores.shape), (10, module.config.num_experts))
                self.assertEqual(scores.dtype, torch.float32)
                self.assertTrue(scores.requires_grad)
                self.assertTrue(torch.equal(scores, module.gate(x.reshape(-1, H))[0]))
                scores[:, 0].sum().backward()
                self.assertTrue(torch.any(module.gate.weight.grad != 0))

    def test_generic_base_initializes_from_golden_config(self):
        owner = DFlashDraftModel(model_config(initializer_range=0))
        node = qwen_spec(router=router_spec(expert_bias=True))
        with torch.device("meta"):
            module = MoEFFN(node, hidden_size=H)
        module.to_empty(device="cpu")
        module.apply(owner._initialize_weights)
        for name, parameter in module.named_parameters():
            self.assertEqual(torch.count_nonzero(parameter).item(), 0, name)
        self.assertEqual(module.gate.expert_bias.dtype, torch.float32)
        self.assertEqual(torch.count_nonzero(module.gate.expert_bias).item(), 0)

    def test_builder_layout_and_round_trip(self):
        node = moe_spec()
        module = build_component("mlp", node, hidden_size=H)
        self.assertIsInstance(module, MoEFFN)
        self.assertIs(module.config, node)
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.normal_(std=0.2)
        self.assertIs(component_layout("mlp", "moe"), MOE_LAYOUT)
        fused = {
            f"experts.{k}": v.detach() for k, v in module.experts.state_dict().items()
        }
        self.assertEqual(tuple(fused["experts.gate_up_proj"].shape), (4, 64, H))
        self.assertEqual(tuple(fused["experts.down_proj"].shape), (4, H, 32))
        parts = split(MOE_LAYOUT, node, fused)
        self.assertEqual(
            sorted(k for k in parts if k.startswith("experts.0.")),
            [
                "experts.0.down_proj.weight",
                "experts.0.gate_proj.weight",
                "experts.0.up_proj.weight",
            ],
        )
        self.assertEqual(tuple(parts["experts.0.gate_proj.weight"].shape), (32, H))
        rebuilt = join(MOE_LAYOUT, node, parts)
        for key, value in fused.items():
            self.assertTrue(torch.equal(rebuilt[key], value), key)
        output = module(torch.randn(2, 3, H, dtype=torch.float32)).update
        self.assertEqual(
            (tuple(output.shape), output.dtype), ((2, 3, H), torch.float32)
        )

    def test_invalid_settings_fail_before_allocation(self):
        with patch("torch.nn.Parameter") as allocate:
            with self.assertRaises(TypeError):
                MoEFFN({}, hidden_size=H)
            with self.assertRaises(TypeError):
                SharedExpert(
                    DenseFFNSpec(intermediate_size=8, hidden_act="silu"), hidden_size=H
                )
            with self.assertRaises(TypeError):
                TopKRouter({}, num_experts=4, hidden_size=H, top_k=2)
            with self.assertRaisesRegex(ValueError, "gated"):
                RoutedExperts(expert_spec(gated=False), num_experts=2, hidden_size=H)
            for width in (0, True, 1.5):
                with self.assertRaises(ValueError):
                    MoEFFN(moe_spec(), hidden_size=width)
            with self.assertRaises(ValueError):
                TopKRouter(router_spec(), num_experts=2, hidden_size=H, top_k=3)
            allocate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
