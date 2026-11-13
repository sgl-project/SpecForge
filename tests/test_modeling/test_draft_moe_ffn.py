"""MoE FFN: the router contract."""

import unittest

import torch

from specforge.config.draft.mlp import MoERouterSpec
from specforge.modeling.draft.layers.mlp.moe import Router, TopKRouter

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


class MoEFFNTest(unittest.TestCase):
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
        router.expert_bias = None
        with self.assertRaisesRegex(ValueError, "disagrees"):
            router.validate_draft_state()
        plain = TopKRouter(router_spec(), num_experts=4, hidden_size=H, top_k=1)
        self.assertIsNone(plain.expert_bias)
        plain.validate_draft_state()
        plain.validate_export_state(dict(plain.state_dict()))
        self.assertEqual(sorted(plain.state_dict()), ["weight"])


if __name__ == "__main__":
    unittest.main()
