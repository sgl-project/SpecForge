# coding=utf-8
"""Serving-side MoE FFN (``specforge.serving.sglang_models``).

The SGLang draft classes need SGLang and a GPU; they are exercised by
``scripts/gates/check_dspark_moe_sglang_equivalence.py``. Here the plain
PyTorch reference, the recipe defaults and the checkpoint-name mapping are
covered on CPU, plus the served block's loader when SGLang is importable.
"""

import importlib.util
import unittest

import torch

from specforge.serving.sglang_models.moe_ffn import (
    PRESET_DEFAULTS,
    DraftMoEFFN,
    routed_expert_count,
    to_qwen_names,
)


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _qwen35_cfg(e=8, k=3, inter=16, hidden=32, shared=24):
    return _Cfg(
        hidden_size=hidden,
        num_experts=e,
        num_experts_per_tok=k,
        moe_intermediate_size=inter,
        shared_expert_intermediate_size=shared,
        n_shared_experts=1,
        moe_preset="qwen3_5_moe",
        scoring_func="softmax",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        moe_router_bias=True,
    )


def _qwen_checkpoint(e, inter, hidden, shared, seed=1):
    torch.manual_seed(seed)
    w = {
        "gate.weight": torch.randn(e, hidden) * 0.2,
        "gate.bias": torch.randn(e),
        "shared_expert.gate_proj.weight": torch.randn(shared, hidden) * 0.1,
        "shared_expert.up_proj.weight": torch.randn(shared, hidden) * 0.1,
        "shared_expert.down_proj.weight": torch.randn(hidden, shared) * 0.1,
        "shared_expert_gate.weight": torch.randn(1, hidden) * 0.1,
    }
    for i in range(e):
        w[f"experts.{i}.gate_proj.weight"] = torch.randn(inter, hidden) * 0.1
        w[f"experts.{i}.up_proj.weight"] = torch.randn(inter, hidden) * 0.1
        w[f"experts.{i}.down_proj.weight"] = torch.randn(hidden, inter) * 0.1
    return w


def _plain_reference(w, x, k):
    e = w["gate.weight"].shape[0]
    logits = x @ w["gate.weight"].T + w["gate.bias"]
    probs = logits.softmax(-1)
    top_w, top_i = probs.topk(k, dim=-1)
    top_w = top_w / top_w.sum(-1, keepdim=True)
    y = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for j in range(k):
            i = int(top_i[t, j])
            h = torch.nn.functional.silu(w[f"experts.{i}.gate_proj.weight"] @ x[t]) * (
                w[f"experts.{i}.up_proj.weight"] @ x[t]
            )
            y[t] += top_w[t, j] * (w[f"experts.{i}.down_proj.weight"] @ h)
    hs = torch.nn.functional.silu(x @ w["shared_expert.gate_proj.weight"].T) * (
        x @ w["shared_expert.up_proj.weight"].T
    )
    y += torch.sigmoid(x @ w["shared_expert_gate.weight"].T) * (
        hs @ w["shared_expert.down_proj.weight"].T
    )
    return y


def _load_reference(ffn, w):
    e = ffn.n_experts
    with torch.no_grad():
        ffn.gate.weight.copy_(w["gate.weight"])
        ffn.gate.bias.copy_(w["gate.bias"])
        for name, proj in (("w1", "gate_proj"), ("w3", "up_proj"), ("w2", "down_proj")):
            getattr(ffn.experts, name).copy_(
                torch.stack([w[f"experts.{i}.{proj}.weight"] for i in range(e)])
            )
        ffn.shared_experts.w1.weight.copy_(w["shared_expert.gate_proj.weight"])
        ffn.shared_experts.w3.weight.copy_(w["shared_expert.up_proj.weight"])
        ffn.shared_experts.w2.weight.copy_(w["shared_expert.down_proj.weight"])
        ffn.shared_experts.gate.weight.copy_(w["shared_expert_gate.weight"])


class TestReferenceFFN(unittest.TestCase):
    def test_qwen3_5_moe_recipe_matches_a_plain_reference(self):
        # Kan's Qwen3.8-27B DSpark MoE export: softmax over (x W^T + folded
        # centering bias), top-k renormalised, no scaling, sigmoid-gated shared
        # expert.
        e, k, inter, hidden, shared = 8, 3, 16, 32, 24
        ffn = DraftMoEFFN(_qwen35_cfg(e, k, inter, hidden, shared)).float().eval()
        self.assertEqual(ffn.gate.bias_mode, "logit")
        self.assertEqual(ffn.shared_expert_gate, "sigmoid")
        w = _qwen_checkpoint(e, inter, hidden, shared)
        _load_reference(ffn, w)
        x = torch.randn(11, hidden)
        with torch.no_grad():
            torch.testing.assert_close(
                ffn(x), _plain_reference(w, x, k), atol=1e-4, rtol=1e-4
            )

    def test_preset_defaults_fill_missing_recipe_keys(self):
        cfg = _Cfg(
            hidden_size=16,
            num_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=8,
            moe_preset="qwen3_5_moe",
            moe_router_bias=True,
        )
        ffn = DraftMoEFFN(cfg)
        self.assertEqual(routed_expert_count(cfg), 4)
        self.assertEqual(
            ffn.scoring_func, PRESET_DEFAULTS["qwen3_5_moe"]["scoring_func"]
        )
        self.assertTrue(ffn.norm_topk_prob)
        self.assertIsNotNone(ffn.shared_experts)
        self.assertEqual(ffn.shared_expert_gate, "sigmoid")

    def test_to_qwen_names_maps_deepseek_layout_and_keeps_qwen_names(self):
        self.assertEqual(
            to_qwen_names("experts.7.w1.weight"), "experts.7.gate_proj.weight"
        )
        self.assertEqual(
            to_qwen_names("experts.7.w3.weight"), "experts.7.up_proj.weight"
        )
        self.assertEqual(
            to_qwen_names("experts.7.w2.weight"), "experts.7.down_proj.weight"
        )
        self.assertEqual(
            to_qwen_names("shared_experts.w1.weight"), "shared_expert.gate_proj.weight"
        )
        self.assertEqual(
            to_qwen_names("shared_experts.w2.weight"), "shared_expert.down_proj.weight"
        )
        self.assertEqual(
            to_qwen_names("shared_experts.gate.weight"), "shared_expert_gate.weight"
        )
        for name in (
            "gate.weight",
            "gate.bias",
            "experts.3.up_proj.weight",
            "shared_expert_gate.weight",
        ):
            self.assertEqual(to_qwen_names(name), name)


@unittest.skipUnless(
    importlib.util.find_spec("sglang") is not None and torch.cuda.is_available(),
    "needs SGLang and a GPU (covered by scripts/gates/check_dspark_moe_sglang_equivalence.py)",
)
class TestServedBlock(unittest.TestCase):
    def test_served_block_loads_and_matches_reference(self):
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "scripts/gates/check_dspark_moe_sglang_equivalence.py"],
            capture_output=True,
            text=True,
            timeout=1800,
        )
        self.assertIn(
            "RESULT: PASS", proc.stdout, proc.stdout[-2000:] + proc.stderr[-2000:]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
