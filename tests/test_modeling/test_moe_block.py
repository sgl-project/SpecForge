# coding=utf-8
"""The MoE block: Qwen3.5-MoE parity, build_ffn and the family's config reading."""

import tempfile
import unittest
from unittest import mock

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock,
)

from specforge.modeling.draft import DFlashDraftModel
from specforge.modeling.draft.moe import MoELayer, build_ffn
from specforge.modeling.draft.moe import experts as experts_module
from specforge.modeling.draft.moe.families.qwen3_5_moe import build_moe
from specforge.modeling.draft.moe.router import TopKRouter

E, H, I, S, K = 4, 64, 32, 48, 2
TINY_DFLASH = {
    "architectures": ["DFlashDraftModel"],
    "model_type": "qwen3",
    "block_size": 4,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "num_target_layers": 4,
    "head_dim": 16,
    "max_position_embeddings": 512,
    "vocab_size": 256,
}
TINY_MOE = {
    "moe_preset": "qwen3_5_moe",
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 48,
    "norm_topk_prob": False,
}


def _config(**overrides):
    values = dict(
        num_experts=E,
        num_experts_per_tok=K,
        hidden_size=H,
        moe_intermediate_size=I,
        shared_expert_intermediate_size=S,
        norm_topk_prob=True,
    )
    return Qwen3_5MoeTextConfig(**{**values, **overrides})


def _pair(**overrides):
    """Our block and transformers' block with the same weights (strict: same names)."""
    layer = build_moe(_config(**overrides))
    reference = Qwen3_5MoeSparseMoeBlock(_config(**overrides))
    reference.load_state_dict(layer.state_dict())
    return layer, reference


def _forward_and_grads(module, x):
    x = x.clone().requires_grad_()
    out = module(x)
    params = dict(module.named_parameters())
    grads = torch.autograd.grad(out.square().sum(), [x, *params.values()])
    return out, dict(zip(["x", *params], grads))


def _tagged(tag):
    return lambda message: f"{tag}: {message}"


def _assert_same_forward_and_backward(test, module, reference, x, **tolerance):
    out, grads = _forward_and_grads(module, x)
    ref_out, ref_grads = _forward_and_grads(reference, x)
    torch.testing.assert_close(out, ref_out, msg=_tagged("output"), **tolerance)
    test.assertEqual(grads.keys(), ref_grads.keys())
    for name, grad in grads.items():
        torch.testing.assert_close(
            grad, ref_grads[name], msg=_tagged(f"grad of {name}"), **tolerance
        )


class MoEBlockTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, 5, H)

    def test_matches_transformers_qwen3_5_moe_sparse_block(self):
        tolerance = dict(rtol=1e-4, atol=1e-6)
        with self.subTest(path="grouped_mm"):
            layer, reference = _pair()
            _assert_same_forward_and_backward(
                self, layer, reference, self.x, **tolerance
            )
        with self.subTest(path="forced fallback loop"):
            layer, reference = _pair()
            with mock.patch.object(experts_module, "_GROUPED_MM_DEVICES", ()):
                _assert_same_forward_and_backward(
                    self, layer, reference, self.x, **tolerance
                )
        with self.subTest(path="unaligned hidden size takes the loop"):
            layer, reference = _pair(hidden_size=63)
            x = torch.randn(2, 5, 63)
            _assert_same_forward_and_backward(self, layer, reference, x, **tolerance)
        with self.subTest(path="norm_topk_prob off keeps the raw top-k scores"):
            # transformers always renormalizes, so this switch has no reference.
            layer = build_moe(_config(norm_topk_prob=False))
            scores, indices, weights = layer.gate(self.x.reshape(-1, H))
            top = scores.topk(K, dim=-1)
            self.assertTrue(torch.equal(indices, top.indices))
            torch.testing.assert_close(weights, top.values)

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
    def test_cuda_bf16_matches_cpu_fp32(self):
        # The routed path alone: the shared expert is plain Linear, no kernel to check.
        layer = build_moe(_config(shared_expert_intermediate_size=0))
        cuda = build_moe(_config(shared_expert_intermediate_size=0))
        cuda.load_state_dict(layer.state_dict())
        cuda = cuda.to(device="cuda", dtype=torch.bfloat16)
        out, grads = _forward_and_grads(layer, self.x)
        cuda_out, cuda_grads = _forward_and_grads(
            cuda, self.x.to("cuda", torch.bfloat16)
        )
        tolerance = dict(rtol=2e-2, atol=5e-3)
        torch.testing.assert_close(
            cuda_out.float().cpu(), out, msg=_tagged("output"), **tolerance
        )
        for name, grad in grads.items():
            torch.testing.assert_close(
                cuda_grads[name].float().cpu(),
                grad,
                msg=_tagged(f"grad of {name}"),
                **tolerance,
            )


class BuildFfnTest(unittest.TestCase):
    def test_build_ffn_switches_on_moe_preset(self):
        config = Qwen3Config(**TINY_DFLASH)
        self.assertIsInstance(build_ffn(config, dense=Qwen3MLP), Qwen3MLP)
        # MoE fields alone keep the dense MLP: the preset is the switch.
        with_fields = Qwen3Config(**TINY_DFLASH, num_experts=8, num_experts_per_tok=1)
        self.assertIsInstance(build_ffn(with_fields, dense=Qwen3MLP), Qwen3MLP)
        moe_config = Qwen3Config(**TINY_DFLASH, **TINY_MOE)
        self.assertIsInstance(build_ffn(moe_config, dense=Qwen3MLP), MoELayer)
        with self.assertRaisesRegex(ValueError, "qwen3_moe"):
            build_ffn(
                Qwen3Config(**TINY_DFLASH, moe_preset="qwen3_moe"), dense=Qwen3MLP
            )
        # DFlash layers build their MLP through it; MoE drafts refuse checkpointing.
        dense_draft = DFlashDraftModel(config)
        self.assertTrue(all(isinstance(l.mlp, Qwen3MLP) for l in dense_draft.layers))
        moe_draft = DFlashDraftModel(moe_config)
        self.assertTrue(all(isinstance(l.mlp, MoELayer) for l in moe_draft.layers))
        with self.assertRaisesRegex(ValueError, "activation checkpointing"):
            moe_draft.gradient_checkpointing_enable()

    def test_from_pretrained_inits_missing_moe_weights_and_keeps_loaded_ones(self):
        model = DFlashDraftModel(Qwen3Config(**TINY_DFLASH, **TINY_MOE))
        state = model.state_dict()
        missing = {"layers.0.mlp.experts.gate_up_proj", "layers.1.mlp.gate.weight"}
        kept = "layers.0.mlp.experts.down_proj"
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(
                tmp, state_dict={k: v for k, v in state.items() if k not in missing}
            )
            loaded = DFlashDraftModel.from_pretrained(tmp).state_dict()
        for key in missing:
            self.assertTrue(torch.isfinite(loaded[key]).all(), key)
            self.assertTrue(0 < loaded[key].std() < 0.1, key)
        torch.testing.assert_close(loaded[kept], state[kept])


class FamilyConfigTest(unittest.TestCase):
    def test_family_reads_the_config_and_rejects_bad_values(self):
        gate = build_moe(_config()).gate
        self.assertEqual(
            (gate.num_experts, gate.top_k, gate.norm_topk_prob), (E, K, True)
        )
        # Qwen3.8's config omits norm_topk_prob; transformers renormalizes, so do we.
        config = _config()
        del config.norm_topk_prob
        self.assertTrue(build_moe(config).gate.norm_topk_prob)
        # Width 0 drops the shared expert: the block is the routed experts alone.
        routed_only = build_moe(_config(shared_expert_intermediate_size=0))
        self.assertIsNone(routed_only.shared_expert)
        self.assertIsNone(routed_only.shared_expert_gate)
        # Bad values are refused at construction.
        with self.assertRaisesRegex(ValueError, "gelu"):
            build_moe(_config(hidden_act="gelu"))
        for top_k in (0, E + 1):
            with self.assertRaises(ValueError):
                TopKRouter(E, H, top_k, norm_topk_prob=True, initializer_range=0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
