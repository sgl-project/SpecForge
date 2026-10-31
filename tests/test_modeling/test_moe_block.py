# coding=utf-8
"""The MoE block: Qwen3.5-MoE and DeepSeek-V4 parity, build_ffn, the families' config reading."""

import tempfile
import unittest
from unittest import mock

import torch
from transformers import Qwen3Config
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4SparseMoeBlock,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock,
)

from specforge.modeling.draft import DFlashDraftModel
from specforge.modeling.draft.layers.ffn.moe import MoELayer, build_ffn
from specforge.modeling.draft.layers.ffn.moe import experts as experts_module
from specforge.modeling.draft.layers.ffn.moe.families import deepseek_v4
from specforge.modeling.draft.layers.ffn.moe.families.qwen3_5_moe import build_moe
from specforge.modeling.draft.layers.ffn.moe.router import Router, TopKRouter

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


def _v4_config(**overrides):
    """A tiny DeepSeek-V4 config: one MoE layer, the expert bias on."""
    values = dict(
        hidden_size=H,
        moe_intermediate_size=I,
        n_routed_experts=E,
        num_experts_per_tok=K,
        n_shared_experts=1,
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        num_hidden_layers=1,
        mlp_layer_types=["moe"],
        dflash_config={"moe_router_enable_expert_bias": True},
    )
    return DeepseekV4Config(**{**values, **overrides})


def _v4_key(name):
    """Our state-dict names as transformers' DeepseekV4SparseMoeBlock spells them."""
    name = name.replace("gate.expert_bias", "gate.e_score_correction_bias")
    return name.replace("shared_expert.", "shared_experts.")


def _v4_pair(**overrides):
    """Our block and transformers' block, same weights, a non-zero expert bias."""
    config = _v4_config(**overrides)
    layer = deepseek_v4.build_moe(config)
    with torch.no_grad():
        layer.gate.expert_bias.copy_(torch.linspace(-0.2, 0.2, E))
    reference = DeepseekV4SparseMoeBlock(config, layer_idx=0)
    reference.load_state_dict({_v4_key(k): v for k, v in layer.state_dict().items()})
    return layer, reference


def _forward_and_grads(module, x):
    x = x.clone().requires_grad_()
    out = module(x)
    params = dict(module.named_parameters())
    grads = torch.autograd.grad(out.square().sum(), [x, *params.values()])
    return out, dict(zip(["x", *params], grads))


def _tagged(tag):
    return lambda message: f"{tag}: {message}"


def _assert_same_forward_and_backward(
    test, module, reference, x, rename=lambda name: name, **tolerance
):
    """``rename`` maps our parameter names to the reference's."""
    out, grads = _forward_and_grads(module, x)
    ref_out, ref_grads = _forward_and_grads(reference, x)
    torch.testing.assert_close(out, ref_out, msg=_tagged("output"), **tolerance)
    test.assertEqual({rename(name) for name in grads}, set(ref_grads))
    for name, grad in grads.items():
        torch.testing.assert_close(
            grad, ref_grads[rename(name)], msg=_tagged(f"grad of {name}"), **tolerance
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
        builders = {
            "qwen3_5_moe": lambda: build_moe(
                _config(shared_expert_intermediate_size=0)
            ),
            "deepseek_v4": lambda: deepseek_v4.build_moe(
                _v4_config(n_shared_experts=0)
            ),
        }
        for family, build in builders.items():
            with self.subTest(family=family):
                layer, cuda = build(), build()
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
    def test_build_ffn_switches_on_moe_preset_and_ffn_layer_types(self):
        config = Qwen3Config(**TINY_DFLASH)
        self.assertIsInstance(build_ffn(config, 0, dense=Qwen3MLP), Qwen3MLP)
        # MoE fields alone keep the dense MLP: the preset is the switch.
        with_fields = Qwen3Config(**TINY_DFLASH, num_experts=8, num_experts_per_tok=1)
        self.assertIsInstance(build_ffn(with_fields, 0, dense=Qwen3MLP), Qwen3MLP)
        moe_config = Qwen3Config(**TINY_DFLASH, **TINY_MOE)
        self.assertIsInstance(build_ffn(moe_config, 0, dense=Qwen3MLP), MoELayer)
        with self.assertRaisesRegex(ValueError, "qwen3_moe"):
            build_ffn(
                Qwen3Config(**TINY_DFLASH, moe_preset="qwen3_moe"), 0, dense=Qwen3MLP
            )
        # DFlash layers build their MLP through it; MoE drafts refuse checkpointing.
        dense_draft = DFlashDraftModel(config)
        self.assertTrue(all(isinstance(l.mlp, Qwen3MLP) for l in dense_draft.layers))
        moe_draft = DFlashDraftModel(moe_config)
        self.assertTrue(all(isinstance(l.mlp, MoELayer) for l in moe_draft.layers))
        with self.assertRaisesRegex(ValueError, "activation checkpointing"):
            moe_draft.gradient_checkpointing_enable()
        # ffn_layer_types picks per layer.
        hybrid = Qwen3Config(
            **TINY_DFLASH, **TINY_MOE, ffn_layer_types=["dense", "moe"]
        )
        layers = DFlashDraftModel(hybrid).layers
        self.assertIsInstance(layers[0].mlp, Qwen3MLP)
        self.assertIsInstance(layers[1].mlp, MoELayer)

    def test_from_pretrained_inits_missing_moe_weights_and_keeps_loaded_ones(self):
        knobs = {"dflash_config": {"moe_router_enable_expert_bias": True}}
        model = DFlashDraftModel(Qwen3Config(**TINY_DFLASH, **TINY_MOE, **knobs))
        for layer in model.layers:  # a trained bias, and counts mid-step
            layer.mlp.gate.expert_bias.copy_(torch.linspace(-0.3, 0.3, E))
            layer.mlp.tokens_per_expert.fill_(7)
        state = model.state_dict()
        missing = {"layers.0.mlp.experts.gate_up_proj", "layers.1.mlp.gate.weight"}
        kept = ("layers.0.mlp.experts.down_proj", "layers.1.mlp.gate.expert_bias")
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(
                tmp, state_dict={k: v for k, v in state.items() if k not in missing}
            )
            reloaded = DFlashDraftModel.from_pretrained(tmp)
        loaded = reloaded.state_dict()
        for key in missing:
            self.assertTrue(torch.isfinite(loaded[key]).all(), key)
            self.assertTrue(0 < loaded[key].std() < 0.1, key)
        for key in kept:
            torch.testing.assert_close(loaded[key], state[key])
        # The step's counts are not saved; a loaded draft starts a fresh window.
        for layer in reloaded.layers:
            self.assertTrue(torch.all(layer.mlp.tokens_per_expert == 0))


class FamilyConfigTest(unittest.TestCase):
    def test_family_reads_the_config_and_rejects_bad_values(self):
        knobs = {
            "moe_router_enable_expert_bias": True,
            "moe_router_bias_update_rate": 0.01,
            "moe_router_aux_loss_coef": 0.25,
            "moe_router_aux_loss_type": "microbatch_wise",
        }
        layer = build_moe(_config(dflash_config=knobs))
        gate = layer.gate
        self.assertIsInstance(gate, Router)  # the contract families implement
        self.assertEqual(
            (gate.num_experts, gate.top_k, gate.norm_topk_prob), (E, K, True)
        )
        self.assertIsNotNone(gate.expert_bias)
        self.assertEqual(gate.bias_update_rate, 0.01)
        self.assertEqual(
            (layer.aux_loss_coef, layer.aux_loss_type), (0.25, "microbatch_wise")
        )

        # A family only supplies score(); select and routing_weights come from Router.
        class Uniform(Router):
            def score(self, x):
                return torch.full(
                    (x.shape[0], self.num_experts), 1.0 / self.num_experts
                )

        scores, indices, weights = Uniform(E, K, route_scale=2.0)(torch.zeros(3, H))
        self.assertEqual(
            (tuple(scores.shape), tuple(indices.shape), tuple(weights.shape)),
            ((3, E), (3, K), (3, K)),
        )
        self.assertTrue(torch.all(weights == 2.0 / E))
        # Defaults: no bias, no loss, batch-wise load, renormalized top-k (Qwen3.8 omits it).
        config = _config()
        del config.norm_topk_prob
        plain = build_moe(config)
        self.assertIsNone(plain.gate.expert_bias)
        self.assertEqual(
            (plain.aux_loss_coef, plain.aux_loss_type), (0.0, "batch_wise")
        )
        self.assertTrue(plain.gate.norm_topk_prob)
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
        with self.assertRaisesRegex(ValueError, "aux_loss_type"):
            build_moe(_config(dflash_config={"moe_router_aux_loss_type": "x"}))


class DeepSeekV4BlockTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, 5, H)

    def test_matches_transformers_deepseek_v4_sparse_block(self):
        tolerance = dict(rtol=1e-4, atol=1e-6)
        with self.subTest(path="grouped_mm"):
            layer, reference = _v4_pair()
            _assert_same_forward_and_backward(
                self, layer, reference, self.x, rename=_v4_key, **tolerance
            )
        with self.subTest(path="forced fallback loop"):
            layer, reference = _v4_pair()
            with mock.patch.object(experts_module, "_GROUPED_MM_DEVICES", ()):
                _assert_same_forward_and_backward(
                    self, layer, reference, self.x, rename=_v4_key, **tolerance
                )

    def test_family_reads_the_config_and_rejects_bad_values(self):
        layer = deepseek_v4.build_moe(_v4_config())
        gate = layer.gate
        self.assertIsInstance(gate, TopKRouter)
        self.assertEqual((gate.num_experts, gate.top_k, gate.route_scale), (E, K, 1.5))
        self.assertIs(gate.score_fn, deepseek_v4.sqrtsoftplus_scores)
        self.assertIsNotNone(gate.expert_bias)  # dflash_config turned it on
        # One ungated shared expert of moe_intermediate_size; none when n_shared_experts is 0.
        self.assertEqual(tuple(layer.shared_expert.gate_proj.weight.shape), (I, H))
        self.assertIsNone(layer.shared_expert_gate)
        self.assertIsNone(
            deepseek_v4.build_moe(_v4_config(n_shared_experts=0)).shared_expert
        )
        # V4 always renormalizes; the flag is not read, as in transformers.
        self.assertTrue(
            deepseek_v4.build_moe(_v4_config(norm_topk_prob=False)).gate.norm_topk_prob
        )
        # The clamp: gate capped above, up clamped both ways.
        activation = deepseek_v4.clamped_swiglu(10.0)
        big = torch.tensor([20.0, -20.0])
        torch.testing.assert_close(
            activation(big, big),
            torch.nn.functional.silu(torch.tensor([10.0, -20.0]))
            * torch.tensor([10.0, -10.0]),
        )
        # Bad values are refused at construction.
        with self.assertRaisesRegex(ValueError, "sqrtsoftplus"):
            deepseek_v4.build_moe(_v4_config(scoring_func="sigmoid"))
        with self.assertRaisesRegex(ValueError, "silu"):
            deepseek_v4.build_moe(_v4_config(hidden_act="gelu"))
        # build_ffn reaches the family through moe_preset.
        config = Qwen3Config(
            **TINY_DFLASH,
            moe_preset="deepseek_v4",
            n_routed_experts=E,
            num_experts_per_tok=K,
            moe_intermediate_size=I,
            routed_scaling_factor=1.5,
        )
        self.assertIsInstance(build_ffn(config, 0, dense=Qwen3MLP), MoELayer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
