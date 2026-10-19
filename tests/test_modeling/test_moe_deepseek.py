# coding=utf-8
"""DeepSeek-V4 MoE family: the deepseek_v4 preset against transformers'
DeepSeek-V4 MoE block, the deepseek checkpoint layout, the FP4/FP8 checkpoint
dtypes against transformers' dequantizer, the serving config and the target
reader."""

import json
import os
import tempfile
import unittest

import torch
from safetensors.torch import save_file
from transformers.integrations.finegrained_fp8 import Fp8Dequantize
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as hf
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

from specforge.modeling.draft.moe import (
    MoELayer,
    from_checkpoint_state_dict,
    resolve_moe_config,
    serving_config,
    to_checkpoint_state_dict,
    to_serving_state_dict,
)
from specforge.modeling.draft.moe.families import deepseek

H, E, K, I = 32, 8, 2, 32


def _moe(**overrides):
    payload = dict(
        hidden_size=H,
        moe_preset="deepseek_v4",
        n_routed_experts=E,
        num_experts_per_tok=K,
        moe_intermediate_size=I,
        dflash_config={},
    )
    payload.update(overrides)
    return payload


def _layer(seed=0, **overrides):
    torch.manual_seed(seed)
    layer = MoELayer(resolve_moe_config(_moe(**overrides)), H)
    layer.reset_parameters(0.2)
    with torch.no_grad():
        layer.gate.balance.bias.copy_(torch.randn(E) * 0.3)
    return layer


def _hf_block():
    config = DeepseekV4Config(
        hidden_size=H,
        moe_intermediate_size=I,
        n_routed_experts=E,
        num_experts_per_tok=K,
        num_hidden_layers=1,
        mlp_layer_types=["moe"],
    )
    block = hf.DeepseekV4SparseMoeBlock(config, layer_idx=0).float()
    torch.manual_seed(0)
    for parameter in block.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    with torch.no_grad():
        block.gate.e_score_correction_bias.copy_(torch.randn(E) * 0.3)
    return block


def _copy_hf_into(layer, block):
    """transformers' fused [E, 2I, H] gate_up / [E, H, I] down -> module layout."""
    with torch.no_grad():
        layer.gate.weight.copy_(block.gate.weight)
        layer.gate.balance.bias.copy_(block.gate.e_score_correction_bias)
        layer.experts.w1.copy_(block.experts.gate_up_proj[:, :I])
        layer.experts.w3.copy_(block.experts.gate_up_proj[:, I:])
        layer.experts.w2.copy_(block.experts.down_proj)
        shared = layer.shared_experts
        shared.w1.weight.copy_(block.shared_experts.gate_proj.weight)
        shared.w3.weight.copy_(block.shared_experts.up_proj.weight)
        shared.w2.weight.copy_(block.shared_experts.down_proj.weight)


class TestDeepSeekPreset(unittest.TestCase):
    def test_recipe(self):
        cfg = resolve_moe_config(_moe())
        self.assertEqual(
            (cfg.scoring_func, cfg.balance, cfg.shared_expert_gate, cfg.layout),
            ("sqrtsoftplus", "noaux_tc", "none", "deepseek"),
        )
        self.assertEqual((cfg.routed_scaling_factor, cfg.swiglu_limit), (1.5, 10.0))
        self.assertEqual(cfg.shared_expert_intermediate_size, I)
        self.assertFalse(cfg.group_limited)

    def test_matches_transformers_deepseek_v4_block(self):
        block = _hf_block()
        layer = MoELayer(resolve_moe_config(_moe()), H).float().eval()
        _copy_hf_into(layer, block)
        # Large inputs push most SwiGLU pre-activations past the clamp.
        x = torch.randn(2, 6, H) * 8
        with torch.no_grad():
            expected = block(x)
            actual = layer(x)
        self.assertLess(float((actual - expected).abs().max()), 1e-4)
        self.assertLess(float((actual - expected).norm() / expected.norm()), 1e-6)


class TestDeepSeekLayout(unittest.TestCase):
    def test_official_names_and_round_trip(self):
        layer = _layer()
        state = layer.state_dict()
        files = to_checkpoint_state_dict(state, layer)
        expected = {
            "gate.weight",
            "gate.bias",
            "shared_experts.w1.weight",
            "shared_experts.w2.weight",
            "shared_experts.w3.weight",
        } | {f"experts.{i}.w{w}.weight" for i in range(E) for w in (1, 2, 3)}
        self.assertEqual(set(files), expected)
        self.assertEqual(files["gate.bias"].dtype, torch.float32)
        self.assertTrue(torch.equal(files["gate.bias"], state["gate.balance.bias"]))
        self.assertTrue(torch.equal(files["experts.5.w2.weight"], layer.experts.w2[5]))
        back = from_checkpoint_state_dict(files, layer)
        self.assertEqual(set(back), set(state))
        for key in state:
            self.assertTrue(torch.equal(back[key], state[key]), key)

    def test_hash_layers_and_missing_experts_are_rejected(self):
        layer = _layer()
        files = to_checkpoint_state_dict(layer.state_dict(), layer)
        hashed = {**files, "gate.tid2eid": torch.zeros(16, K, dtype=torch.int32)}
        with self.assertRaisesRegex(ValueError, "hash-routed"):
            from_checkpoint_state_dict(hashed, layer)
        del files["experts.3.w1.weight"]
        with self.assertRaisesRegex(KeyError, "missing expert indices"):
            from_checkpoint_state_dict(files, layer)

    def test_router_centering_has_no_file_name(self):
        layer = _layer(dflash_config={"moe_router_center": "sample"}).train()
        layer(torch.randn(4, H) + 1.0)
        layer.apply_pending_update()
        # Training checkpoints keep the centering buffers under their names ...
        files = to_checkpoint_state_dict(layer.state_dict(), layer)
        self.assertIn("gate.input_mean", files)
        # ... but the export fold would produce a logit bias DeepSeek cannot name.
        with self.assertRaisesRegex(ValueError, "selection-only bias"):
            to_serving_state_dict(layer.state_dict(), layer)

    def test_select_experts_keeps_router_rows(self):
        layer = _layer()
        files = to_checkpoint_state_dict(layer.state_dict(), layer)
        picked = deepseek.select_deepseek_experts(files, [6, 1])
        self.assertTrue(
            torch.equal(picked["experts.0.w3.weight"], files["experts.6.w3.weight"])
        )
        self.assertTrue(
            torch.equal(picked["gate.weight"], files["gate.weight"][[6, 1]])
        )
        self.assertTrue(torch.equal(picked["gate.bias"], files["gate.bias"][[6, 1]]))
        self.assertIn("shared_experts.w2.weight", picked)
        self.assertNotIn("experts.2.w1.weight", picked)


class TestDeepSeekServing(unittest.TestCase):
    def test_expert_counts_are_the_only_served_fields(self):
        self.assertEqual(
            serving_config(_moe()),
            {"dspark_n_routed_experts": E, "dspark_num_experts_per_tok": K},
        )

    def test_unservable_recipes_are_refused(self):
        for override in (
            {"scoring_func": "softmax"},
            {"balance": "aux_loss"},
            {"routed_scaling_factor": 1.0},
            {"swiglu_limit": 0.0},
            {"shared_expert_gate": "sigmoid"},
            {"shared_expert_intermediate_size": 2 * I},
            {"n_group": 4, "topk_group": 2},
            {"dflash_config": {"moe_router_center": "sample"}},
            {"moe_router_bias": True},
        ):
            with (
                self.subTest(**override),
                self.assertRaisesRegex(ValueError, "cannot serve"),
            ):
                serving_config(_moe(**override))


def _fp8_reference(weight, scale):
    return Fp8Dequantize(None)._dequantize_one(weight, scale, torch.float32)


class TestCheckpointDtypes(unittest.TestCase):
    def test_fp8_blocks_match_transformers(self):
        torch.manual_seed(0)
        weight = torch.randn(256, 384) * 3
        q, scale = deepseek.quantize_fp8(weight)
        self.assertEqual(tuple(scale.shape), (2, 3))
        self.assertEqual(scale.dtype, torch.float8_e8m0fnu)
        ours = deepseek.dequantize_fp8(q, scale, torch.float32)
        self.assertTrue(torch.equal(ours, _fp8_reference(q, scale)))
        # Dequantized values survive a requantization exactly (the export
        # bundle relies on this for weights it did not change).
        again = deepseek.dequantize_fp8(*deepseek.quantize_fp8(ours), torch.float32)
        self.assertTrue(torch.equal(again, ours))
        with self.assertRaisesRegex(ValueError, "scale"):
            deepseek.dequantize_fp8(q, scale[:1])

    def test_fp4_matches_transformers(self):
        torch.manual_seed(1)
        weight = torch.randn(64, 256)
        packed, scale = deepseek.quantize_fp4(weight)
        self.assertEqual((packed.dtype, tuple(packed.shape)), (torch.int8, (64, 128)))
        self.assertEqual(tuple(scale.shape), (64, 256 // deepseek.FP4_GROUP))
        ours = deepseek.dequantize_fp4(packed, scale, torch.float32)
        self.assertTrue(torch.equal(ours, _fp8_reference(packed, scale)))
        # Values round-trip exactly; the codes need not (+0/-0, and a group
        # whose largest value lands on 3 requantizes with half the scale).
        again = deepseek.dequantize_fp4(*deepseek.quantize_fp4(ours), torch.float32)
        self.assertTrue(torch.equal(again, ours))
        # Nearest grid point: within half the coarsest grid step of the scale.
        step = 2.0 * scale.float().repeat_interleave(deepseek.FP4_GROUP, 1)
        self.assertTrue(bool(((ours - weight).abs() <= step / 2 + 1e-6).all()))


def _write_snapshot(root, layers):
    """A tiny sharded DeepSeek-V4 checkpoint in the official dtypes."""
    config = {"architectures": ["DeepseekV4ForCausalLM"], "n_routed_experts": E}
    with open(os.path.join(root, "config.json"), "w") as stream:
        json.dump(config, stream)
    weight_map = {}
    for shard, tensors in enumerate(layers):
        name = f"model-{shard:05d}.safetensors"
        save_file(tensors, os.path.join(root, name))
        weight_map.update({key: name for key in tensors})
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as stream:
        json.dump({"weight_map": weight_map}, stream)


def _official_layer(prefix, seed):
    torch.manual_seed(seed)
    tensors = {
        prefix + "gate.weight": torch.randn(E, H).to(torch.bfloat16),
        prefix + "gate.bias": torch.randn(E),
    }
    for i in range(E):
        for w, shape in (("w1", (I, H)), ("w3", (I, H)), ("w2", (H, I))):
            q, s = deepseek.quantize_fp4(torch.randn(*shape))
            tensors[f"{prefix}experts.{i}.{w}.weight"] = q
            tensors[f"{prefix}experts.{i}.{w}.scale"] = s
    for w, shape in (("w1", (I, H)), ("w3", (I, H)), ("w2", (H, I))):
        # smaller than one 128x128 block: a single scale covers the tensor
        tensors[f"{prefix}shared_experts.{w}.weight"] = torch.randn(*shape).to(
            torch.float8_e4m3fn
        )
        tensors[f"{prefix}shared_experts.{w}.scale"] = torch.full((1, 1), 0.5).to(
            torch.float8_e8m0fnu
        )
    return tensors


class TestSnapshotReading(unittest.TestCase):
    def test_reads_only_the_prefix_and_dequantizes(self):
        prefix = "layers.3.ffn."
        with tempfile.TemporaryDirectory() as root:
            learned = _official_layer(prefix, seed=3)
            other = {
                "layers.1.ffn.gate.weight": torch.zeros(E, H, dtype=torch.bfloat16)
            }
            _write_snapshot(root, [other, learned])
            self.assertEqual(deepseek.read_config(root)["n_routed_experts"], E)
            raw = dict(deepseek.iter_checkpoint_tensors(root, prefix))
        self.assertEqual(set(raw), set(learned))
        source = deepseek.dequantize_tensors(
            {key[len(prefix) :]: value for key, value in raw.items()}
        )
        self.assertNotIn("experts.0.w1.scale", source)
        self.assertEqual(source["gate.bias"].dtype, torch.float32)
        self.assertTrue(
            torch.equal(
                source["shared_experts.w2.weight"],
                _fp8_reference(
                    learned[prefix + "shared_experts.w2.weight"],
                    learned[prefix + "shared_experts.w2.scale"],
                ).to(torch.bfloat16),
            )
        )


if __name__ == "__main__":
    unittest.main()
