import copy
import tempfile
import unittest
from types import SimpleNamespace

import torch
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.algorithms.dspark_kv.providers import resume_contract
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.target_kv import (
    inverse_target_kv_rope,
    target_kv_config,
    validate_target_kv_config,
)
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import DSparkKVTrainStrategy


def config(mode="derope_reproject", norm=True):
    result = Qwen3Config(
        architectures=["DSparkDraftModel"],
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=2,
        num_target_layers=3,
        vocab_size=32,
        layer_types=["full_attention"] * 2,
        block_size=4,
        attention_dropout=0.0,
        attention_bias=False,
        dflash_config={
            "attention_mode": "gqa",
            "projector_type": "dspark",
            "mask_token_id": 31,
            "markov_rank": 4,
            "enable_confidence_head": True,
            "conditioning_source": "target_kv",
            "target_layer_ids": [0, 2],
            "target_kv_heads": 2,
            "target_kv_head_dim": 4,
            "target_kv_position_mode": mode,
            "target_kv_context_key_norm": norm,
            "target_kv_rope_theta": 1e7,
            "target_kv_rotary_dim": 2,
            "target_kv_rope_layout": "neox",
        },
    )
    result._attn_implementation = "sdpa"
    return result


def rotate_target(raw, positions, rotary_dim=2):
    """Independent scalar-pair reference, not the implementation's rotate_half."""
    result = raw.clone()
    half = rotary_dim // 2
    for pair in range(half):
        angles = positions.to(raw.dtype) / (1e7 ** (2 * pair / rotary_dim))
        c, s = angles.cos()[..., None, None], angles.sin()[..., None, None]
        a, b = raw[:, :, :, 0, :, pair], raw[:, :, :, 0, :, pair + half]
        result[:, :, :, 0, :, pair] = a * c - b * s
        result[:, :, :, 0, :, pair + half] = a * s + b * c
    return result


def wrapper(draft, chunk=1):
    return OnlineDSparkModel(
        draft_model=draft,
        target_lm_head=torch.nn.Linear(16, 32, bias=False).requires_grad_(False),
        target_embed_tokens=torch.nn.Embedding(32, 16).requires_grad_(False),
        mask_token_id=31,
        block_size=4,
        num_anchors=2,
        attention_backend="sdpa",
        objective_chunk_blocks=chunk,
    )


class TargetKVTest(unittest.TestCase):
    def test_partial_inverse_nonzero_batched_positions(self):
        torch.manual_seed(4)
        raw = torch.randn(2, 5, 2, 2, 2, 8, dtype=torch.float64)
        positions = torch.tensor([[7, 8, 31, 1024, 8191], [0, 255, 8192, 16384, 32768]])
        cached = rotate_target(raw, positions, rotary_dim=4)
        restored = inverse_target_kv_rope(
            cached, positions, rope_theta=1e7, rotary_dim=4
        )
        torch.testing.assert_close(restored, raw, rtol=1e-11, atol=1e-11)
        torch.testing.assert_close(
            restored[:, :, :, 1], cached[:, :, :, 1], rtol=0, atol=0
        )
        torch.testing.assert_close(
            restored[:, :, :, 0, :, 4:], cached[:, :, :, 0, :, 4:], rtol=0, atol=0
        )
        with torch.autocast("cpu", dtype=torch.bfloat16):
            bf16 = inverse_target_kv_rope(
                cached.bfloat16(), positions, rope_theta=1e7, rotary_dim=4
            )
        self.assertEqual(bf16.dtype, torch.bfloat16)
        torch.testing.assert_close(bf16.float(), raw.float(), rtol=0.03, atol=0.02)

    def test_invalid_geometry_is_rejected(self):
        for field, value in (
            ("target_kv_position_mode", "other"),
            ("target_kv_context_key_norm", "true"),
            ("target_kv_heads", True),
            ("target_kv_rotary_dim", 3),
            ("target_kv_rotary_dim", 6),
            ("target_kv_rope_theta", float("nan")),
            ("target_kv_rope_theta", True),
            ("target_kv_rope_layout", "interleaved"),
            ("target_layer_ids", [0, 0]),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                cfg = config()
                cfg.dflash_config[field] = value
                target_kv_config(cfg)
        with self.assertRaises(ValueError):
            target_kv_config(config("cached", True))
        kv = torch.randn(2, 3, 2, 2, 2, 4)
        for positions in (torch.zeros(2, 3), torch.zeros(1, 3, dtype=torch.long)):
            with self.assertRaises(ValueError):
                inverse_target_kv_rope(kv, positions, rope_theta=1e7, rotary_dim=2)

    def test_actual_target_config_is_checked(self):
        target = SimpleNamespace(
            model_type="qwen3_5_text",
            num_key_value_heads=2,
            head_dim=4,
            hidden_size=16,
            vocab_size=32,
            layer_types=["full_attention", "linear_attention", "full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 1e7,
                "partial_rotary_factor": 0.5,
            },
        )
        validate_target_kv_config(config(), SimpleNamespace(text_config=target))
        for field, value in (
            ("head_dim", 8),
            ("num_key_value_heads", 1),
            ("layer_types", ["linear_attention"] * 3),
            ("model_type", "qwen3"),
            ("hidden_size", 32),
            ("rope_parameters", {"rope_type": "linear", "rope_theta": 1e7}),
            ("rope_parameters", {"rope_type": "default", "rope_theta": 1e4}),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                altered = copy.deepcopy(target)
                setattr(altered, field, value)
                validate_target_kv_config(config(), altered)

    def test_controls_have_identical_parameters_and_rng(self):
        states, rng = [], []
        for mode, norm in (
            ("cached", False),
            ("derope_reproject", False),
            ("derope_reproject", True),
        ):
            torch.manual_seed(42)
            states.append(DSparkDraftModel(config(mode, norm)).state_dict())
            rng.append(torch.get_rng_state())
        for state, random_state in zip(states[1:], rng[1:]):
            self.assertEqual(state.keys(), states[0].keys())
            for name in state:
                torch.testing.assert_close(state[name], states[0][name], rtol=0, atol=0)
            self.assertTrue(torch.equal(random_state, rng[0]))

    def test_projection_then_norm_then_draft_rope(self):
        torch.manual_seed(7)
        positions = torch.tensor([[31, 32, 33], [8, 9, 10]])
        raw = torch.randn(2, 3, 2, 2, 2, 4)
        noise = torch.randn(2, 2, 16)
        for norm in (False, True):
            model = DSparkDraftModel(config(norm=norm))
            attention = model.layers[0].self_attn
            context_rope = model.rotary_emb(noise, positions)
            draft_rope = model.rotary_emb(noise, positions[:, -2:] + 3)
            _, keys, values = attention._compute_qkv(
                noise,
                None,
                draft_rope,
                target_kv=raw,
                target_position_embeddings=context_rope,
            )
            projected = attention.target_k_proj(raw[:, :, :, 0].flatten(2)).view(
                2, 3, 2, 4
            )
            if norm:
                projected = attention.k_norm(projected)
            expected = projected.clone()
            c, s = context_rope
            for pair in range(2):
                a, b = projected[..., pair], projected[..., pair + 2]
                expected[..., pair] = a * c[..., pair, None] - b * s[..., pair, None]
                expected[..., pair + 2] = (
                    b * c[..., pair, None] + a * s[..., pair, None]
                )
            torch.testing.assert_close(keys[:, :, :3], expected.transpose(1, 2))
            expected_v = (
                attention.target_v_proj(raw[:, :, :, 1].flatten(2))
                .view(2, 3, 2, 4)
                .transpose(1, 2)
            )
            torch.testing.assert_close(values[:, :, :3], expected_v, rtol=0, atol=0)

    def test_absolute_translation_and_masked_context(self):
        torch.manual_seed(11)
        model = DSparkDraftModel(config()).eval()
        raw = torch.randn(2, 6, 2, 2, 2, 4)
        context = torch.tensor([[17, 18, 19, 20, 21, 22], [31, 32, 33, 34, 35, 36]])
        positions = torch.cat((context, context[:, [4, 5, 2, 3]]), dim=1)
        noise = torch.randn(2, 4, 16)
        mask = torch.ones(2, 1, 4, 10, dtype=torch.bool)
        mask[..., 3:6] = False

        def evaluate(kv, shift=0):
            return model(
                position_ids=positions + shift,
                noise_embedding=noise,
                target_kv=kv,
                attention_mask=mask,
            )

        cached = rotate_target(raw, context)
        baseline = evaluate(cached)
        shifted = evaluate(rotate_target(raw, context + 256), 256)
        torch.testing.assert_close(shifted, baseline, rtol=2e-4, atol=2e-5)
        cached[:, 3:] *= 100
        torch.testing.assert_close(evaluate(cached), baseline, rtol=0, atol=0)

    def test_full_objective_backward_and_supervision_only_hidden(self):
        for chunk in (0, 1):
            torch.manual_seed(17)
            draft = DSparkDraftModel(config())
            if chunk:
                draft.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            model = wrapper(draft, chunk)
            tensors = {
                "input_ids": torch.randint(0, 31, (2, 7)),
                "loss_mask": torch.ones(2, 7),
                "target_kv": torch.randn(2, 7, 2, 2, 2, 4, requires_grad=True),
                "target_last_hidden_states": torch.randn(2, 7, 16, requires_grad=True),
            }
            batch = TrainBatch(
                sample_ids=["a", "b"], strategy="dspark_kv", tensors=tensors
            )
            result = DSparkKVTrainStrategy(model).forward_loss(batch)
            self.assertTrue(torch.isfinite(result.loss))
            result.loss.backward()
            for name, parameter in draft.named_parameters():
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            for value in (tensors["target_kv"], tensors["target_last_hidden_states"]):
                self.assertIsNone(value.grad)
            self.assertTrue(
                all(
                    not p.requires_grad and p.grad is None
                    for p in model.lm_head.parameters()
                )
            )
            self.assertTrue(
                all(
                    not p.requires_grad and p.grad is None
                    for p in model.embed_tokens.parameters()
                )
            )
            with self.assertRaisesRegex(ValueError, "forbids target_hidden"):
                draft(
                    position_ids=torch.zeros(2, 8, dtype=torch.long),
                    noise_embedding=torch.randn(2, 1, 16),
                    target_kv=tensors["target_kv"],
                    target_hidden=tensors["target_last_hidden_states"],
                )

    def test_config_roundtrip_resume_contract_and_generation_guard(self):
        contracts = []
        for mode, norm in (
            ("cached", False),
            ("derope_reproject", False),
            ("derope_reproject", True),
        ):
            draft = DSparkDraftModel(config(mode, norm))
            contracts.append(resume_contract(None, draft, wrapper(draft)))
            with tempfile.TemporaryDirectory() as path:
                draft.config.save_pretrained(path)
                loaded = Qwen3Config.from_pretrained(path)
                self.assertEqual(target_kv_config(loaded), draft.target_kv_config)
            with self.assertRaisesRegex(NotImplementedError, "KV-aware"):
                draft.spec_generate(None, None, 1, [], 0)
        self.assertNotEqual(contracts[0], contracts[1])
        self.assertNotEqual(contracts[1], contracts[2])


if __name__ == "__main__":
    unittest.main()
