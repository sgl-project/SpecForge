import copy
import unittest

import torch
from torch.testing import assert_close
from transformers import DynamicCache, Qwen3Config

from specforge.algorithms.common.dflash_family_model import create_dflash_block_mask
from specforge.modeling.draft.dflash import DFlashDraftModel, Qwen3DFlashMLAAttention
from specforge.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS
from specforge.modeling.draft.dspark import DSparkDraftModel


def _mla_config(
    *,
    architecture: str = "DFlashDraftModel",
    implementation: str = "sdpa",
    q_lora_rank: int | None = 12,
) -> Qwen3Config:
    dflash_config = {
        "attention_mode": "mla",
        "target_layer_ids": [1],
        "mla_rope_interleaved": True,
        "mla_use_output_gate": True,
    }
    if architecture == "DSparkDraftModel":
        dflash_config.update(
            {
                "projector_type": "dspark",
                "markov_rank": 4,
                "enable_confidence_head": True,
            }
        )
    config = Qwen3Config(
        architectures=[architecture],
        block_size=3,
        hidden_size=24,
        intermediate_size=48,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_hidden_layers=1,
        num_target_layers=4,
        head_dim=6,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=6,
        max_position_embeddings=64,
        vocab_size=64,
        layer_types=["full_attention"],
        attention_bias=False,
        attention_dropout=0.0,
        dflash_config=dflash_config,
    )
    config._attn_implementation = implementation
    return config


def _attention_forward(
    attention: Qwen3DFlashMLAAttention,
    hidden_states: torch.Tensor,
    target_hidden: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    past_key_values: DynamicCache | None = None,
    position_offset: int = 0,
):
    key_len = target_hidden.shape[1] + hidden_states.shape[1]
    position_ids = torch.arange(
        position_offset,
        position_offset + key_len,
        device=hidden_states.device,
    ).unsqueeze(0)
    return attention(
        hidden_states=hidden_states,
        target_hidden=target_hidden,
        position_embeddings=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
    )


class TestDFlashMLASelection(unittest.TestCase):
    def test_dflash_selects_mla_from_config(self):
        model = DFlashDraftModel(_mla_config())

        self.assertEqual(model.attention_mode, "mla")
        self.assertIsInstance(model.layers[0].self_attn, Qwen3DFlashMLAAttention)
        self.assertTrue(model.layers[0].self_attn.rope_interleaved)
        self.assertTrue(model.layers[0].self_attn.use_output_gate)

    def test_dspark_uses_the_same_generic_mla_backbone(self):
        model = DSparkDraftModel(_mla_config(architecture="DSparkDraftModel"))

        self.assertIsInstance(model.layers[0].self_attn, Qwen3DFlashMLAAttention)
        self.assertEqual(model.config.dflash_config["attention_mode"], "mla")
        self.assertIsNotNone(model.markov_head)
        self.assertIsNotNone(model.confidence_head)

    def test_direct_query_projection_is_supported(self):
        model = DFlashDraftModel(_mla_config(q_lora_rank=None))
        attention = model.layers[0].self_attn

        self.assertTrue(hasattr(attention, "q_proj"))
        self.assertFalse(hasattr(attention, "q_a_proj"))

    def test_forward_backward_is_finite(self):
        torch.manual_seed(7)
        model = DFlashDraftModel(_mla_config()).train()
        hidden_states = torch.randn(2, 3, 24, requires_grad=True)
        target_hidden = torch.randn(2, 5, 24)
        position_ids = torch.arange(8).expand(2, -1)
        output = model(
            position_ids=position_ids,
            noise_embedding=hidden_states,
            target_hidden=target_hidden,
            attention_mask=torch.ones(2, 1, 3, 8, dtype=torch.bool),
        )

        self.assertEqual(output.shape, (2, 3, 24))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertIsNotNone(hidden_states.grad)
        self.assertTrue(torch.isfinite(hidden_states.grad).all())
        attention = model.layers[0].self_attn
        for name in (
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "g_proj",
            "o_proj",
        ):
            parameter = getattr(attention, name).weight
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


class TestDFlashMLAAttention(unittest.TestCase):
    def test_eager_matches_sdpa_with_boolean_mask(self):
        torch.manual_seed(11)
        eager_config = _mla_config(implementation="eager")
        sdpa_config = _mla_config(implementation="sdpa")
        eager = Qwen3DFlashMLAAttention(
            eager_config,
            layer_idx=0,
            kernels=DEFAULT_DFLASH_KERNELS,
        ).eval()
        sdpa = Qwen3DFlashMLAAttention(
            sdpa_config,
            layer_idx=0,
            kernels=DEFAULT_DFLASH_KERNELS,
        ).eval()
        sdpa.load_state_dict(eager.state_dict())

        eager_hidden = torch.randn(1, 3, 24, requires_grad=True)
        eager_target = torch.randn(1, 4, 24, requires_grad=True)
        sdpa_hidden = eager_hidden.detach().clone().requires_grad_(True)
        sdpa_target = eager_target.detach().clone().requires_grad_(True)
        mask = torch.tensor(
            [
                [
                    [
                        [True, True, True, True, True, False, False],
                        [True, True, False, True, True, True, False],
                        [False, False, False, False, False, False, False],
                    ]
                ]
            ]
        )

        eager_output, eager_weights = _attention_forward(
            eager,
            eager_hidden,
            eager_target,
            mask,
        )
        sdpa_output, _ = _attention_forward(
            sdpa,
            sdpa_hidden,
            sdpa_target,
            mask,
        )

        assert_close(eager_output, sdpa_output, rtol=1e-5, atol=1e-6)
        self.assertIsNotNone(eager_weights)
        self.assertEqual(eager_output[:, -1].abs().sum().item(), 0.0)
        forbidden = eager_weights.masked_select(~mask.expand_as(eager_weights))
        assert_close(forbidden, torch.zeros_like(forbidden), rtol=0, atol=0)

        output_grad = torch.randn_like(eager_output)
        eager_grads = torch.autograd.grad(
            (eager_output * output_grad).sum(),
            (eager_hidden, eager_target),
        )
        sdpa_grads = torch.autograd.grad(
            (sdpa_output * output_grad).sum(),
            (sdpa_hidden, sdpa_target),
        )
        for eager_grad, sdpa_grad in zip(eager_grads, sdpa_grads):
            assert_close(eager_grad, sdpa_grad, rtol=1e-5, atol=1e-6)

    def test_dynamic_cache_accepts_different_key_and_value_dims(self):
        attention = Qwen3DFlashMLAAttention(
            _mla_config(),
            layer_idx=0,
            kernels=DEFAULT_DFLASH_KERNELS,
        ).eval()
        cache = DynamicCache()

        first_output, _ = _attention_forward(
            attention,
            torch.randn(1, 2, 24),
            torch.randn(1, 3, 24),
            None,
            past_key_values=cache,
        )
        self.assertEqual(first_output.shape, (1, 2, 24))
        self.assertEqual(cache.get_seq_length(), 5)

        second_output, _ = _attention_forward(
            attention,
            torch.randn(1, 2, 24),
            torch.randn(1, 1, 24),
            None,
            past_key_values=cache,
            position_offset=5,
        )
        self.assertEqual(second_output.shape, (1, 2, 24))
        self.assertEqual(cache.get_seq_length(), 8)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_flex_attention_forward_backward_cuda(self):
        config = _mla_config(implementation="flex_attention")
        # CUDA FlexAttention requires both QK and V head dimensions >= 16.
        config.qk_nope_head_dim = 12
        config.v_head_dim = 16
        attention = (
            Qwen3DFlashMLAAttention(
                config,
                layer_idx=0,
                kernels=DEFAULT_DFLASH_KERNELS,
            )
            .to(device="cuda", dtype=torch.bfloat16)
            .train()
        )
        hidden_states = torch.randn(
            1,
            6,
            24,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        target_hidden = torch.randn(
            1,
            6,
            24,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        block_mask = create_dflash_block_mask(
            anchor_positions=torch.tensor([[2, 5]], device="cuda"),
            block_keep_mask=torch.tensor([[True, True]], device="cuda"),
            S=6,
            block_size=3,
            device=torch.device("cuda"),
        )

        output, _ = _attention_forward(
            attention,
            hidden_states,
            target_hidden,
            block_mask,
        )
        self.assertEqual(output.shape, (1, 6, 24))
        self.assertTrue(torch.isfinite(output).all())
        output.float().square().mean().backward()
        for tensor in (hidden_states, target_hidden):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())


class TestDFlashMLAConfigValidation(unittest.TestCase):
    def test_rejects_missing_or_invalid_dimensions(self):
        cases = {
            "kv_lora_rank": None,
            "q_lora_rank": 0,
            "qk_nope_head_dim": -1,
            "qk_rope_head_dim": 3,
            "v_head_dim": 0,
        }
        for field, value in cases.items():
            with self.subTest(field=field, value=value):
                config = copy.deepcopy(_mla_config())
                setattr(config, field, value)
                with self.assertRaisesRegex(ValueError, field):
                    DFlashDraftModel(config)

    def test_rejects_unknown_attention_mode(self):
        config = _mla_config()
        config.dflash_config["attention_mode"] = "latent-ish"

        with self.assertRaisesRegex(ValueError, "attention_mode"):
            DFlashDraftModel(config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
