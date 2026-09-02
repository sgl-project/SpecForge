import copy
import math
import unittest

import torch
from torch.testing import assert_close
from transformers import DynamicCache, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3RotaryEmbedding,
)

from specforge.algorithms.common.dflash_family_model import create_dflash_block_mask
from specforge.modeling.draft.dflash import (
    DFlashDraftModel,
    Qwen3DFlashAttention,
    Qwen3DFlashMLAAttention,
    apply_mla_rope,
)
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


def _position_embeddings(
    config: Qwen3Config,
    key_len: int,
    *,
    position_offset: int = 0,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Model-level rotary embeddings over the MLA partial-RoPE slice."""

    rope_config = copy.deepcopy(config)
    rope_config.head_dim = config.qk_rope_head_dim
    rotary = Qwen3RotaryEmbedding(rope_config).to(device)
    position_ids = torch.arange(
        position_offset,
        position_offset + key_len,
        device=device,
    ).unsqueeze(0)
    reference = torch.zeros((1, key_len, 1), device=device, dtype=dtype)
    return rotary(reference, position_ids)


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
    position_embeddings = _position_embeddings(
        attention.config,
        key_len,
        position_offset=position_offset,
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    return attention(
        hidden_states=hidden_states,
        target_hidden=target_hidden,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
    )


class TestDFlashMLASelection(unittest.TestCase):
    def test_dflash_selects_mla_from_config(self):
        model = DFlashDraftModel(_mla_config())

        self.assertEqual(model.attention_mode, "mla")
        self.assertIsInstance(model.layers[0].self_attn, Qwen3DFlashMLAAttention)
        # Omitted rope_interleave defaults to the DeepSeek interleaved layout.
        self.assertTrue(model.layers[0].self_attn.rope_interleaved)
        # The model-level rotary rotates only the partial-RoPE slice.
        cos, _ = model.rotary_emb(torch.zeros(1, 2, 1), torch.arange(2).unsqueeze(0))
        self.assertEqual(cos.shape[-1], model.config.qk_rope_head_dim)

    def test_omitting_attention_mode_preserves_gqa(self):
        config = _mla_config()
        del config.dflash_config["attention_mode"]
        model = DFlashDraftModel(config)

        self.assertEqual(model.attention_mode, "gqa")
        self.assertIsInstance(model.layers[0].self_attn, Qwen3DFlashAttention)
        cos, _ = model.rotary_emb(torch.zeros(1, 2, 1), torch.arange(2).unsqueeze(0))
        self.assertEqual(cos.shape[-1], config.head_dim)

    def test_dspark_uses_the_same_generic_mla_backbone(self):
        config = _mla_config(architecture="DSparkDraftModel")
        # MLA carries its own head geometry, so DSpark's GQA/MHA head-count
        # policy must not apply: 3 does not divide 4.
        config.num_key_value_heads = 3
        model = DSparkDraftModel(config)

        self.assertIsInstance(model.layers[0].self_attn, Qwen3DFlashMLAAttention)
        self.assertEqual(model.config.dflash_config["attention_mode"], "mla")
        self.assertIsNotNone(model.markov_head)
        self.assertIsNotNone(model.confidence_head)

    def test_direct_query_projection_is_supported(self):
        model = DFlashDraftModel(_mla_config(q_lora_rank=None))
        attention = model.layers[0].self_attn

        self.assertTrue(hasattr(attention, "q_proj"))
        self.assertFalse(hasattr(attention, "q_a_proj"))

    def test_projection_biases_match_deepseek_mla(self):
        config = _mla_config()
        config.attention_bias = True
        attention = DFlashDraftModel(config).layers[0].self_attn

        self.assertIsNotNone(attention.q_a_proj.bias)
        self.assertIsNone(attention.q_b_proj.bias)
        self.assertIsNotNone(attention.kv_a_proj_with_mqa.bias)
        self.assertIsNone(attention.kv_b_proj.bias)
        self.assertIsNotNone(attention.o_proj.bias)

        direct_config = _mla_config(q_lora_rank=None)
        direct_config.attention_bias = True
        direct_attention = DFlashDraftModel(direct_config).layers[0].self_attn
        self.assertIsNone(direct_attention.q_proj.bias)

    def test_deepseek_yarn_scales_attention_logits(self):
        config = _mla_config()
        config.rope_parameters = {
            "rope_type": "yarn",
            "rope_theta": 10_000.0,
            "factor": 40.0,
            "original_max_position_embeddings": 4_096,
            "mscale": 0.707,
            "mscale_all_dim": 0.707,
        }
        attention = DFlashDraftModel(config).layers[0].self_attn

        mscale = 0.1 * 0.707 * math.log(40.0) + 1.0
        qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        expected = mscale**2 / math.sqrt(qk_head_dim)
        self.assertAlmostEqual(attention.scaling, expected)

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
            "o_proj",
        ):
            parameter = getattr(attention, name).weight
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


class TestDFlashMLAAttention(unittest.TestCase):
    def test_rope_conventions_match_independent_reference_values(self):
        hidden = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        cos = torch.tensor([[[0.8, 0.6, 0.8, 0.6]]])
        sin = torch.tensor([[[0.6, 0.8, 0.6, 0.8]]])
        expected = {
            True: torch.tensor([[[[-0.4, 2.2, -1.4, 4.8]]]]),
            False: torch.tensor([[[[-1.0, -2.0, 3.0, 4.0]]]]),
        }

        for interleaved, reference in expected.items():
            with self.subTest(interleaved=interleaved):
                actual = apply_mla_rope(
                    hidden,
                    cos,
                    sin,
                    interleaved=interleaved,
                )
                assert_close(actual, reference)

    def test_query_rope_uses_the_suffix_positions(self):
        torch.manual_seed(17)
        config = _mla_config(q_lora_rank=None)
        attention = Qwen3DFlashMLAAttention(
            config,
            layer_idx=0,
            kernels=DEFAULT_DFLASH_KERNELS,
        ).eval()
        hidden_states = torch.randn(1, 2, config.hidden_size)
        target_hidden = torch.randn(1, 3, config.hidden_size)
        cos, sin = _position_embeddings(config, key_len=5)

        query, _, _ = attention._compute_qkv(
            hidden_states,
            target_hidden,
            (cos, sin),
        )
        raw_query = attention.q_proj(hidden_states).view(
            1,
            2,
            attention.num_heads,
            attention.qk_head_dim,
        )
        raw_query = raw_query.transpose(1, 2)
        raw_query_rope = raw_query[..., attention.qk_nope_head_dim :]
        expected_rope = apply_mla_rope(
            raw_query_rope,
            cos[:, -hidden_states.shape[1] :],
            sin[:, -hidden_states.shape[1] :],
            interleaved=True,
        )

        assert_close(query[..., attention.qk_nope_head_dim :], expected_rope)

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
        torch.manual_seed(13)
        attention = Qwen3DFlashMLAAttention(
            _mla_config(),
            layer_idx=0,
            kernels=DEFAULT_DFLASH_KERNELS,
        ).eval()
        cache = DynamicCache()
        first_hidden = torch.randn(1, 2, 24)
        first_target = torch.randn(1, 3, 24)
        second_hidden = torch.randn(1, 2, 24)
        second_target = torch.randn(1, 1, 24)

        first_output, _ = _attention_forward(
            attention,
            first_hidden,
            first_target,
            None,
            past_key_values=cache,
        )
        self.assertEqual(first_output.shape, (1, 2, 24))
        self.assertEqual(cache.get_seq_length(), 5)

        second_output, _ = _attention_forward(
            attention,
            second_hidden,
            second_target,
            None,
            past_key_values=cache,
            position_offset=5,
        )
        self.assertEqual(second_output.shape, (1, 2, 24))
        self.assertEqual(cache.get_seq_length(), 8)

        full_target = torch.cat(
            [first_target, first_hidden, second_target],
            dim=1,
        )
        uncached_output, _ = _attention_forward(
            attention,
            second_hidden,
            full_target,
            None,
        )
        assert_close(second_output, uncached_output, rtol=1e-5, atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_flex_attention_forward_backward_cuda(self):
        config = _mla_config(implementation="flex_attention")
        # CUDA FlexAttention requires both QK and V head dimensions >= 16.
        config.qk_nope_head_dim = 12
        config.v_head_dim = 24
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


class TestDFlashMLASpecGenerate(unittest.TestCase):
    def test_spec_generate_decode_smoke(self):
        torch.manual_seed(3)
        target_config = Qwen3Config(
            hidden_size=24,
            intermediate_size=48,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            head_dim=6,
            max_position_embeddings=128,
            vocab_size=64,
            tie_word_embeddings=False,
        )
        target_config._attn_implementation = "sdpa"
        target = Qwen3ForCausalLM(target_config).eval()

        config = _mla_config()
        config.dflash_config["mask_token_id"] = 0
        model = DFlashDraftModel(config).eval()

        input_ids = torch.randint(1, 64, (1, 6))
        output_ids = model.spec_generate(
            target,
            input_ids,
            max_new_tokens=8,
            stop_token_ids=None,
            temperature=0.0,
        )

        self.assertEqual(output_ids.shape[0], 1)
        self.assertLessEqual(output_ids.shape[1], input_ids.shape[1] + 8)
        self.assertTrue(torch.equal(output_ids[:, :6], input_ids))


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

    def test_rejects_non_boolean_rope_interleave(self):
        # A JSON null or string must fail loudly instead of silently flipping
        # the rotation convention.
        for value in (None, "false"):
            with self.subTest(value=value):
                config = _mla_config()
                config.rope_interleave = value
                with self.assertRaisesRegex(ValueError, "rope_interleave"):
                    DFlashDraftModel(config)

    def test_rope_interleave_false_selects_neox_rotation(self):
        config = _mla_config()
        config.rope_interleave = False

        attention = DFlashDraftModel(config).layers[0].self_attn

        self.assertFalse(attention.rope_interleaved)

    def test_mha_requires_equal_head_counts(self):
        config = _mla_config()
        config.dflash_config["attention_mode"] = "mha"
        self.assertNotEqual(config.num_key_value_heads, config.num_attention_heads)

        with self.assertRaisesRegex(ValueError, "num_key_value_heads"):
            DFlashDraftModel(config)

        config.num_key_value_heads = config.num_attention_heads
        model = DFlashDraftModel(config)
        self.assertIsInstance(model.layers[0].self_attn, Qwen3DFlashAttention)
        self.assertEqual(model.layers[0].self_attn.num_key_value_groups, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
