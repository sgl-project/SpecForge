"""Correctness and composition tests for DFlash-family KDA attention."""

from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from unittest import mock

import torch
from torch.testing import assert_close
from transformers import Qwen3Config, Qwen3ForCausalLM

from specforge.modeling.draft.dflash import (
    DFlashDraftModel,
    DFlashGenerationOutput,
    Qwen3DFlashAttention,
    Qwen3DFlashAttentionBase,
    Qwen3DFlashKVAttentionBase,
    resolve_dflash_attention_modes,
)
from specforge.modeling.draft.domino import DominoDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.kda import Qwen3DFlashKDAAttention, fla_kda, reference_kda

CUDA_AND_FLA = torch.cuda.is_available() and importlib.util.find_spec("fla") is not None


def _kda_config(
    *,
    architecture: str = "DFlashDraftModel",
    projector_type: str | None = None,
    attention_modes: list[str] | None = None,
    backend: str = "reference",
) -> Qwen3Config:
    config = Qwen3Config(
        hidden_size=24,
        intermediate_size=48,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=3,
        head_dim=6,
        max_position_embeddings=128,
        vocab_size=64,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    config.architectures = [architecture]
    config.num_target_layers = 8
    config.block_size = 2
    config.draft_vocab_size = 64
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    config.use_sliding_window = False
    config.dflash_config = {
        "attention_modes": attention_modes or ["kda", "gqa", "kda"],
        "mask_token_id": 0,
    }
    if projector_type is not None:
        config.dflash_config["projector_type"] = projector_type
    if projector_type == "domino":
        config.dflash_config.update({"emb_dim": 16, "gru_hidden_dim": 12})
    config.linear_attn_config = {
        "head_dim": 6,
        "num_heads": 4,
        "short_conv_kernel_size": 3,
        "use_full_rank_gate": False,
        "gate_lower_bound": -5.0,
        "backend": backend,
    }
    return config


def _attention_forward(
    attention: Qwen3DFlashKDAAttention,
    hidden_states: torch.Tensor,
    *,
    target_hidden: torch.Tensor | None = None,
    past_key_values=None,
):
    batch_size, query_len = hidden_states.shape[:2]
    if target_hidden is None:
        target_hidden = torch.randn(batch_size, 3, hidden_states.shape[-1])
    position_embeddings = (
        torch.ones(batch_size, target_hidden.shape[1] + query_len, 6),
        torch.zeros(batch_size, target_hidden.shape[1] + query_len, 6),
    )
    return attention(
        hidden_states=hidden_states,
        target_hidden=target_hidden,
        position_embeddings=position_embeddings,
        attention_mask=None,
        past_key_values=past_key_values,
    )[0]


class TestKDAComposition(unittest.TestCase):
    def test_hybrid_layers_share_one_attention_contract(self):
        model = DFlashDraftModel(_kda_config())

        self.assertEqual(model.attention_mode, "hybrid")
        self.assertEqual(model.attention_modes, ("kda", "gqa", "kda"))
        self.assertIsInstance(model.layers[0].self_attn, Qwen3DFlashKDAAttention)
        self.assertIsInstance(model.layers[1].self_attn, Qwen3DFlashAttention)
        for layer in model.layers:
            self.assertIsInstance(layer.self_attn, Qwen3DFlashAttentionBase)
        self.assertIsInstance(model.layers[1].self_attn, Qwen3DFlashKVAttentionBase)

    def test_dflash_dspark_and_domino_use_the_same_kda_class(self):
        cases = (
            (DFlashDraftModel, _kda_config()),
            (
                DSparkDraftModel,
                _kda_config(
                    architecture="DSparkDraftModel",
                    projector_type="dspark",
                ),
            ),
            (
                DominoDraftModel,
                _kda_config(
                    architecture="DominoDraftModel",
                    projector_type="domino",
                ),
            ),
        )

        for model_cls, config in cases:
            with self.subTest(model=model_cls.__name__):
                model = model_cls(config)
                self.assertIsInstance(
                    model.layers[0].self_attn, Qwen3DFlashKDAAttention
                )
                self.assertIn("Qwen3DFlashDecoderLayer", model._no_split_modules)

    def test_fsdp_wraps_the_whole_hybrid_decoder_block(self):
        model = DFlashDraftModel(_kda_config())
        block_names = set(model._no_split_modules)

        self.assertEqual(block_names, {"Qwen3DFlashDecoderLayer"})
        self.assertTrue(
            all(
                type(layer).__name__ in block_names
                and isinstance(layer.self_attn, Qwen3DFlashAttentionBase)
                for layer in model.layers
            )
        )

    def test_uniform_attention_mode_remains_backwards_compatible(self):
        config = _kda_config()
        config.dflash_config = {"attention_mode": "gqa", "mask_token_id": 0}

        self.assertEqual(resolve_dflash_attention_modes(config), ("gqa",) * 3)
        model = DFlashDraftModel(config)
        self.assertEqual(model.attention_mode, "gqa")
        self.assertTrue(
            all(
                isinstance(layer.self_attn, Qwen3DFlashAttention)
                for layer in model.layers
            )
        )

    def test_rejects_ambiguous_or_malformed_layer_modes(self):
        cases = {
            "only one": {"attention_mode": "gqa", "attention_modes": ["gqa"] * 3},
            "exactly": {"attention_modes": ["kda", "gqa"]},
            "selected": {"attention_modes": ["kda", "dense", "kda"]},
        }
        for message, dflash_config in cases.items():
            with self.subTest(message=message):
                config = _kda_config()
                config.dflash_config = dflash_config
                with self.assertRaisesRegex(ValueError, message):
                    DFlashDraftModel(config)

    def test_rejects_kda_without_context_injection(self):
        config = _kda_config(attention_modes=["kda"] * 3)

        with self.assertRaisesRegex(ValueError, "inject target context"):
            DFlashDraftModel(config)

    def test_rejects_mixed_context_families(self):
        for modes in (["kda", "gqa", "mla"], ["gqa", "mha", "gqa"]):
            with self.subTest(modes=modes):
                config = _kda_config(attention_modes=modes)
                with self.assertRaisesRegex(ValueError, "one consistent"):
                    DFlashDraftModel(config)

    def test_rejects_invalid_kda_dimensions_backend_and_gate(self):
        cases = (
            ("head_dim", 0, "head_dim"),
            ("short_conv_kernel_size", -1, "short_conv_kernel_size"),
            ("backend", "mystery", "backend"),
            ("gate_lower_bound", 1.0, "gate_lower_bound"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                config = _kda_config()
                config.linear_attn_config[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    DFlashDraftModel(config)


class TestKDAReferenceMath(unittest.TestCase):
    def test_recurrence_matches_independent_scalar_oracle(self):
        q = torch.tensor([[[[2.0]], [[4.0]]]])
        k = torch.tensor([[[[3.0]], [[5.0]]]])
        v = torch.tensor([[[[7.0]], [[11.0]]]])
        raw_gate = torch.zeros_like(q)
        beta = torch.zeros(1, 2, 1)
        A_log = torch.zeros(1)
        dt_bias = torch.zeros(1)

        actual = reference_kda(
            q, k, v, raw_gate, beta, A_log, dt_bias, lower_bound=None
        )

        decay = torch.exp(-torch.nn.functional.softplus(torch.tensor(0.0)))
        state_0 = torch.tensor(0.0) * decay + 1.0 * (7.0 - 0.0) * 0.5
        output_0 = state_0
        state_1 = state_0 * decay
        state_1 = state_1 + 1.0 * (11.0 - state_1) * 0.5
        expected = torch.tensor([[[[output_0]], [[state_1]]]])
        assert_close(actual, expected)

    def test_proposal_blocks_do_not_share_recurrent_or_convolution_state(self):
        torch.manual_seed(3)
        attention = Qwen3DFlashKDAAttention(
            _kda_config(), layer_idx=0, kernels=mock.Mock()
        ).eval()
        hidden_states = torch.randn(1, 4, 24)
        perturbed = hidden_states.clone()
        perturbed[:, :2] += 100

        output = _attention_forward(attention, hidden_states)
        perturbed_output = _attention_forward(attention, perturbed)

        self.assertFalse(torch.equal(output[:, :2], perturbed_output[:, :2]))
        assert_close(output[:, 2:], perturbed_output[:, 2:], rtol=0, atol=0)

    def test_batched_execution_matches_individual_sequences(self):
        torch.manual_seed(5)
        attention = Qwen3DFlashKDAAttention(
            _kda_config(), layer_idx=0, kernels=mock.Mock()
        ).eval()
        hidden_states = torch.randn(2, 4, 24)

        batched = _attention_forward(attention, hidden_states)
        individual = torch.cat(
            [_attention_forward(attention, row.unsqueeze(0)) for row in hidden_states]
        )

        assert_close(batched, individual, rtol=1e-5, atol=1e-6)

    def test_forward_backward_is_finite_and_reaches_all_parameter_groups(self):
        torch.manual_seed(7)
        model = DFlashDraftModel(_kda_config()).train()
        noise_embedding = torch.randn(2, 4, 24, requires_grad=True)
        target_hidden = torch.randn(2, 5, 72, requires_grad=True)
        position_ids = torch.arange(9).expand(2, -1)
        attention_mask = torch.ones(2, 1, 4, 9, dtype=torch.bool)

        output = model(
            position_ids=position_ids,
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
        )

        self.assertEqual(output.shape, (2, 4, 24))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        for tensor in (noise_embedding, target_hidden):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        attention = model.layers[0].self_attn
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "f_a_proj",
            "f_b_proj",
            "b_proj",
            "g_a_proj",
            "g_b_proj",
            "o_proj",
        ):
            parameter = getattr(attention, name).weight
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for name in ("A_log", "dt_bias"):
            parameter = getattr(attention, name)
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_checkpoint_parameter_names_match_serving_layout(self):
        attention = Qwen3DFlashKDAAttention(
            _kda_config(), layer_idx=0, kernels=mock.Mock()
        )

        expected = {
            "q_proj.weight",
            "k_proj.weight",
            "v_proj.weight",
            "q_conv1d.weight",
            "k_conv1d.weight",
            "v_conv1d.weight",
            "A_log",
            "dt_bias",
            "f_a_proj.weight",
            "f_b_proj.weight",
            "b_proj.weight",
            "g_a_proj.weight",
            "g_b_proj.weight",
            "o_norm.weight",
            "o_proj.weight",
        }
        self.assertEqual(set(attention.state_dict()), expected)

    def test_mixed_precision_cast_keeps_fsdp_block_dtype_uniform(self):
        attention = Qwen3DFlashKDAAttention(
            _kda_config(), layer_idx=0, kernels=mock.Mock()
        ).to(dtype=torch.bfloat16)

        self.assertEqual(attention.q_proj.weight.dtype, torch.bfloat16)
        self.assertEqual(attention.A_log.dtype, torch.bfloat16)
        self.assertEqual(attention.dt_bias.dtype, torch.bfloat16)

    def test_kda_leaves_the_shared_dense_attention_cache_untouched(self):
        attention = Qwen3DFlashKDAAttention(
            _kda_config(), layer_idx=0, kernels=mock.Mock()
        )
        cache = mock.Mock()

        output = _attention_forward(
            attention,
            torch.randn(1, 2, 24),
            past_key_values=cache,
        )

        self.assertEqual(output.shape, (1, 2, 24))
        cache.update.assert_not_called()


class TestKDAFLADispatch(unittest.TestCase):
    def test_buckets_dynamic_block_counts_without_changing_outputs(self):
        seen_batch_sizes = []

        def fake_chunk_kda(**kwargs):
            seen_batch_sizes.append(kwargs["q"].shape[0])
            return kwargs["q"], None

        q = torch.randn(3, 2, 2, 3)
        with mock.patch(
            "specforge.modeling.draft.kda._load_fla_chunk_kda",
            return_value=fake_chunk_kda,
        ):
            output = fla_kda(
                q,
                torch.randn_like(q),
                torch.randn_like(q),
                torch.randn_like(q),
                torch.randn(3, 2, 2),
                torch.randn(2),
                torch.randn(6),
                -5.0,
            )

        self.assertEqual(seen_batch_sizes, [4])
        assert_close(output, q)

    def test_splits_independent_blocks_below_cuda_grid_limit(self):
        calls = []

        def fake_chunk_kda(**kwargs):
            calls.append(kwargs["q"].shape[0])
            return kwargs["q"], None

        q = torch.randn(5, 2, 2, 3)
        args = (
            q,
            torch.randn_like(q),
            torch.randn_like(q),
            torch.randn_like(q),
            torch.randn(5, 2, 2),
            torch.randn(2),
            torch.randn(6),
            -5.0,
        )
        with (
            mock.patch(
                "specforge.modeling.draft.kda._load_fla_chunk_kda",
                return_value=fake_chunk_kda,
            ),
            mock.patch("specforge.modeling.draft.kda._CUDA_MAX_GRID_DIM_Z", 5),
        ):
            output = fla_kda(*args)

        self.assertEqual(calls, [2, 2, 1])
        assert_close(output, q)

    def test_casts_fsdp_bf16_gate_parameters_to_kernel_fp32(self):
        seen_dtypes = []

        def fake_chunk_kda(**kwargs):
            seen_dtypes.append((kwargs["A_log"].dtype, kwargs["dt_bias"].dtype))
            return kwargs["q"], None

        q = torch.randn(1, 2, 2, 4, dtype=torch.bfloat16)
        with mock.patch(
            "specforge.modeling.draft.kda._load_fla_chunk_kda",
            return_value=fake_chunk_kda,
        ):
            fla_kda(
                q,
                torch.randn_like(q),
                torch.randn_like(q),
                torch.randn_like(q),
                torch.randn(1, 2, 2, dtype=torch.bfloat16),
                torch.randn(2, dtype=torch.bfloat16),
                torch.randn(8, dtype=torch.bfloat16),
                lower_bound=-5.0,
            )

        self.assertEqual(seen_dtypes, [(torch.float32, torch.float32)])

    @unittest.skipUnless(CUDA_AND_FLA, "FLA KDA parity requires CUDA and fla-core")
    def test_real_kernel_matches_reference_forward_and_backward(self):
        torch.manual_seed(13)
        device = torch.device("cuda")
        # Match the production KDA head width. FLA's optional TileLang backend
        # has narrower synthetic-width specializations that are not used by
        # the supported Qwen/Kimi configurations.
        # Three real sequences exercises the power-of-two FLA launch bucket
        # (padded to four) while the reference recurrence stays unpadded.
        shape = (3, 64, 2, 128)
        inputs = (
            torch.randn(shape, device=device, dtype=torch.bfloat16),
            torch.randn(shape, device=device, dtype=torch.bfloat16),
            torch.randn(shape, device=device, dtype=torch.bfloat16),
            torch.randn(shape, device=device, dtype=torch.bfloat16).mul_(0.1),
            torch.randn(shape[:-1], device=device, dtype=torch.bfloat16),
            torch.rand(shape[2], device=device, dtype=torch.float32).add_(0.5).log_(),
            torch.randn(shape[2] * shape[3], device=device, dtype=torch.float32).mul_(
                0.1
            ),
        )
        output_weight = torch.randn(shape, device=device, dtype=torch.float32)

        def run(kda_fn):
            differentiable = tuple(
                tensor.detach().clone().requires_grad_() for tensor in inputs
            )
            output = kda_fn(*differentiable, lower_bound=-5.0)
            gradients = torch.autograd.grad(
                (output.float() * output_weight).mean(), differentiable
            )
            return output.detach().float(), tuple(
                gradient.detach().float() for gradient in gradients
            )

        expected_output, expected_gradients = run(reference_kda)
        actual_output, actual_gradients = run(fla_kda)

        assert_close(actual_output, expected_output, rtol=4e-2, atol=4e-2)
        for index, (actual, expected) in enumerate(
            zip(actual_gradients, expected_gradients)
        ):
            with self.subTest(gradient=index):
                assert_close(actual, expected, rtol=1.5e-1, atol=2e-2)


class TestKDAInference(unittest.TestCase):
    def test_hf_save_and_reload_preserves_hybrid_architecture_and_outputs(self):
        torch.manual_seed(9)
        model = DFlashDraftModel(_kda_config()).eval()
        inputs = {
            "position_ids": torch.arange(7).unsqueeze(0),
            "noise_embedding": torch.randn(1, 4, 24),
            "target_hidden": torch.randn(1, 3, 72),
            "attention_mask": torch.ones(1, 1, 4, 7, dtype=torch.bool),
        }
        expected = model(**inputs)

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = DFlashDraftModel.from_pretrained(directory).eval()
            actual = loaded(**inputs)

        self.assertEqual(loaded.attention_modes, ("kda", "gqa", "kda"))
        assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_spec_generate_decode_smoke(self):
        torch.manual_seed(11)
        target_config = Qwen3Config(
            hidden_size=24,
            intermediate_size=48,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=8,
            head_dim=6,
            max_position_embeddings=128,
            vocab_size=64,
            tie_word_embeddings=False,
        )
        target_config._attn_implementation = "sdpa"
        target = Qwen3ForCausalLM(target_config).eval()
        draft = DFlashDraftModel(_kda_config()).eval()

        input_ids = torch.randint(1, 64, (1, 6))
        output_ids = draft.spec_generate(
            target,
            input_ids,
            max_new_tokens=6,
            stop_token_ids=None,
            temperature=0.0,
        )

        self.assertEqual(output_ids.shape[0], 1)
        self.assertLessEqual(output_ids.shape[1], input_ids.shape[1] + 6)
        self.assertTrue(torch.equal(output_ids[:, :6], input_ids))

    def test_spec_generate_can_return_acceptance_telemetry(self):
        torch.manual_seed(12)
        target_config = Qwen3Config(
            hidden_size=24,
            intermediate_size=48,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=8,
            head_dim=6,
            max_position_embeddings=128,
            vocab_size=64,
            tie_word_embeddings=False,
        )
        target_config._attn_implementation = "sdpa"
        target = Qwen3ForCausalLM(target_config).eval()
        draft = DFlashDraftModel(_kda_config()).eval()

        input_ids = torch.randint(1, 64, (1, 6))
        output = draft.spec_generate(
            target,
            input_ids,
            max_new_tokens=6,
            stop_token_ids=None,
            temperature=0.0,
            return_dict=True,
        )

        self.assertIsInstance(output, DFlashGenerationOutput)
        self.assertGreater(len(output.acceptance_lengths), 0)
        self.assertTrue(all(length >= 1 for length in output.acceptance_lengths))
        self.assertAlmostEqual(
            output.mean_acceptance_length,
            sum(output.acceptance_lengths) / len(output.acceptance_lengths),
        )
        self.assertTrue(torch.equal(output.sequences[:, :6], input_ids))

    def test_train_and_eval_forward_match_without_dropout(self):
        torch.manual_seed(13)
        model = DFlashDraftModel(_kda_config())
        inputs = {
            "position_ids": torch.arange(7).unsqueeze(0),
            "noise_embedding": torch.randn(1, 4, 24),
            "target_hidden": torch.randn(1, 3, 72),
            "attention_mask": torch.ones(1, 1, 4, 7, dtype=torch.bool),
        }

        model.train()
        train_output = model(**copy.deepcopy(inputs))
        model.eval()
        eval_output = model(**copy.deepcopy(inputs))

        assert_close(train_output, eval_output, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
