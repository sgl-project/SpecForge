"""Tests for the context-scanning KDA policy (``linear_attn_config.context_state``)."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from unittest import mock

import torch
from torch.testing import assert_close
from transformers import Qwen3Config, Qwen3ForCausalLM

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.kda import (
    Qwen3DFlashKDAAttention,
    fla_kda,
    reference_kda,
    scan_kda_context_states,
)

CUDA_AND_FLA = torch.cuda.is_available() and importlib.util.find_spec("fla") is not None

HIDDEN, HEADS, HEAD_DIM, BLOCK, CONV = 24, 4, 6, 2, 3


def _config(
    *,
    context_state: str = "scan",
    attention_modes: list[str] | None = None,
    backend: str = "reference",
) -> Qwen3Config:
    config = Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=48,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=3,
        head_dim=HEAD_DIM,
        max_position_embeddings=128,
        vocab_size=64,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    config.architectures = ["DFlashDraftModel"]
    config.num_target_layers = 8
    config.block_size = BLOCK
    config.draft_vocab_size = 64
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    config.use_sliding_window = False
    config.dflash_config = {
        "attention_modes": attention_modes or ["kda", "gqa", "kda"],
        "mask_token_id": 0,
    }
    config.linear_attn_config = {
        "head_dim": HEAD_DIM,
        "num_heads": HEADS,
        "short_conv_kernel_size": CONV,
        "use_full_rank_gate": False,
        "gate_lower_bound": -5.0,
        "backend": backend,
        "context_state": context_state,
    }
    return config


def _attention(context_state: str = "scan") -> Qwen3DFlashKDAAttention:
    return Qwen3DFlashKDAAttention(
        _config(context_state=context_state), layer_idx=0, kernels=mock.Mock()
    ).eval()


def _run(
    attention: Qwen3DFlashKDAAttention,
    blocks: torch.Tensor,
    context: torch.Tensor,
    *,
    anchors: torch.Tensor | None = None,
    cache=None,
) -> torch.Tensor:
    return attention(
        hidden_states=blocks,
        target_hidden=context,
        position_embeddings=(None, None),
        attention_mask=None,
        past_key_values=cache,
        anchor_positions=anchors,
    )[0]


def _oracle(
    attention: Qwen3DFlashKDAAttention,
    context_prefix: torch.Tensor,
    block: torch.Tensor,
) -> torch.Tensor:
    """KDA over the contiguous sequence ``[context_prefix ; block]``."""

    sequence = torch.cat((context_prefix, block), dim=1)
    length = sequence.shape[1]
    shape = (1, length, HEADS, HEAD_DIM)
    q = attention.q_conv1d(attention.q_proj(sequence)).view(shape)
    k = attention.k_conv1d(attention.k_proj(sequence)).view(shape)
    v = attention.v_conv1d(attention.v_proj(sequence)).view(shape)
    raw_gate = attention.f_b_proj(attention.f_a_proj(sequence)).view(shape)
    beta = attention.b_proj(sequence).view(1, length, HEADS)
    output = reference_kda(
        q, k, v, raw_gate, beta, attention.A_log, attention.dt_bias, -5.0
    )[:, context_prefix.shape[1] :]
    gate = attention._output_gate(block).view(1, BLOCK, HEADS, HEAD_DIM)
    return attention.o_proj(attention.o_norm(output, gate).flatten(-2))


def _tiny_target() -> Qwen3ForCausalLM:
    target_config = Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=48,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=8,
        head_dim=HEAD_DIM,
        max_position_embeddings=128,
        vocab_size=64,
        tie_word_embeddings=False,
    )
    target_config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(target_config).eval()


class TestContextScanSemantics(unittest.TestCase):
    def test_every_anchor_matches_the_contiguous_sequence_oracle(self):
        torch.manual_seed(1)
        attention = _attention()
        context = torch.randn(1, 9, HIDDEN)
        # Unsorted, duplicated, empty-context (0), partial-conv-window (1) and
        # full-context (9) anchors in one block-parallel call.
        anchors = torch.tensor([[7, 0, 3, 9, 3, 1]])
        blocks = torch.randn(1, anchors.shape[1] * BLOCK, HIDDEN)

        output = _run(attention, blocks, context, anchors=anchors)

        for index, anchor in enumerate(anchors[0].tolist()):
            with self.subTest(anchor=anchor, block=index):
                block = blocks[:, index * BLOCK : (index + 1) * BLOCK]
                expected = _oracle(attention, context[:, :anchor], block)
                assert_close(
                    output[:, index * BLOCK : (index + 1) * BLOCK],
                    expected,
                    rtol=1e-5,
                    atol=1e-6,
                )

    def test_default_anchor_is_the_full_context(self):
        torch.manual_seed(2)
        attention = _attention()
        context = torch.randn(1, 5, HIDDEN)
        blocks = torch.randn(1, 2 * BLOCK, HIDDEN)

        implicit = _run(attention, blocks, context)
        explicit = _run(attention, blocks, context, anchors=torch.tensor([[5, 5]]))

        assert_close(implicit, explicit, rtol=0, atol=0)

    def test_rows_with_different_anchors_match_per_row_results(self):
        torch.manual_seed(3)
        attention = _attention()
        context = torch.randn(2, 6, HIDDEN)
        anchors = torch.tensor([[2, 6, 0], [5, 5, 3]])
        blocks = torch.randn(2, 3 * BLOCK, HIDDEN)

        batched = _run(attention, blocks, context, anchors=anchors)

        for row in range(2):
            single = _run(
                attention,
                blocks[row : row + 1],
                context[row : row + 1],
                anchors=anchors[row : row + 1],
            )
            assert_close(batched[row : row + 1], single, rtol=1e-5, atol=1e-6)

    def test_context_changes_reach_the_block_output(self):
        torch.manual_seed(4)
        attention = _attention()
        blocks = torch.randn(1, BLOCK, HIDDEN)
        context = torch.randn(1, 4, HIDDEN)

        output = _run(attention, blocks, context)
        perturbed = _run(attention, blocks, context + 1.0)

        self.assertFalse(torch.allclose(output, perturbed))

    def test_reset_policy_keeps_block_local_semantics(self):
        torch.manual_seed(5)
        attention = _attention(context_state="reset")
        blocks = torch.randn(1, 2 * BLOCK, HIDDEN)

        first = _run(
            attention, blocks, torch.randn(1, 5, HIDDEN), anchors=torch.tensor([[1, 4]])
        )
        second = _run(attention, blocks, torch.randn(1, 3, HIDDEN))

        assert_close(first, second, rtol=0, atol=0)
        self.assertFalse(attention.scans_context)

    def test_rejects_anchor_shape_mismatch(self):
        attention = _attention()
        with self.assertRaisesRegex(ValueError, "anchor_positions"):
            _run(
                attention,
                torch.randn(1, 2 * BLOCK, HIDDEN),
                torch.randn(1, 4, HIDDEN),
                anchors=torch.tensor([[1, 2, 3]]),
            )


class TestTwoLevelScan(unittest.TestCase):
    def test_grouped_scan_matches_direct_prefix_states(self):
        torch.manual_seed(6)
        length, heads, dim = 13, 2, 3
        k = torch.randn(1, length, heads, dim)
        v = torch.randn(1, length, heads, dim)
        raw_gate = torch.randn(1, length, heads, dim) * 0.1
        beta = torch.randn(1, length, heads)
        A_log = torch.rand(heads).add(0.5).log()
        dt_bias = torch.randn(heads * dim) * 0.1
        anchors = torch.tensor([[0, 5, 5, 13, 2, 8, 12, 1]])

        for group_size in (None, 1, 2, 5, 8):
            with self.subTest(group_size=group_size):
                states = scan_kda_context_states(
                    reference_kda,
                    k,
                    v,
                    raw_gate,
                    beta,
                    A_log,
                    dt_bias,
                    -5.0,
                    anchors,
                    group_size=group_size,
                )
                self.assertEqual(states.shape, (1, 8, heads, dim, dim))
                for index, anchor in enumerate(anchors[0].tolist()):
                    _, expected = reference_kda(
                        torch.zeros_like(k[:, :anchor]),
                        k[:, :anchor],
                        v[:, :anchor],
                        raw_gate[:, :anchor],
                        beta[:, :anchor],
                        A_log,
                        dt_bias,
                        -5.0,
                        output_final_state=True,
                    )
                    assert_close(states[0, index], expected[0], rtol=1e-5, atol=1e-6)

    def test_anchors_beyond_the_context_clamp_to_its_end(self):
        torch.manual_seed(7)
        k = torch.randn(1, 4, 2, 3)
        v = torch.randn(1, 4, 2, 3)
        raw_gate = torch.zeros(1, 4, 2, 3)
        beta = torch.zeros(1, 4, 2)
        A_log = torch.zeros(2)
        dt_bias = torch.zeros(6)

        states = scan_kda_context_states(
            reference_kda,
            k,
            v,
            raw_gate,
            beta,
            A_log,
            dt_bias,
            None,
            torch.tensor([[4, 9]]),
        )

        assert_close(states[0, 0], states[0, 1], rtol=0, atol=0)

    def test_reference_varlen_layout_matches_per_sequence_runs(self):
        torch.manual_seed(8)
        heads, dim = 2, 3
        q = torch.randn(1, 7, heads, dim)
        k = torch.randn(1, 7, heads, dim)
        v = torch.randn(1, 7, heads, dim)
        raw_gate = torch.randn(1, 7, heads, dim) * 0.1
        beta = torch.randn(1, 7, heads)
        A_log = torch.zeros(heads)
        dt_bias = torch.zeros(heads * dim)
        initial_state = torch.randn(2, heads, dim, dim)
        cu_seqlens = torch.tensor([0, 3, 7])

        output, final = reference_kda(
            q,
            k,
            v,
            raw_gate,
            beta,
            A_log,
            dt_bias,
            -5.0,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )

        for index, (start, end) in enumerate(((0, 3), (3, 7))):
            expected_output, expected_final = reference_kda(
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                raw_gate[:, start:end],
                beta[:, start:end],
                A_log,
                dt_bias,
                -5.0,
                initial_state=initial_state[index : index + 1],
                output_final_state=True,
            )
            assert_close(output[:, start:end], expected_output, rtol=0, atol=0)
            assert_close(final[index : index + 1], expected_final, rtol=0, atol=0)


class TestRunningState(unittest.TestCase):
    def test_incremental_slices_match_the_single_shot_result(self):
        torch.manual_seed(9)
        attention = _attention()
        context = torch.randn(1, 7, HIDDEN)
        block = torch.randn(1, BLOCK, HIDDEN)
        cache = mock.Mock()

        single_prefix = _run(attention, block, context[:, :3])
        single_full = _run(attention, block, context)

        attention.reset_state()
        first = _run(attention, block, context[:, :3], cache=cache)
        second = _run(attention, block, context[:, 3:], cache=cache)

        assert_close(first, single_prefix, rtol=1e-5, atol=1e-6)
        assert_close(second, single_full, rtol=1e-5, atol=1e-6)
        cache.update.assert_not_called()

    def test_reset_state_starts_a_fresh_request(self):
        torch.manual_seed(10)
        attention = _attention()
        context = torch.randn(1, 4, HIDDEN)
        block = torch.randn(1, BLOCK, HIDDEN)
        cache = mock.Mock()

        expected = _run(attention, block, context)
        _run(attention, block, torch.randn(1, 6, HIDDEN), cache=cache)
        stale = _run(attention, block, context, cache=cache)
        attention.reset_state()
        fresh = _run(attention, block, context, cache=cache)

        self.assertFalse(torch.allclose(stale, expected))
        assert_close(fresh, expected, rtol=1e-5, atol=1e-6)

    def test_empty_slice_leaves_the_running_state_alone(self):
        torch.manual_seed(11)
        attention = _attention()
        context = torch.randn(1, 5, HIDDEN)
        block = torch.randn(1, BLOCK, HIDDEN)
        cache = mock.Mock()

        expected = _run(attention, block, context, cache=cache)
        again = _run(attention, block, context[:, :0], cache=cache)

        assert_close(again, expected, rtol=0, atol=0)


class TestModelIntegration(unittest.TestCase):
    def _inputs(self, *, batch_size: int, num_blocks: int, context_len: int):
        draft_len = num_blocks * BLOCK
        return {
            "position_ids": torch.arange(context_len + draft_len).expand(
                batch_size, -1
            ),
            "noise_embedding": torch.randn(batch_size, draft_len, HIDDEN),
            "target_hidden": torch.randn(batch_size, context_len, 3 * HIDDEN),
            "attention_mask": torch.ones(
                batch_size, 1, draft_len, context_len + draft_len, dtype=torch.bool
            ),
        }

    def test_scan_gradients_reach_context_and_all_parameter_groups(self):
        torch.manual_seed(12)
        model = DFlashDraftModel(_config()).train()
        inputs = self._inputs(batch_size=2, num_blocks=2, context_len=5)
        inputs["noise_embedding"].requires_grad_()
        inputs["target_hidden"].requires_grad_()

        output = model(**inputs, anchor_positions=torch.tensor([[1, 5], [0, 3]]))
        self.assertEqual(output.shape, (2, 2 * BLOCK, HIDDEN))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()

        for name in ("noise_embedding", "target_hidden"):
            self.assertIsNotNone(inputs[name].grad, name)
            self.assertTrue(torch.isfinite(inputs[name].grad).all(), name)
        attention = model.layers[0].self_attn
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "q_conv1d",
            "k_conv1d",
            "v_conv1d",
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

    def test_dense_only_stacks_ignore_anchor_positions(self):
        torch.manual_seed(13)
        model = DFlashDraftModel(_config(attention_modes=["gqa"] * 3)).eval()
        inputs = self._inputs(batch_size=1, num_blocks=2, context_len=4)

        plain = model(**inputs)
        with_anchors = model(**inputs, anchor_positions=torch.tensor([[1, 4]]))

        assert_close(plain, with_anchors, rtol=0, atol=0)

    def test_rejects_unknown_context_state(self):
        config = _config()
        config.linear_attn_config["context_state"] = "mystery"

        with self.assertRaisesRegex(ValueError, "context_state"):
            DFlashDraftModel(config)

    def test_save_and_reload_preserves_policy_and_outputs(self):
        torch.manual_seed(14)
        model = DFlashDraftModel(_config()).eval()
        inputs = self._inputs(batch_size=1, num_blocks=2, context_len=3)
        anchors = torch.tensor([[0, 3]])
        expected = model(**inputs, anchor_positions=anchors)

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            loaded = DFlashDraftModel.from_pretrained(directory).eval()
            actual = loaded(**inputs, anchor_positions=anchors)

        self.assertTrue(loaded.layers[0].self_attn.scans_context)
        assert_close(actual, expected, rtol=1e-6, atol=1e-7)

    def test_spec_generate_runs_and_resets_between_requests(self):
        torch.manual_seed(15)
        target = _tiny_target()
        draft = DFlashDraftModel(_config()).eval()
        input_ids = torch.randint(1, 64, (1, 6))

        first = draft.spec_generate(
            target, input_ids, max_new_tokens=6, stop_token_ids=None, temperature=0.0
        )
        second = draft.spec_generate(
            target, input_ids, max_new_tokens=6, stop_token_ids=None, temperature=0.0
        )

        self.assertTrue(torch.equal(first[:, :6], input_ids))
        self.assertLessEqual(first.shape[1], input_ids.shape[1] + 6)
        self.assertTrue(torch.equal(first, second))
        for layer in draft.layers:
            attention = layer.self_attn
            if isinstance(attention, Qwen3DFlashKDAAttention):
                self.assertIsNotNone(attention._running_state)


class TestFLAContextDispatch(unittest.TestCase):
    def test_varlen_pads_sequence_count_and_length_to_buckets(self):
        calls = []

        def fake_chunk_kda(**kwargs):
            bounds = kwargs["cu_seqlens"].tolist()
            calls.append((tuple(kwargs["q"].shape), bounds, kwargs["initial_state"]))
            heads, dim = kwargs["k"].shape[2], kwargs["k"].shape[3]
            final = torch.zeros(len(bounds) - 1, heads, dim, kwargs["v"].shape[3])
            return kwargs["q"], final

        q = torch.randn(1, 10, 2, 3)
        initial_state = torch.randn(2, 2, 3, 3)
        with mock.patch(
            "specforge.modeling.draft.kda._load_fla_chunk_kda",
            return_value=fake_chunk_kda,
        ):
            output, final = fla_kda(
                q,
                torch.randn_like(q),
                torch.randn_like(q),
                torch.randn_like(q),
                torch.randn(1, 10, 2),
                torch.randn(2),
                torch.randn(6),
                -5.0,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=torch.tensor([0, 4, 10]),
            )

        ((shape, bounds, padded_state),) = calls
        self.assertEqual(shape, (1, 64, 2, 3))
        self.assertEqual(bounds[:3], [0, 4, 10])
        self.assertEqual(len(bounds) - 1, 4)
        self.assertEqual(bounds[-1], 64)
        self.assertEqual(tuple(padded_state.shape), (4, 2, 3, 3))
        assert_close(padded_state[:2], initial_state)
        assert_close(output, q)
        self.assertEqual(tuple(final.shape), (2, 2, 3, 3))

    def test_varlen_aligned_input_needs_no_padding(self):
        calls = []

        def fake_chunk_kda(**kwargs):
            calls.append((tuple(kwargs["q"].shape), kwargs["cu_seqlens"].tolist()))
            return kwargs["q"], None

        q = torch.randn(1, 128, 2, 3)
        with mock.patch(
            "specforge.modeling.draft.kda._load_fla_chunk_kda",
            return_value=fake_chunk_kda,
        ):
            fla_kda(
                q,
                torch.randn_like(q),
                torch.randn_like(q),
                torch.randn_like(q),
                torch.randn(1, 128, 2),
                torch.randn(2),
                torch.randn(6),
                -5.0,
                cu_seqlens=torch.tensor([0, 64, 128]),
            )

        self.assertEqual(calls, [((1, 128, 2, 3), [0, 64, 128])])

    def test_bucketed_blocks_pad_and_slice_initial_states(self):
        calls = []

        def fake_chunk_kda(**kwargs):
            calls.append(tuple(kwargs["initial_state"].shape))
            return kwargs["q"], kwargs["initial_state"]

        q = torch.randn(3, 2, 2, 3)
        initial_state = torch.randn(3, 2, 3, 3)
        with mock.patch(
            "specforge.modeling.draft.kda._load_fla_chunk_kda",
            return_value=fake_chunk_kda,
        ):
            output, final = fla_kda(
                q,
                torch.randn_like(q),
                torch.randn_like(q),
                torch.randn_like(q),
                torch.randn(3, 2, 2),
                torch.randn(2),
                torch.randn(6),
                -5.0,
                initial_state=initial_state,
                output_final_state=True,
            )

        self.assertEqual(calls, [(4, 2, 3, 3)])
        assert_close(output, q)
        assert_close(final, initial_state)

    @unittest.skipUnless(CUDA_AND_FLA, "FLA KDA parity requires CUDA and fla-core")
    def test_real_kernel_matches_reference_with_states_and_varlen(self):
        torch.manual_seed(16)
        device = torch.device("cuda")
        heads, dim = 2, 128
        context_len, num_blocks = 200, 5
        k = torch.randn(1, context_len, heads, dim, device=device, dtype=torch.bfloat16)
        v = torch.randn_like(k)
        raw_gate = torch.randn_like(k).mul_(0.1)
        beta = torch.randn(1, context_len, heads, device=device, dtype=torch.bfloat16)
        A_log = torch.rand(heads, device=device).add_(0.5).log_()
        dt_bias = torch.randn(heads * dim, device=device).mul_(0.1)
        anchors = torch.tensor([[0, 37, 37, 130, 200]], device=device)

        expected = scan_kda_context_states(
            reference_kda, k, v, raw_gate, beta, A_log, dt_bias, -5.0, anchors
        )
        actual = scan_kda_context_states(
            fla_kda, k, v, raw_gate, beta, A_log, dt_bias, -5.0, anchors
        )
        assert_close(actual, expected, rtol=4e-2, atol=4e-2)

        block_shape = (num_blocks, 8, heads, dim)
        block_inputs = (
            torch.randn(block_shape, device=device, dtype=torch.bfloat16),
            torch.randn(block_shape, device=device, dtype=torch.bfloat16),
            torch.randn(block_shape, device=device, dtype=torch.bfloat16),
            torch.randn(block_shape, device=device, dtype=torch.bfloat16).mul_(0.1),
            torch.randn(block_shape[:-1], device=device, dtype=torch.bfloat16),
        )
        expected_output = reference_kda(
            *block_inputs, A_log, dt_bias, -5.0, initial_state=expected[0]
        )
        actual_output = fla_kda(
            *block_inputs, A_log, dt_bias, -5.0, initial_state=expected[0]
        )
        assert_close(
            actual_output.float(), expected_output.float(), rtol=4e-2, atol=4e-2
        )


class _RecordingDraft(torch.nn.Module):
    accepts_anchor_positions = True

    def __init__(self) -> None:
        super().__init__()
        self.sliding_window = None
        self.received = None

    def forward(
        self,
        position_ids,
        noise_embedding,
        target_hidden,
        attention_mask,
        anchor_positions=None,
        **kwargs,
    ):
        self.received = anchor_positions
        return torch.zeros(*noise_embedding.shape[:2], HIDDEN)


class _LegacyDraft(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sliding_window = None

    def forward(self, position_ids, noise_embedding, target_hidden, attention_mask):
        return torch.zeros(*noise_embedding.shape[:2], HIDDEN)


class TestTrainerAnchorPlumbing(unittest.TestCase):
    def _trainer(self, draft):
        from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel

        return OnlineDFlashModel(
            draft_model=draft,
            target_lm_head=torch.nn.Linear(HIDDEN, 64),
            target_embed_tokens=torch.nn.Embedding(64, HIDDEN),
            mask_token_id=0,
            block_size=BLOCK,
            attention_backend="sdpa",
            num_anchors=3,
        )

    def test_declaring_drafts_receive_the_sampled_anchors(self):
        torch.manual_seed(17)
        draft = _RecordingDraft()
        trainer = self._trainer(draft)
        input_ids = torch.randint(1, 64, (1, 8))

        anchors, _, output = trainer._forward_draft_blocks(
            input_ids, torch.randn(1, 8, 3 * HIDDEN), torch.ones(1, 8)
        )

        self.assertEqual(output.shape, (1, 3 * BLOCK, HIDDEN))
        self.assertIsNotNone(draft.received)
        self.assertTrue(torch.equal(draft.received, anchors))

    def test_legacy_drafts_are_called_without_the_keyword(self):
        torch.manual_seed(18)
        trainer = self._trainer(_LegacyDraft())
        input_ids = torch.randint(1, 64, (1, 8))

        _, _, output = trainer._forward_draft_blocks(
            input_ids, torch.randn(1, 8, 3 * HIDDEN), torch.ones(1, 8)
        )

        self.assertEqual(output.shape, (1, 3 * BLOCK, HIDDEN))

    def test_dflash_draft_models_declare_anchor_support(self):
        self.assertTrue(DFlashDraftModel.accepts_anchor_positions)


if __name__ == "__main__":
    unittest.main(verbosity=2)
