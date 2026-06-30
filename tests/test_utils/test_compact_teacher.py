"""CPU-friendly equivalence tests for the compact (draft-vocab) teacher path.

These compare ``specforge.core.compact_teacher`` against a plain reimplementation of
``specforge.core.eagle3._compute_target_p`` (the same math, without ``torch.compile``)
and, best-effort, against the real compiled function. They are CPU-friendly toy
tests; GPU-only checks (memory, tensor-parallel, export) live elsewhere.
"""

import unittest
from unittest import mock

import torch

from specforge.core.compact_teacher import (
    DEFAULT_VOCAB_CHUNK_SIZE,
    build_offline_teacher_inputs,
    compute_target_from_hidden,
    compute_target_p_padded_from_hidden,
    tiled_logsumexp_argmax,
    validate_compact_teacher_config,
    validate_compact_teacher_enabled,
    validate_vocab_mapping_consistency,
)
from specforge.core.lk_loss import compute_acceptance_rate, compute_lk_loss


def _reference_target_p(target, t2d, loss_mask):
    """Plain mirror of ``specforge.core.eagle3._compute_target_p`` (no torch.compile)."""
    target_head = target.float()
    target_token_ids = target_head.argmax(-1)
    target_mask = t2d[target_token_ids][..., None].int()
    position_mask = target_mask * loss_mask
    draft_target_head = target_head[..., t2d]
    target_p = torch.softmax(draft_target_head, dim=2)
    target_logsumexp = torch.logsumexp(target_head, dim=-1, keepdim=True)
    target_p_on_draft = torch.exp(draft_target_head - target_logsumexp)
    return target_p, target_p_on_draft, target_token_ids, position_mask


def _reference_target_p_padded(target, t2d, loss_mask, length):
    """Plain mirror of ``specforge.core.eagle3._compute_target_p_padded``."""
    target_p, target_p_on_draft, target_token_ids, position_mask = _reference_target_p(
        target, t2d, loss_mask
    )
    target_p_padded = torch.nn.functional.pad(
        target_p, (0, 0, 0, length), value=1 / target_p.shape[-1]
    )
    target_p_on_draft_padded = torch.nn.functional.pad(
        target_p_on_draft, (0, 0, 0, length), value=0.0
    )
    target_token_ids_padded = torch.nn.functional.pad(
        target_token_ids, (0, length), value=0
    )
    return (
        target_p_padded,
        target_p_on_draft_padded,
        target_token_ids_padded,
        position_mask,
    )


def _make_sorted_t2d(vocab_size, draft_vocab_size, generator):
    """Boolean draft-membership mask in ascending id order (as the real pipeline)."""
    perm = torch.randperm(vocab_size, generator=generator)
    selected = torch.sort(perm[:draft_vocab_size]).values
    t2d = torch.zeros(vocab_size, dtype=torch.bool)
    t2d[selected] = True
    return t2d


class _TargetHeadDouble(torch.nn.Module):
    """TargetHead stand-in with `.fc.weight` and a call-counting `forward`."""

    def __init__(self, weight):
        super().__init__()
        self.fc = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(weight)
        self.forward_calls = 0

    def forward(self, x):
        self.forward_calls += 1
        return self.fc(x)


class TestCompactTeacherEquivalence(unittest.TestCase):
    def setUp(self):
        self.gen = torch.Generator().manual_seed(0)
        self.batch, self.seq, self.hidden = 2, 3, 8
        self.vocab, self.draft_vocab = 64, 16

    def _make_inputs(self, dtype):
        hidden = torch.randn(self.batch, self.seq, self.hidden, generator=self.gen).to(
            dtype
        )
        weight = torch.randn(self.vocab, self.hidden, generator=self.gen).to(dtype)
        t2d = _make_sorted_t2d(self.vocab, self.draft_vocab, self.gen)
        loss_mask = torch.randint(
            0, 2, (self.batch, self.seq, 1), generator=self.gen
        ).int()
        return hidden, weight, t2d, loss_mask

    def test_matches_reference_bf16(self):
        # bf16 head logits then fp32 reduction, matching target.float() semantics.
        hidden, weight, t2d, loss_mask = self._make_inputs(torch.bfloat16)
        full_logits = torch.nn.functional.linear(hidden, weight)  # [B,S,V] bf16

        ref = _reference_target_p(full_logits, t2d, loss_mask)
        got = compute_target_from_hidden(
            hidden,
            weight,
            t2d,
            loss_mask,
            chunk_size=7,  # small chunk -> exercises tiling
        )

        torch.testing.assert_close(got[0], ref[0], rtol=1e-4, atol=1e-5)  # target_p
        torch.testing.assert_close(
            got[1], ref[1], rtol=1e-4, atol=1e-5
        )  # target_p_on_draft
        self.assertTrue(torch.equal(got[2], ref[2]))  # target_token_ids (exact)
        self.assertTrue(torch.equal(got[3], ref[3]))  # position_mask (exact)

    def test_chunk_size_independence(self):
        hidden, weight, t2d, loss_mask = self._make_inputs(torch.float32)
        full = compute_target_from_hidden(
            hidden, weight, t2d, loss_mask, chunk_size=self.vocab
        )
        tiled = compute_target_from_hidden(hidden, weight, t2d, loss_mask, chunk_size=5)
        for a, b in zip(full, tiled):
            torch.testing.assert_close(a.float(), b.float(), rtol=1e-5, atol=1e-6)

    def test_argmax_lowest_index_on_ties_across_chunks(self):
        # Two identical, maximal vocab rows in different chunks -> lowest index wins.
        hidden = torch.ones(1, 1, 4, dtype=torch.float32)
        weight = torch.zeros(8, 4, dtype=torch.float32)
        weight[1] = 10.0  # chunk 0 (chunk_size=3)
        weight[5] = 10.0  # chunk 1 -> identical maximal logit
        t2d = torch.zeros(8, dtype=torch.bool)
        t2d[[1, 5]] = True
        loss_mask = torch.ones(1, 1, 1, dtype=torch.int)

        logz, argmax_id = tiled_logsumexp_argmax(hidden, weight, chunk_size=3)
        full_argmax = torch.nn.functional.linear(hidden, weight).argmax(-1)
        self.assertEqual(int(argmax_id.item()), 1)
        self.assertTrue(torch.equal(argmax_id, full_argmax))
        # logZ finite and matches one-shot logsumexp
        ref_logz = torch.logsumexp(
            torch.nn.functional.linear(hidden, weight).float(), dim=-1, keepdim=True
        )
        torch.testing.assert_close(logz, ref_logz, rtol=1e-5, atol=1e-6)

    def test_outputs_are_detached(self):
        hidden, weight, t2d, loss_mask = self._make_inputs(torch.float32)
        hidden.requires_grad_(True)
        weight.requires_grad_(True)
        got = compute_target_from_hidden(hidden, weight, t2d, loss_mask, chunk_size=9)
        for tensor in got:
            self.assertFalse(tensor.requires_grad)

    def test_padded_path_compatible(self):
        # The compact outputs must pad exactly like _compute_target_p_padded expects.
        hidden, weight, t2d, loss_mask = self._make_inputs(torch.float32)
        target_p, target_p_on_draft, token_ids, _ = compute_target_from_hidden(
            hidden, weight, t2d, loss_mask, chunk_size=11
        )
        length = 4
        padded_p = torch.nn.functional.pad(
            target_p, (0, 0, 0, length), value=1 / target_p.shape[-1]
        )
        padded_on_draft = torch.nn.functional.pad(
            target_p_on_draft, (0, 0, 0, length), value=0.0
        )
        padded_ids = torch.nn.functional.pad(token_ids, (0, length), value=0)
        # Pad regions carry the documented constants; data region is unchanged.
        self.assertTrue(torch.all(padded_p[:, self.seq :, :] == 1 / self.draft_vocab))
        self.assertTrue(torch.all(padded_on_draft[:, self.seq :, :] == 0.0))
        self.assertTrue(torch.all(padded_ids[:, self.seq :] == 0))
        torch.testing.assert_close(padded_p[:, : self.seq, :], target_p)

    def test_padded_from_hidden_matches_reference(self):
        # The wired offline path uses compute_target_p_padded_from_hidden; it must
        # match padding the reference _compute_target_p outputs.
        hidden, weight, t2d, loss_mask = self._make_inputs(torch.bfloat16)
        full_logits = torch.nn.functional.linear(hidden, weight)
        length = 5
        ref = _reference_target_p_padded(full_logits, t2d, loss_mask, length)
        got = compute_target_p_padded_from_hidden(
            hidden, weight, t2d, loss_mask, length, chunk_size=7
        )
        torch.testing.assert_close(got[0], ref[0], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(got[1], ref[1], rtol=1e-4, atol=1e-5)
        self.assertTrue(torch.equal(got[2], ref[2]))
        self.assertTrue(torch.equal(got[3], ref[3]))
        self.assertTrue(torch.all(got[0][:, self.seq :, :] == 1 / self.draft_vocab))
        self.assertTrue(torch.all(got[1][:, self.seq :, :] == 0.0))
        self.assertTrue(torch.all(got[2][:, self.seq :] == 0))

    def test_compute_target_from_hidden_input_validation(self):
        hidden, weight, t2d, loss_mask = self._make_inputs(torch.float32)
        with self.assertRaises(ValueError):  # non-bool t2d
            compute_target_from_hidden(hidden, weight, t2d.int(), loss_mask)
        with self.assertRaises(ValueError):  # t2d length != vocab
            compute_target_from_hidden(hidden, weight, t2d[:-1], loss_mask)
        with self.assertRaises(ValueError):  # hidden size != lm_head hidden
            compute_target_from_hidden(hidden[..., :-1], weight, t2d, loss_mask)

    def test_multistep_ttt_slice_shift_equivalence(self):
        # Run both teacher sources through the real SdpaLikeAdapter.step_view + padding
        # shift for ttt_length>1 and assert per-step equality.
        try:
            from specforge.core.eagle3_adapters import SdpaLikeAdapter
            from specforge.utils import padding
        except Exception as exc:  # pragma: no cover - import/env dependent
            self.skipTest(f"adapter/padding import unavailable: {exc}")

        gen = torch.Generator().manual_seed(3)
        batch, seq, hidden_size, vocab, draft = 2, 6, 16, 96, 24
        hidden = torch.randn(batch, seq, hidden_size, generator=gen).bfloat16()
        weight = torch.randn(vocab, hidden_size, generator=gen).bfloat16()
        t2d = _make_sorted_t2d(vocab, draft, gen)
        loss_mask = torch.randint(0, 2, (batch, seq, 1), generator=gen).int()
        full_logits = torch.nn.functional.linear(hidden, weight)
        ttt_length = 3
        ref = _reference_target_p_padded(full_logits, t2d, loss_mask, ttt_length)
        got = compute_target_p_padded_from_hidden(
            hidden, weight, t2d, loss_mask, ttt_length, chunk_size=7
        )

        adapter = SdpaLikeAdapter(model=None)  # step_view does not use the model
        pm_ref, pm_got = ref[3], got[3]
        lm_ref, lm_got = loss_mask.clone(), loss_mask.clone()
        gi = torch.zeros(batch, seq, dtype=torch.long)
        for idx in range(ttt_length):
            common = dict(
                idx=idx,
                ttt_length=ttt_length,
                global_input_ids=gi,
                attention_mask=None,
                position_ids=None,
                hidden_states=hidden,
                seq_length=seq,
            )
            s_ref = adapter.step_view(
                loss_mask=lm_ref,
                target_p_padded=ref[0],
                target_p_on_draft_padded=ref[1],
                target_token_ids_padded=ref[2],
                position_mask=pm_ref,
                **common,
            )
            s_got = adapter.step_view(
                loss_mask=lm_got,
                target_p_padded=got[0],
                target_p_on_draft_padded=got[1],
                target_token_ids_padded=got[2],
                position_mask=pm_got,
                **common,
            )
            torch.testing.assert_close(
                s_got.target_p, s_ref.target_p, rtol=1e-4, atol=1e-5
            )
            torch.testing.assert_close(
                s_got.target_p_on_draft, s_ref.target_p_on_draft, rtol=1e-4, atol=1e-5
            )
            self.assertTrue(torch.equal(s_got.target_token_ids, s_ref.target_token_ids))
            self.assertTrue(torch.equal(s_got.position_mask, s_ref.position_mask))
            self.assertTrue(torch.equal(s_got.loss_mask, s_ref.loss_mask))
            if idx != ttt_length - 1:
                pm_ref = padding(pm_ref, left=False)
                pm_got = padding(pm_got, left=False)
                lm_ref = padding(lm_ref, left=False)
                lm_got = padding(lm_got, left=False)

    def test_wrong_tiling_is_detected(self):
        # A naive per-chunk logsumexp sum is wrong and must differ from the correct one.
        gen = torch.Generator().manual_seed(5)
        hidden = torch.randn(1, 1, 8, generator=gen)
        weight = torch.randn(40, 8, generator=gen)
        correct_logz, _ = tiled_logsumexp_argmax(hidden, weight, chunk_size=10)
        ref = torch.logsumexp(
            torch.nn.functional.linear(hidden, weight).float(), dim=-1, keepdim=True
        )
        torch.testing.assert_close(correct_logz, ref, rtol=1e-5, atol=1e-6)
        broken = torch.stack(
            [
                torch.logsumexp(
                    torch.nn.functional.linear(hidden, weight[s : s + 10]).float(),
                    dim=-1,
                    keepdim=True,
                )
                for s in range(0, 40, 10)
            ]
        ).sum(0)
        self.assertFalse(torch.allclose(broken, ref, rtol=1e-3, atol=1e-3))

    def test_matches_real_compute_target_p_best_effort(self):
        # Tie the equivalence to the real (compiled) function when importable.
        try:
            from specforge.core.eagle3 import _compute_target_p
        except Exception as exc:  # pragma: no cover - import/env dependent
            self.skipTest(f"could not import _compute_target_p: {exc}")

        hidden, weight, t2d, loss_mask = self._make_inputs(torch.float32)
        full_logits = torch.nn.functional.linear(hidden, weight)
        try:
            ref = _compute_target_p(target=full_logits, t2d=t2d, loss_mask=loss_mask)
        except Exception as exc:  # pragma: no cover - torch.compile env dependent
            self.skipTest(f"_compute_target_p (compiled) failed in this env: {exc}")

        got = compute_target_from_hidden(hidden, weight, t2d, loss_mask, chunk_size=7)
        torch.testing.assert_close(got[0], ref[0], rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(got[1], ref[1], rtol=1e-4, atol=1e-5)
        self.assertTrue(torch.equal(got[2], ref[2]))
        self.assertTrue(torch.equal(got[3], ref[3]))


class TestCompactTeacherValidation(unittest.TestCase):
    def test_rejects_equal_vocab_sizes(self):
        t2d = torch.ones(32, dtype=torch.bool)
        with self.assertRaises(ValueError):
            validate_compact_teacher_config(draft_vocab_size=32, vocab_size=32, t2d=t2d)

    def test_rejects_missing_mapping(self):
        with self.assertRaises(ValueError):
            validate_compact_teacher_config(
                draft_vocab_size=16, vocab_size=64, t2d=None
            )

    def test_rejects_wrong_dtype(self):
        t2d = torch.zeros(64, dtype=torch.int64)
        t2d[:16] = 1
        with self.assertRaises(ValueError):
            validate_compact_teacher_config(draft_vocab_size=16, vocab_size=64, t2d=t2d)

    def test_rejects_mismatched_selection_count(self):
        t2d = torch.zeros(64, dtype=torch.bool)
        t2d[:10] = True  # 10 selected != draft_vocab_size 16
        with self.assertRaises(ValueError):
            validate_compact_teacher_config(draft_vocab_size=16, vocab_size=64, t2d=t2d)

    def test_accepts_valid_config(self):
        t2d = torch.zeros(64, dtype=torch.bool)
        t2d[:16] = True
        validate_compact_teacher_config(draft_vocab_size=16, vocab_size=64, t2d=t2d)


class TestCompactTeacherEnabledValidation(unittest.TestCase):
    def _t2d(self):
        t = torch.zeros(64, dtype=torch.bool)
        t[:16] = True
        return t

    def _weight(self):
        return torch.zeros(64, 8)

    def test_rejects_online(self):
        with self.assertRaises(ValueError):
            validate_compact_teacher_enabled(
                is_online=True,
                is_vlm=False,
                draft_vocab_size=16,
                vocab_size=64,
                t2d=self._t2d(),
                target_head_weight=self._weight(),
            )

    def test_rejects_vlm(self):
        with self.assertRaises(ValueError):
            validate_compact_teacher_enabled(
                is_online=False,
                is_vlm=True,
                draft_vocab_size=16,
                vocab_size=64,
                t2d=self._t2d(),
                target_head_weight=self._weight(),
            )

    def test_rejects_missing_head_weight(self):
        with self.assertRaises(ValueError):
            validate_compact_teacher_enabled(
                is_online=False,
                is_vlm=False,
                draft_vocab_size=16,
                vocab_size=64,
                t2d=self._t2d(),
                target_head_weight=None,
            )

    def test_rejects_equal_vocab(self):
        t = torch.ones(32, dtype=torch.bool)
        with self.assertRaises(ValueError):
            validate_compact_teacher_enabled(
                is_online=False,
                is_vlm=False,
                draft_vocab_size=32,
                vocab_size=32,
                t2d=t,
                target_head_weight=torch.zeros(32, 8),
            )

    def test_rejects_bad_head_shape(self):
        with self.assertRaises(ValueError):  # wrong row count
            validate_compact_teacher_enabled(
                is_online=False,
                is_vlm=False,
                draft_vocab_size=16,
                vocab_size=64,
                t2d=self._t2d(),
                target_head_weight=torch.zeros(32, 8),
            )
        with self.assertRaises(ValueError):  # not 2-D
            validate_compact_teacher_enabled(
                is_online=False,
                is_vlm=False,
                draft_vocab_size=16,
                vocab_size=64,
                t2d=self._t2d(),
                target_head_weight=torch.zeros(64),
            )

    def test_rejects_nonpositive_chunk_size(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                validate_compact_teacher_enabled(
                    is_online=False,
                    is_vlm=False,
                    draft_vocab_size=16,
                    vocab_size=64,
                    t2d=self._t2d(),
                    target_head_weight=self._weight(),
                    chunk_size=bad,
                )

    def test_accepts_valid_offline(self):
        validate_compact_teacher_enabled(
            is_online=False,
            is_vlm=False,
            draft_vocab_size=16,
            vocab_size=64,
            t2d=self._t2d(),
            target_head_weight=self._weight(),
            chunk_size=128,
        )


class TestVocabMappingConsistency(unittest.TestCase):
    def _consistent(self, vocab=64, draft=16):
        gen = torch.Generator().manual_seed(0)
        t2d = _make_sorted_t2d(vocab, draft, gen)
        selected = torch.nonzero(t2d).flatten()
        d2t = selected - torch.arange(draft)
        return t2d, d2t

    def test_accepts_consistent(self):
        t2d, d2t = self._consistent()
        validate_vocab_mapping_consistency(t2d, d2t)

    def test_rejects_wrong_order(self):
        t2d, d2t = self._consistent()
        d2t_bad = d2t.clone()
        d2t_bad[0] = d2t_bad[0] + 1  # breaks selected == d2t + arange deterministically
        with self.assertRaises(ValueError):
            validate_vocab_mapping_consistency(t2d, d2t_bad)

    def test_rejects_count_mismatch(self):
        t2d = torch.zeros(64, dtype=torch.bool)
        t2d[:16] = True
        with self.assertRaises(ValueError):
            validate_vocab_mapping_consistency(t2d, torch.zeros(10, dtype=torch.long))


class TestCompactTeacherMemory(unittest.TestCase):
    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA required for memory regression"
    )
    def test_peak_memory_drop_vs_full_path(self):
        # Route both branches through build_offline_teacher_inputs (the run_forward seam).
        dev = "cuda"
        batch, seq, hidden_size, vocab, draft = 1, 2048, 4096, 128256, 32000
        length = 7
        hidden = torch.randn(batch, seq, hidden_size, device=dev, dtype=torch.bfloat16)
        weight = torch.randn(vocab, hidden_size, device=dev, dtype=torch.bfloat16)
        double = _TargetHeadDouble(weight).to(device=dev, dtype=torch.bfloat16)
        sel = torch.sort(torch.randperm(vocab, device=dev)[:draft]).values
        t2d = torch.zeros(vocab, dtype=torch.bool, device=dev)
        t2d[sel] = True
        loss_mask = torch.randint(0, 2, (batch, seq, 1), device=dev).int()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        full_logits, kw_full = build_offline_teacher_inputs(
            compact=False,
            target_model=double,
            target_hidden=hidden,
            chunk_size_arg=None,
        )
        full_out = _reference_target_p_padded(full_logits, t2d, loss_mask, length)
        torch.cuda.synchronize()
        full_peak = torch.cuda.max_memory_allocated() - base
        self.assertEqual(kw_full, {})
        self.assertEqual(double.forward_calls, 1)
        del full_logits, full_out
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        double.forward_calls = 0
        linear_out_dims = []
        orig_linear = torch.nn.functional.linear

        def _record_linear(inp, weight, *a, **k):
            out = orig_linear(inp, weight, *a, **k)
            linear_out_dims.append(out.shape[-1])
            return out

        torch.cuda.reset_peak_memory_stats()
        base2 = torch.cuda.memory_allocated()
        target_c, kw = build_offline_teacher_inputs(
            compact=True,
            target_model=double,
            target_hidden=hidden,
            chunk_size_arg=32768,
        )
        self.assertIsNone(target_c)
        self.assertEqual(double.forward_calls, 0)  # no full-vocab logits produced
        with mock.patch("torch.nn.functional.linear", _record_linear):
            compact_out = compute_target_p_padded_from_hidden(
                hidden,
                kw["target_head_weight"],
                t2d,
                loss_mask,
                length,
                chunk_size=kw["compact_teacher_chunk_size"],
            )
        torch.cuda.synchronize()
        compact_peak = torch.cuda.max_memory_allocated() - base2
        del compact_out
        self.assertTrue(linear_out_dims)
        self.assertNotIn(vocab, linear_out_dims)  # no [.., vocab] projection in compact

        fp32_full_cast = batch * seq * vocab * 4
        drop = full_peak - compact_peak
        print(
            f"\n[memory] full_peak={full_peak/1e9:.3f}GB compact_peak={compact_peak/1e9:.3f}GB "
            f"drop={drop/1e9:.3f}GB fp32_cast={fp32_full_cast/1e9:.3f}GB"
        )
        self.assertLess(compact_peak, full_peak)
        self.assertGreaterEqual(drop, fp32_full_cast - 64 * 1024 * 1024)


class TestBuildOfflineTeacherInputs(unittest.TestCase):
    def setUp(self):
        gen = torch.Generator().manual_seed(0)
        self.hidden = torch.randn(2, 3, 8, generator=gen)
        self.weight = torch.randn(64, 8, generator=gen)

    def test_legacy_calls_forward_returns_full_logits(self):
        double = _TargetHeadDouble(self.weight)
        target, kw = build_offline_teacher_inputs(
            compact=False,
            target_model=double,
            target_hidden=self.hidden,
            chunk_size_arg=None,
        )
        self.assertEqual(double.forward_calls, 1)
        self.assertEqual(kw, {})
        self.assertEqual(tuple(target.shape), (2, 3, 64))

    def test_compact_skips_forward_returns_kwargs(self):
        double = _TargetHeadDouble(self.weight)
        target, kw = build_offline_teacher_inputs(
            compact=True,
            target_model=double,
            target_hidden=self.hidden,
            chunk_size_arg=None,
        )
        self.assertEqual(double.forward_calls, 0)
        self.assertIsNone(target)
        self.assertIs(kw["target_head_weight"], double.fc.weight)
        self.assertTrue(torch.equal(kw["target_hidden_for_compact"], self.hidden))
        self.assertEqual(kw["compact_teacher_chunk_size"], DEFAULT_VOCAB_CHUNK_SIZE)

    def test_compact_resolves_explicit_chunk(self):
        double = _TargetHeadDouble(self.weight)
        _, kw = build_offline_teacher_inputs(
            compact=True,
            target_model=double,
            target_hidden=self.hidden,
            chunk_size_arg=64,
        )
        self.assertEqual(kw["compact_teacher_chunk_size"], 64)


def _kl_loss(logits, target_p, position_mask):
    logp = torch.log_softmax(logits.float(), dim=-1)
    return -(position_mask * (target_p * logp)).sum(-1).mean()


def _acc_and_loss(logits, target_p, target_p_on_draft, position_mask, lk_loss_type):
    kl = _kl_loss(logits, target_p, position_mask)
    acc, log_acc = compute_acceptance_rate(
        logits=logits, target_probs=target_p_on_draft, position_mask=position_mask
    )
    if lk_loss_type is None:
        return acc, kl
    loss = compute_lk_loss(
        kl_loss=kl,
        acceptance_rate=acc,
        log_acceptance_rate=log_acc,
        lk_loss_type=lk_loss_type,
        kl_scale=1.0,
        kl_decay=1.0,
    )
    return acc, loss


class TestAcceptanceEquivalence(unittest.TestCase):
    def test_acceptance_and_loss_match_full(self):
        gen = torch.Generator().manual_seed(2)
        batch, seq, hidden_size, vocab, draft = 2, 4, 16, 80, 20
        hidden = torch.randn(batch, seq, hidden_size, generator=gen).bfloat16()
        weight = torch.randn(vocab, hidden_size, generator=gen).bfloat16()
        t2d = _make_sorted_t2d(vocab, draft, gen)
        loss_mask = torch.randint(0, 2, (batch, seq, 1), generator=gen).int()
        full = _reference_target_p(
            torch.nn.functional.linear(hidden, weight), t2d, loss_mask
        )
        comp = compute_target_from_hidden(hidden, weight, t2d, loss_mask, chunk_size=7)
        logits = torch.randn(batch, seq, draft, generator=gen)
        for lk in (None, "alpha", "lambda"):
            acc_full, loss_full = _acc_and_loss(logits, full[0], full[1], full[3], lk)
            acc_comp, loss_comp = _acc_and_loss(logits, comp[0], comp[1], comp[3], lk)
            torch.testing.assert_close(acc_comp, acc_full, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(loss_comp, loss_full, rtol=1e-4, atol=1e-5)

    def test_approximate_diverges_with_out_of_draft_mass(self):
        vocab, draft = 8, 3
        t2d = torch.zeros(vocab, dtype=torch.bool)
        t2d[[0, 1, 2]] = True
        full_logits = torch.zeros(1, 1, vocab)
        full_logits[0, 0, 7] = 10.0  # dominant token is outside the draft vocab
        loss_mask = torch.ones(1, 1, 1, dtype=torch.int)
        _, exact_on_draft, _, _ = _reference_target_p(full_logits, t2d, loss_mask)
        approx_on_draft = torch.softmax(full_logits[..., t2d], dim=-1)
        self.assertLess(float(exact_on_draft.sum()), 0.5)
        self.assertAlmostEqual(float(approx_on_draft.sum()), 1.0, places=4)
        logits = torch.zeros(1, 1, draft)
        acc_exact, _ = compute_acceptance_rate(
            logits=logits, target_probs=exact_on_draft, position_mask=loss_mask
        )
        acc_approx, _ = compute_acceptance_rate(
            logits=logits, target_probs=approx_on_draft, position_mask=loss_mask
        )
        self.assertGreater(float(acc_approx), float(acc_exact) + 1e-3)


class TestCompactTeacherRankReplication(unittest.TestCase):
    def test_deterministic_pure_function(self):
        # The offline teacher uses the full, rank-replicated TargetHead.fc, so it is a
        # pure function of (hidden, weight, t2d): every TP rank gets identical outputs.
        gen = torch.Generator().manual_seed(11)
        hidden = torch.randn(2, 4, 16, generator=gen)
        weight = torch.randn(70, 16, generator=gen)
        t2d = _make_sorted_t2d(70, 18, gen)
        loss_mask = torch.randint(0, 2, (2, 4, 1), generator=gen).int()
        a = compute_target_p_padded_from_hidden(
            hidden, weight, t2d, loss_mask, 3, chunk_size=7
        )
        b = compute_target_p_padded_from_hidden(
            hidden, weight, t2d, loss_mask, 3, chunk_size=7
        )
        for x, y in zip(a, b):
            self.assertTrue(torch.equal(x, y))

    def test_chunk_size_invariant(self):
        gen = torch.Generator().manual_seed(12)
        hidden = torch.randn(1, 3, 16, generator=gen)
        weight = torch.randn(70, 16, generator=gen)
        t2d = _make_sorted_t2d(70, 18, gen)
        loss_mask = torch.ones(1, 3, 1, dtype=torch.int)
        full = compute_target_p_padded_from_hidden(
            hidden, weight, t2d, loss_mask, 2, chunk_size=70
        )
        tiled = compute_target_p_padded_from_hidden(
            hidden, weight, t2d, loss_mask, 2, chunk_size=9
        )
        for x, y in zip(full, tiled):
            torch.testing.assert_close(x.float(), y.float(), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
