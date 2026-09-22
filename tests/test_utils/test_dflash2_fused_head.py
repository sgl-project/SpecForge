# coding=utf-8
"""Tests for the fused Triton DFlash2 unary head."""

import os
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.modeling.draft.dflash2 import DFlash2DraftModel

try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

CUDA_TRITON = torch.cuda.is_available() and TRITON_AVAILABLE


def _draft_config(vocab_size=512, block_size=4, **dflash_overrides):
    config = Qwen3Config(
        architectures=["DFlash2DraftModel"],
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        num_target_layers=4,
        head_dim=16,
        max_position_embeddings=256,
        vocab_size=vocab_size,
        layer_types=["full_attention", "full_attention"],
        dflash_config={
            "block_size": block_size,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": vocab_size - 1,
            "selector_rank": 8,
            "selector_top_k": 16,
            "target_layer_ids": [1, 2],
            **dflash_overrides,
        },
    )
    config._attn_implementation = "sdpa"
    return config


def _online_model(device, dtype, *, env="1", **kwargs):
    torch.manual_seed(11)
    config = _draft_config()
    draft = DFlash2DraftModel(config)
    with torch.no_grad():
        # A fresh selector is a unary no-op; give it a transition to train.
        draft.candidate_selector.successor_codebook.normal_(std=0.2)
    head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    embed = nn.Embedding(config.vocab_size, config.hidden_size)
    with torch.no_grad():
        head.weight.normal_(std=0.3)
    head.requires_grad_(False)
    embed.requires_grad_(False)
    with mock.patch.dict(os.environ, {"SPECFORGE_DFLASH_FUSED_HEAD": env}):
        model = OnlineDFlashModel(
            draft_model=draft,
            target_lm_head=head,
            target_embed_tokens=embed,
            mask_token_id=config.vocab_size - 1,
            block_size=4,
            attention_backend="sdpa",
            num_anchors=12,
            **kwargs,
        )
    return model.to(device=device, dtype=dtype)


class FusedHeadGateTest(unittest.TestCase):
    def test_identity_transform_is_a_bitwise_upcast(self):
        model = DFlash2DraftModel(_draft_config())
        logits = torch.randn(3, 7, dtype=torch.bfloat16)

        self.assertTrue(model.unary_logits_transform_is_identity())
        actual = model.transform_unary_logits(logits)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(torch.equal(actual, logits.float()))

    def test_non_identity_transform_disables_the_fast_path(self):
        for overrides in (
            {"output_multiplier": 0.5},
            {"final_logit_softcapping": 30.0},
        ):
            with self.subTest(overrides=overrides):
                model = DFlash2DraftModel(_draft_config(**overrides))
                self.assertFalse(model.unary_logits_transform_is_identity())

    def test_cpu_and_env_off_models_use_the_reference_head(self):
        model = _online_model("cpu", torch.bfloat16)
        hidden = torch.zeros(1, 1, 4, 64, dtype=torch.bfloat16)
        self.assertFalse(model._use_fused_unary_head(hidden))

        disabled = _online_model("cpu", torch.bfloat16, env="0")
        self.assertFalse(disabled._fused_unary_head_requested)
        self.assertTrue(model._fused_unary_head_requested)


def _reference_head(hidden, weight, targets, topk_ids, grad_nlq, grad_topv):
    """Reference neg-log-q and input gradients from the same BF16 logits.

    Returns the eager path (autograd rounds the FP32 logit gradient to BF16
    before the BF16 input-gradient GEMM, as the reference objective does) and a
    full-FP32 backward (FP32 logit gradient and FP32 GEMM).
    """
    eager_hidden = hidden.detach().clone().requires_grad_(True)
    eager_logits = F.linear(eager_hidden, weight).float()
    neg_log_q = F.cross_entropy(eager_logits, targets, reduction="none")
    torch.autograd.backward(
        (neg_log_q, eager_logits.gather(-1, topk_ids)), (grad_nlq, grad_topv)
    )

    fp32_logits = eager_logits.detach().clone().requires_grad_(True)
    fp32_neg_log_q = F.cross_entropy(fp32_logits, targets, reduction="none")
    torch.autograd.backward(
        (fp32_neg_log_q, fp32_logits.gather(-1, topk_ids)), (grad_nlq, grad_topv)
    )
    fp32_grad = fp32_logits.grad @ weight.float()
    return neg_log_q.detach(), eager_logits.detach(), eager_hidden.grad, fp32_grad


@unittest.skipUnless(CUDA_TRITON, "fused DFlash2 head requires CUDA and Triton")
class FusedHeadKernelTest(unittest.TestCase):
    def _compare(self, vocab_size, top_k=16, rows=37, hidden_size=64):
        from specforge.core.dflash_head_triton import dflash_unary_head_fused

        torch.manual_seed(vocab_size)
        device = torch.device("cuda")
        hidden = torch.randn(rows, hidden_size, device=device, dtype=torch.bfloat16)
        weight = 0.25 * torch.randn(
            vocab_size, hidden_size, device=device, dtype=torch.bfloat16
        )
        # Duplicate head rows create exact logit ties for argmax and top-k,
        # and a zero hidden row makes every logit of that row equal.
        weight[5] = weight[5] * 4
        weight[vocab_size - 3] = weight[5]
        hidden[3] = 0
        targets = torch.randint(vocab_size, (rows,), device=device)
        targets[0] = 5
        targets[1] = vocab_size - 1
        grad_nlq = torch.rand(rows, device=device)
        grad_topv = torch.randn(rows, top_k, device=device)
        grad_nlq[::4] = 0
        grad_topv[::4] = 0
        grad_topv[1::4] = 0

        actual_hidden = hidden.clone().requires_grad_(True)
        neg_log_q, topv, topi, argmax = dflash_unary_head_fused(
            actual_hidden, weight, targets, top_k
        )
        torch.autograd.backward((neg_log_q, topv), (grad_nlq, grad_topv))
        eager_nlq, eager_logits, eager_grad, fp32_grad = _reference_head(
            hidden, weight, targets, topi, grad_nlq, grad_topv
        )

        torch.testing.assert_close(neg_log_q, eager_nlq, rtol=1e-5, atol=1e-5)
        self.assertTrue(torch.equal(argmax, eager_logits.argmax(dim=-1)))
        self.assertEqual(argmax[3].item(), 0)
        # BF16 -> FP32 is exact, so the candidate values match bit for bit.
        sorted_values = eager_logits.sort(dim=-1, descending=True).values
        boundary_tie = sorted_values[:, top_k - 1] == sorted_values[:, top_k]
        self.assertTrue(boundary_tie[3].item())
        _assert_same_topk(self, topv, topi, eager_logits, top_k)
        self.assertEqual(topi[3].tolist(), list(range(top_k)))

        zero_rows = (grad_nlq == 0) & (grad_topv == 0).all(dim=-1)
        self.assertTrue(zero_rows.any())
        self.assertTrue(torch.all(actual_hidden.grad[zero_rows] == 0))
        # The BF16 logit gradient and BF16 GEMM round like the eager path; the
        # remaining error against the FP32 backward is that shared rounding.
        scale = fp32_grad.abs().max()
        torch.testing.assert_close(
            actual_hidden.grad.float(), fp32_grad, rtol=2e-2, atol=5e-3 * scale
        )
        self.assertLess(
            ((actual_hidden.grad.float() - fp32_grad).norm() / fp32_grad.norm()).item(),
            5e-3,
        )
        torch.testing.assert_close(actual_hidden.grad, eager_grad, rtol=1e-2, atol=1e-3)

    def test_matches_reference_for_odd_vocabularies(self):
        for vocab_size in (1000, 50003):
            with self.subTest(vocab_size=vocab_size):
                self._compare(vocab_size)

    def test_wide_candidate_sets_and_retained_graphs(self):
        from specforge.core.dflash_head_triton import (
            MAX_FUSED_TOP_K,
            dflash_unary_head_fused,
        )

        torch.manual_seed(4)
        device = torch.device("cuda")
        weight = torch.randn(3001, 32, device=device, dtype=torch.bfloat16)
        targets = torch.randint(3001, (19,), device=device)
        for top_k in (MAX_FUSED_TOP_K, MAX_FUSED_TOP_K + 36):
            with self.subTest(top_k=top_k):
                hidden = torch.randn(
                    19, 32, device=device, dtype=torch.bfloat16, requires_grad=True
                )
                neg_log_q, topv, topi, _ = dflash_unary_head_fused(
                    hidden, weight, targets, top_k
                )
                reference_values, reference_ids = (
                    F.linear(hidden.detach(), weight).float().topk(top_k, dim=-1)
                )
                self.assertTrue(torch.equal(topv, reference_values))
                self.assertTrue(
                    torch.equal(
                        topi.sort(dim=-1).values, reference_ids.sort(dim=-1).values
                    )
                )
                loss = neg_log_q.sum() + topv.sum()
                loss.backward(retain_graph=True)
                with self.assertRaisesRegex(RuntimeError, "modified by an inplace"):
                    loss.backward()

    def test_argmax_and_neg_log_q_without_topk(self):
        from specforge.core.dflash_head_triton import dflash_unary_head_fused

        torch.manual_seed(3)
        device = torch.device("cuda")
        hidden = torch.randn(
            2, 5, 32, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        weight = torch.randn(777, 32, device=device, dtype=torch.bfloat16)
        targets = torch.randint(777, (2, 5), device=device)

        neg_log_q, topv, topi, argmax = dflash_unary_head_fused(
            hidden, weight, targets, 0
        )
        neg_log_q.sum().backward()

        reference = hidden.detach().clone().requires_grad_(True)
        logits = F.linear(reference, weight).float()
        expected = F.cross_entropy(
            logits.reshape(-1, 777), targets.reshape(-1), reduction="none"
        ).reshape(2, 5)
        expected.sum().backward()
        self.assertEqual(tuple(topv.shape), (2, 5, 0))
        self.assertEqual(tuple(topi.shape), (2, 5, 0))
        torch.testing.assert_close(neg_log_q, expected, rtol=1e-5, atol=1e-5)
        self.assertTrue(torch.equal(argmax, logits.argmax(dim=-1)))
        torch.testing.assert_close(hidden.grad, reference.grad, rtol=1e-2, atol=1e-3)

    def test_topk_matches_fp32_topk_at_full_vocabulary(self):
        """Random and peaked BF16 rows over the 248k Qwen3.8 vocabulary."""
        from specforge.core.dflash_head_triton import _launch_stats

        torch.manual_seed(5)
        device = torch.device("cuda")
        rows, vocab_size, top_k = 96, 248320, 16
        random_logits = 1.5 * torch.randn(rows, vocab_size, device=device)
        # A few dominant tokens over a long flat tail, as a trained head emits.
        peaked_logits = 2.0 * torch.randn(rows, vocab_size, device=device)
        hot = torch.randint(vocab_size, (rows, 8), device=device)
        peaked_logits.scatter_(1, hot, 15.0 + 5 * torch.rand(rows, 8, device=device))
        for name, logits in (("random", random_logits), ("peaked", peaked_logits)):
            with self.subTest(logits=name):
                logits = logits.to(torch.bfloat16)
                targets = torch.randint(vocab_size, (rows,), device=device)
                lse = torch.empty(rows, device=device)
                target_logit = torch.empty_like(lse)
                argmax = torch.empty(rows, device=device, dtype=torch.long)
                topv = torch.empty(rows, top_k, device=device)
                topi = torch.empty(rows, top_k, device=device, dtype=torch.long)

                _launch_stats(logits, targets, lse, target_logit, argmax, topv, topi)

                reference = logits.float()
                torch.testing.assert_close(
                    lse, torch.logsumexp(reference, dim=-1), rtol=1e-6, atol=1e-5
                )
                self.assertTrue(torch.equal(argmax, reference.argmax(dim=-1)))
                self.assertTrue(
                    torch.equal(
                        target_logit, reference.gather(1, targets[:, None])[:, 0]
                    )
                )
                _assert_same_topk(self, topv, topi, reference, top_k)


def _assert_same_topk(test, values, ids, reference_logits, top_k):
    """Assert ``torch.topk``'s values and candidate sets, lowest index on ties.

    The fused kernel orders equal values by ascending index; ``torch.topk``'s
    sort leaves equal values in an unspecified order, so ids compare as sets.
    """
    reference_values, reference_ids = reference_logits.topk(top_k, dim=-1)
    test.assertTrue(torch.equal(values, reference_values))
    test.assertTrue(
        torch.equal(ids.sort(dim=-1).values, reference_ids.sort(dim=-1).values)
    )
    test.assertTrue(torch.equal(reference_logits.gather(-1, ids), values))
    equal_neighbors = values[:, 1:] == values[:, :-1]
    test.assertTrue(
        torch.all(ids[:, 1:][equal_neighbors] > ids[:, :-1][equal_neighbors])
    )


def _run_model(model, batch, seed):
    torch.manual_seed(seed)
    model.zero_grad(set_to_none=True)
    loss, accuracy, metrics = model(**batch)
    loss.backward()
    grads = {
        name: parameter.grad.detach().float().clone()
        for name, parameter in model.draft_model.named_parameters()
        if parameter.grad is not None
    }
    return loss.detach().float(), accuracy.detach(), metrics, grads


@unittest.skipUnless(CUDA_TRITON, "fused DFlash2 head requires CUDA and Triton")
class FusedHeadModelParityTest(unittest.TestCase):
    def _batch(self, device, width):
        torch.manual_seed(21)
        input_ids = torch.randint(0, 500, (2, 48), device=device)
        loss_mask = torch.ones(2, 48, device=device)
        loss_mask[0, :5] = 0
        loss_mask[1, 30:] = 0
        return {
            "input_ids": input_ids,
            "hidden_states": torch.randn(
                2, 48, width, device=device, dtype=torch.bfloat16
            ),
            "loss_mask": loss_mask,
        }

    def test_fused_head_matches_reference_objective(self):
        device = torch.device("cuda")
        cases = [
            {"loss_type": "dflash", "loss_decay_gamma": 7.0},
            {"loss_type": "dpace"},
            {"loss_type": "dflash", "lk_loss_type": "lambda"},
        ]
        for case in cases:
            for stop_gradient in (False, True):
                for chunk_blocks in (0, 2):
                    kwargs = dict(
                        case,
                        selector_stop_gradient=stop_gradient,
                        objective_chunk_blocks=chunk_blocks,
                    )
                    with self.subTest(**kwargs):
                        self._compare(device, kwargs)

    def _compare(self, device, kwargs):
        fused = _online_model(device, torch.bfloat16, env="1", **kwargs)
        reference = _online_model(device, torch.bfloat16, env="0", **kwargs)
        batch = self._batch(device, 2 * 64)
        self.assertTrue(
            fused._use_fused_unary_head(
                torch.zeros(1, 1, 4, 64, device=device, dtype=torch.bfloat16)
            )
        )
        self.assertFalse(
            reference._use_fused_unary_head(
                torch.zeros(1, 1, 4, 64, device=device, dtype=torch.bfloat16)
            )
        )

        fused_loss, fused_acc, fused_metrics, fused_grads = _run_model(
            fused, batch, seed=5
        )
        ref_loss, ref_acc, ref_metrics, ref_grads = _run_model(reference, batch, seed=5)

        torch.testing.assert_close(fused_loss, ref_loss, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(fused_acc, ref_acc, rtol=1e-4, atol=1e-6)
        self.assertEqual(
            set(fused_metrics["ratio_metrics"]), set(ref_metrics["ratio_metrics"])
        )
        for name, (numerator, denominator) in ref_metrics["ratio_metrics"].items():
            actual_numerator, actual_denominator = fused_metrics["ratio_metrics"][name]
            torch.testing.assert_close(
                torch.as_tensor(actual_numerator).float(),
                torch.as_tensor(numerator).float(),
                rtol=1e-4,
                atol=1e-5,
                msg=lambda message, name=name: f"{name}: {message}",
            )
            torch.testing.assert_close(
                torch.as_tensor(actual_denominator).float(),
                torch.as_tensor(denominator).float(),
                rtol=1e-4,
                atol=1e-5,
                msg=lambda message, name=name: f"{name}: {message}",
            )
        self.assertEqual(set(fused_grads), set(ref_grads))
        for name, expected in ref_grads.items():
            actual = fused_grads[name]
            error = (actual - expected).norm() / expected.norm().clamp_min(1e-12)
            self.assertLess(error.item(), 1e-3, name)


if __name__ == "__main__":
    unittest.main()
