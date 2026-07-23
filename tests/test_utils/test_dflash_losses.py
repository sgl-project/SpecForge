# coding=utf-8
"""Formula-level tests for DFlash/D-PACE/DSpark loss behavior.

The tests load the algorithm-family models with a lightweight draft model stub
so they can run on CPU without importing the full modeling stack.
They still exercise the online wrappers: anchor sampling, draft
output, and LM head output are made deterministic.
"""

import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]

_stub_dflash_draft = types.ModuleType("specforge.modeling.draft.dflash")


class _DFlashDraftStub(nn.Module):
    pass


_stub_dflash_draft.DFlashDraftModel = _DFlashDraftStub

_spec = importlib.util.spec_from_file_location(
    "specforge.algorithms.common.dflash_family_model",
    REPO / "specforge" / "algorithms" / "common" / "dflash_family_model.py",
)
_dflash_module = importlib.util.module_from_spec(_spec)

_pkg_specforge = types.ModuleType("specforge")
_pkg_specforge.__path__ = [str(REPO / "specforge")]
_pkg_algorithms = types.ModuleType("specforge.algorithms")
_pkg_algorithms.__path__ = [str(REPO / "specforge" / "algorithms")]
_pkg_common = types.ModuleType("specforge.algorithms.common")
_pkg_common.__path__ = [str(REPO / "specforge" / "algorithms" / "common")]
_pkg_modeling = types.ModuleType("specforge.modeling")
_pkg_modeling.__path__ = [str(REPO / "specforge" / "modeling")]
_pkg_draft = types.ModuleType("specforge.modeling.draft")
_pkg_draft.__path__ = [str(REPO / "specforge" / "modeling" / "draft")]

with patch.dict(
    sys.modules,
    {
        "specforge": _pkg_specforge,
        "specforge.algorithms": _pkg_algorithms,
        "specforge.algorithms.common": _pkg_common,
        "specforge.algorithms.common.dflash_family_model": _dflash_module,
        "specforge.modeling": _pkg_modeling,
        "specforge.modeling.draft": _pkg_draft,
        "specforge.modeling.draft.dflash": _stub_dflash_draft,
    },
):
    _spec.loader.exec_module(_dflash_module)
OnlineDFlashModel = _dflash_module.OnlineDFlashModel
OnlineDSparkModel = _dflash_module.OnlineDSparkModel


class _FixedDraft(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, position_ids, noise_embedding, target_hidden, attention_mask):
        bsz, draft_len = noise_embedding.shape[:2]
        return torch.zeros(
            bsz,
            draft_len,
            self.hidden_size,
            dtype=noise_embedding.dtype,
            device=noise_embedding.device,
        )

    def apply_logits_head(self, base_logits, **kwargs):
        del kwargs
        return base_logits

    def predict_confidence(self, hidden_states, prev_token_ids=None):
        del hidden_states, prev_token_ids
        return None


class _FixedDSparkDraft(_FixedDraft):
    def __init__(self, hidden_size: int, confidence_logits: torch.Tensor = None):
        super().__init__(hidden_size)
        if confidence_logits is not None:
            self.register_buffer("confidence_logits", confidence_logits)
        else:
            self.confidence_logits = None

    def apply_logits_head(self, base_logits, prev_token_ids, hidden_states):
        del prev_token_ids, hidden_states
        return base_logits

    def predict_confidence(self, hidden_states, prev_token_ids=None):
        del prev_token_ids
        if self.confidence_logits is None:
            return None
        return self.confidence_logits.to(device=hidden_states.device)


class _LearnableDSparkDraft(_FixedDraft):
    def __init__(self, hidden_size: int):
        super().__init__(hidden_size)
        self.signal = nn.Parameter(torch.linspace(-0.2, 0.2, hidden_size))
        self.confidence_head = nn.Identity()

    def forward(self, position_ids, noise_embedding, target_hidden, attention_mask):
        del position_ids, target_hidden, attention_mask
        bsz, draft_len = noise_embedding.shape[:2]
        return self.signal.view(1, 1, -1).expand(bsz, draft_len, -1)

    def apply_logits_head(self, base_logits, prev_token_ids, hidden_states):
        del prev_token_ids, hidden_states
        return base_logits

    def predict_confidence(self, hidden_states, prev_token_ids=None):
        del prev_token_ids
        return hidden_states[..., 0]


class _FixedHead(nn.Module):
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self.register_buffer("fixed_logits", logits)

    def forward(self, hidden_states):
        return self.fixed_logits.to(device=hidden_states.device)


class _DualFixedHead(nn.Module):
    def __init__(self, draft_logits: torch.Tensor, target_logits: torch.Tensor):
        super().__init__()
        self.register_buffer("draft_logits", draft_logits)
        self.register_buffer("target_logits", target_logits)
        self._calls = 0

    def forward(self, hidden_states):
        use_target = self._calls % 2 == 1
        self._calls += 1
        if use_target:
            return self.target_logits.to(device=hidden_states.device)
        bsz, n_blocks, block_size, vocab_size = self.draft_logits.shape
        return self.draft_logits.reshape(bsz, n_blocks * block_size, vocab_size).to(
            device=hidden_states.device
        )


def _fixed_noise_embed(self, input_ids, anchor_positions, block_keep_mask):
    bsz, n_blocks = anchor_positions.shape
    return torch.zeros(
        bsz,
        n_blocks * self.block_size,
        self.embed_tokens.embedding_dim,
        dtype=torch.double,
        device=input_ids.device,
    )


def _fixed_anchor_sampler(anchors, keep_mask):
    def _sample(self, seq_len, loss_mask, device):
        return anchors.to(device), keep_mask.to(device)

    return _sample


def _make_model(logits, anchors, keep_mask, draft_model=None, lm_head=None, **kwargs):
    bsz, n_blocks, block_size, vocab_size = logits.shape
    model = OnlineDFlashModel(
        draft_model=draft_model or _FixedDraft(hidden_size=4),
        target_lm_head=lm_head
        or _FixedHead(logits.reshape(bsz, n_blocks * block_size, vocab_size)),
        target_embed_tokens=nn.Embedding(vocab_size, 4).double(),
        mask_token_id=0,
        block_size=block_size,
        attention_backend="sdpa",
        num_anchors=n_blocks,
        **kwargs,
    ).double()
    model._sample_anchor_positions = types.MethodType(
        _fixed_anchor_sampler(anchors, keep_mask), model
    )
    model._create_noise_embed = types.MethodType(_fixed_noise_embed, model)
    return model


def _make_dspark_model(
    logits, anchors, keep_mask, draft_model=None, lm_head=None, **kwargs
):
    bsz, n_blocks, block_size, vocab_size = logits.shape
    model = OnlineDSparkModel(
        draft_model=draft_model or _FixedDraft(hidden_size=4),
        target_lm_head=lm_head
        or _FixedHead(logits.reshape(bsz, n_blocks * block_size, vocab_size)),
        target_embed_tokens=nn.Embedding(vocab_size, 4).double(),
        mask_token_id=0,
        block_size=block_size,
        attention_backend="sdpa",
        num_anchors=n_blocks,
        **kwargs,
    ).double()
    model._sample_anchor_positions = types.MethodType(
        _fixed_anchor_sampler(anchors, keep_mask), model
    )
    model._create_noise_embed = types.MethodType(_fixed_noise_embed, model)
    return model


def _sample_tensors():
    torch.manual_seed(123)
    bsz, n_blocks, block_size, vocab_size = 2, 2, 5, 13
    seq_len = 9
    logits = torch.randn(bsz, n_blocks, block_size, vocab_size, dtype=torch.double)
    input_ids = torch.tensor(
        [
            [1, 4, 2, 8, 3, 7, 5, 6, 9],
            [2, 5, 1, 4, 7, 3, 8, 10, 11],
        ],
        dtype=torch.long,
    )
    loss_mask = torch.ones(bsz, seq_len, dtype=torch.double)
    loss_mask[0, 7] = 0.0
    loss_mask[1, 6] = 0.0
    anchors = torch.tensor([[0, 3], [1, 4]], dtype=torch.long)
    keep_mask = torch.tensor([[True, True], [True, False]])
    hidden_states = torch.zeros(bsz, seq_len, 4, dtype=torch.double)
    return logits, input_ids, loss_mask, hidden_states, anchors, keep_mask


def _targets_and_mask(input_ids, loss_mask, anchors, keep_mask, block_size):
    bsz, seq_len = input_ids.shape
    n_blocks = anchors.shape[1]
    offsets = torch.arange(block_size).view(1, 1, -1)
    label_indices = anchors.unsqueeze(-1) + offsets
    safe_indices = label_indices.clamp(max=seq_len - 1)
    targets = torch.gather(
        input_ids.unsqueeze(1).expand(-1, n_blocks, -1),
        2,
        safe_indices,
    )
    binary_mask = keep_mask.unsqueeze(-1).expand(-1, -1, block_size).double()
    binary_mask = binary_mask * (label_indices < seq_len).double()
    binary_mask = binary_mask * (offsets > 0).double()
    gathered_loss_mask = torch.gather(
        loss_mask.unsqueeze(1).expand(-1, n_blocks, -1),
        2,
        safe_indices,
    )
    binary_mask = binary_mask * gathered_loss_mask
    return targets, binary_mask


def _dspark_targets_and_mask(input_ids, loss_mask, anchors, keep_mask, block_size):
    bsz, seq_len = input_ids.shape
    n_blocks = anchors.shape[1]
    offsets = torch.arange(1, block_size + 1).view(1, 1, -1)
    label_indices = anchors.unsqueeze(-1) + offsets
    safe_indices = label_indices.clamp(max=seq_len - 1)
    safe_indices = torch.where(
        keep_mask.unsqueeze(-1),
        safe_indices,
        torch.zeros_like(safe_indices),
    )
    targets = torch.gather(
        input_ids.unsqueeze(1).expand(-1, n_blocks, -1),
        2,
        safe_indices,
    )
    gathered_loss_mask = torch.gather(
        loss_mask.unsqueeze(1).expand(-1, n_blocks, -1),
        2,
        safe_indices,
    )
    eval_mask = (label_indices < seq_len) & (gathered_loss_mask > 0.5)
    eval_mask = eval_mask & keep_mask.unsqueeze(-1)
    eval_mask = eval_mask.to(torch.int32).cumprod(dim=-1).bool()
    return targets, eval_mask


def _neg_log_q(logits, targets):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)


def _naive_dpace_weight(prob, binary_mask, alpha, loss_type):
    smooth = (1.0 - alpha) * prob + alpha
    smooth = torch.where(binary_mask > 0, smooth, torch.ones_like(smooth))
    prefix = torch.cumprod(smooth, dim=-1)
    if loss_type == "dpace-cumulative-confidence-only":
        return prefix
    suffix = torch.flip(
        torch.cumsum(torch.flip(prefix * binary_mask, dims=[-1]), dim=-1),
        dims=[-1],
    )
    if loss_type == "dpace":
        return suffix
    if loss_type == "dpace-continuation-value-only":
        return suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
    raise ValueError(loss_type)


def _naive_dflash_loss(neg_log_q, binary_mask, gamma):
    weight = binary_mask
    if gamma is not None and gamma > 0:
        block_size = neg_log_q.shape[-1]
        positions = torch.arange(block_size, dtype=neg_log_q.dtype).view(1, 1, -1)
        decay = torch.exp(-(positions - 1).clamp(min=0) / gamma)
        weight = weight * decay
    return (neg_log_q * weight).sum() / (weight.sum() + 1e-6)


class TestDFlashLosses(unittest.TestCase):
    def setUp(self):
        (
            self.logits,
            self.input_ids,
            self.loss_mask,
            self.hidden_states,
            self.anchors,
            self.keep_mask,
        ) = _sample_tensors()
        self.targets, self.binary_mask = _targets_and_mask(
            self.input_ids,
            self.loss_mask,
            self.anchors,
            self.keep_mask,
            self.logits.shape[2],
        )
        self.neg_log_q = _neg_log_q(self.logits, self.targets)
        self.q = torch.exp(-self.neg_log_q)

    def _forward_loss(self, **kwargs):
        model = _make_model(self.logits, self.anchors, self.keep_mask, **kwargs)
        loss, accuracy, _metrics = model(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(accuracy))
        return loss

    def test_dflash_default_matches_existing_weighted_mean(self):
        got = self._forward_loss()
        want = _naive_dflash_loss(self.neg_log_q, self.binary_mask, gamma=None)
        torch.testing.assert_close(got, want, rtol=0, atol=1e-10)

    def test_dflash_decay_gamma_is_preserved(self):
        gamma = 7.0
        got = self._forward_loss(loss_type="dflash", loss_decay_gamma=gamma)
        want = _naive_dflash_loss(self.neg_log_q, self.binary_mask, gamma=gamma)
        torch.testing.assert_close(got, want, rtol=0, atol=1e-8)

    def test_dpace_full_matches_naive_reference(self):
        alpha = 0.5
        got = self._forward_loss(loss_type="dpace", dpace_alpha=alpha)
        weight = _naive_dpace_weight(self.q, self.binary_mask, alpha, "dpace")
        want = (self.neg_log_q * weight * self.binary_mask).sum() / float(
            self.input_ids.shape[0]
        )
        torch.testing.assert_close(got, want, rtol=0, atol=1e-10)

    def test_cumulative_confidence_ablation_matches_naive_reference(self):
        alpha = 0.5
        got = self._forward_loss(
            loss_type="dpace-cumulative-confidence-only", dpace_alpha=alpha
        )
        weight = _naive_dpace_weight(
            self.q,
            self.binary_mask,
            alpha,
            "dpace-cumulative-confidence-only",
        )
        want = (self.neg_log_q * weight * self.binary_mask).sum() / float(
            self.input_ids.shape[0]
        )
        torch.testing.assert_close(got, want, rtol=0, atol=1e-10)

    def test_continuation_value_ablation_matches_naive_reference(self):
        alpha = 0.5
        got = self._forward_loss(
            loss_type="dpace-continuation-value-only", dpace_alpha=alpha
        )
        weight = _naive_dpace_weight(
            self.q,
            self.binary_mask,
            alpha,
            "dpace-continuation-value-only",
        )
        want = (self.neg_log_q * weight * self.binary_mask).sum() / float(
            self.input_ids.shape[0]
        )
        torch.testing.assert_close(got, want, rtol=0, atol=1e-10)

    def test_dpace_loss_reduces_by_batch_size(self):
        alpha = 0.5
        got = self._forward_loss(loss_type="dpace", dpace_alpha=alpha)
        weight = _naive_dpace_weight(self.q, self.binary_mask, alpha, "dpace")
        weighted_sum = (self.neg_log_q * weight * self.binary_mask).sum()
        token_count_loss = weighted_sum / ((weight * self.binary_mask).sum() + 1e-6)
        batch_loss = weighted_sum / float(self.input_ids.shape[0])
        torch.testing.assert_close(got, batch_loss, rtol=0, atol=1e-10)
        self.assertFalse(torch.allclose(got, token_count_loss))

    def test_alpha_changes_dpace_loss(self):
        low_alpha = self._forward_loss(loss_type="dpace", dpace_alpha=0.1)
        high_alpha = self._forward_loss(loss_type="dpace", dpace_alpha=0.9)
        self.assertNotAlmostEqual(low_alpha.item(), high_alpha.item(), places=8)

    def test_dflash_family_chunking_matches_full_loss_metrics_and_gradients(self):
        for loss_type in (
            "dflash",
            "dpace",
            "dpace-cumulative-confidence-only",
            "dpace-continuation-value-only",
        ):
            with self.subTest(loss_type=loss_type):
                torch.manual_seed(77)
                head = nn.Linear(4, self.logits.shape[-1], bias=False).double()
                full = _make_model(
                    self.logits,
                    self.anchors,
                    self.keep_mask,
                    draft_model=_LearnableDSparkDraft(4).double(),
                    lm_head=head,
                    loss_type=loss_type,
                    loss_decay_gamma=3.0,
                    objective_chunk_blocks=0,
                )
                chunked = _make_model(
                    self.logits,
                    self.anchors,
                    self.keep_mask,
                    draft_model=_LearnableDSparkDraft(4).double(),
                    lm_head=copy.deepcopy(head),
                    loss_type=loss_type,
                    loss_decay_gamma=3.0,
                    objective_chunk_blocks=1,
                )
                chunked.load_state_dict(full.state_dict())

                full_loss, full_accuracy, full_metrics = full(
                    input_ids=self.input_ids,
                    hidden_states=self.hidden_states,
                    loss_mask=self.loss_mask,
                )
                chunked_loss, chunked_accuracy, chunked_metrics = chunked(
                    input_ids=self.input_ids,
                    hidden_states=self.hidden_states,
                    loss_mask=self.loss_mask,
                )

                torch.testing.assert_close(
                    chunked_loss,
                    full_loss,
                    rtol=1e-6,
                    atol=1e-8,
                )
                torch.testing.assert_close(chunked_accuracy, full_accuracy)
                torch.testing.assert_close(
                    chunked_metrics["accuracy_denom"],
                    full_metrics["accuracy_denom"],
                )

                full_loss.backward()
                chunked_loss.backward()
                torch.testing.assert_close(
                    chunked.draft_model.signal.grad,
                    full.draft_model.signal.grad,
                    rtol=1e-6,
                    atol=1e-7,
                )
                torch.testing.assert_close(
                    chunked.lm_head.weight.grad,
                    full.lm_head.weight.grad,
                    rtol=1e-6,
                    atol=1e-7,
                )

    def test_invalid_loss_type_rejected(self):
        with self.assertRaisesRegex(ValueError, "loss_type"):
            _make_model(
                self.logits,
                self.anchors,
                self.keep_mask,
                loss_type="topk_mask",
            )

    def test_invalid_dpace_alpha_rejected(self):
        with self.assertRaisesRegex(ValueError, "dpace_alpha"):
            _make_model(
                self.logits,
                self.anchors,
                self.keep_mask,
                loss_type="dpace",
                dpace_alpha=1.5,
            )

    def test_dflash_draft_stub_does_not_leak_to_sys_modules(self):
        self.assertIsNot(
            sys.modules.get("specforge.modeling.draft.dflash"),
            _stub_dflash_draft,
        )

    def test_dspark_ce_only_uses_next_token_labels_and_contiguous_mask(self):
        model = _make_dspark_model(
            self.logits,
            self.anchors,
            self.keep_mask,
            dspark_ce_loss_alpha=1.0,
            dspark_l1_loss_alpha=0.0,
            dspark_confidence_head_alpha=0.0,
        )
        loss, accuracy, _metrics = model(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
        )
        self.assertTrue(torch.isfinite(accuracy))
        targets, eval_mask = _dspark_targets_and_mask(
            self.input_ids,
            self.loss_mask,
            self.anchors,
            self.keep_mask,
            self.logits.shape[2],
        )
        neg_log_q = _neg_log_q(self.logits, targets)
        weights = eval_mask.double()
        want = (neg_log_q * weights).sum() / weights.sum()
        torch.testing.assert_close(loss, want, rtol=0, atol=1e-10)
        acc_num, acc_den = _metrics["ratio_metrics"]["acc"]
        torch.testing.assert_close(acc_den, weights.sum().float())
        torch.testing.assert_close(accuracy, acc_num / acc_den)
        self.assertIn("ce_position", _metrics["ratio_metrics"])

    def test_dspark_ce_only_skips_target_distribution(self):
        target_logits = torch.randn_like(self.logits)
        model = _make_dspark_model(
            self.logits,
            self.anchors,
            self.keep_mask,
            lm_head=_DualFixedHead(self.logits, target_logits).double(),
            dspark_ce_loss_alpha=1.0,
            dspark_l1_loss_alpha=0.0,
            dspark_confidence_head_alpha=0.0,
        )

        with patch.object(torch, "softmax", side_effect=AssertionError("unexpected")):
            loss, _accuracy, _metrics = model(
                input_ids=self.input_ids,
                hidden_states=self.hidden_states,
                loss_mask=self.loss_mask,
                target_last_hidden_states=torch.zeros_like(self.hidden_states),
            )

        self.assertTrue(torch.isfinite(loss))

    def test_dspark_l1_and_confidence_match_token_pooled_objective(self):
        torch.manual_seed(321)
        target_logits = torch.randn_like(self.logits)
        confidence_logits = torch.randn(
            self.logits.shape[:3],
            dtype=torch.double,
        )
        draft = _FixedDSparkDraft(
            hidden_size=4,
            confidence_logits=confidence_logits,
        ).double()
        head = _DualFixedHead(self.logits, target_logits).double()
        model = _make_dspark_model(
            self.logits,
            self.anchors,
            self.keep_mask,
            draft_model=draft,
            lm_head=head,
            dspark_ce_loss_alpha=0.1,
            dspark_l1_loss_alpha=0.9,
            dspark_confidence_head_alpha=1.0,
        )
        loss, _accuracy, metrics = model(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
            target_last_hidden_states=torch.zeros_like(self.hidden_states),
        )
        targets, eval_mask = _dspark_targets_and_mask(
            self.input_ids,
            self.loss_mask,
            self.anchors,
            self.keep_mask,
            self.logits.shape[2],
        )
        weights = eval_mask.double()
        denominator = weights.sum()
        ce = (_neg_log_q(self.logits, targets) * weights).sum() / denominator
        draft_probs = torch.softmax(self.logits.float(), dim=-1)
        target_probs = torch.softmax(target_logits.float(), dim=-1)
        l1_dist = (draft_probs - target_probs).abs().sum(dim=-1).double()
        l1 = (l1_dist * weights).sum() / denominator
        accept_rate = (1.0 - 0.5 * l1_dist).clamp(0.0, 1.0)
        conf = F.binary_cross_entropy_with_logits(
            confidence_logits.float(),
            accept_rate.float(),
            reduction="none",
        ).double()
        conf = (conf * weights).sum() / denominator
        want = 0.1 * ce + 0.9 * l1 + conf
        torch.testing.assert_close(loss, want, rtol=0, atol=1e-6)
        ratio_metrics = metrics["ratio_metrics"]
        ce_num, ce_den = ratio_metrics["ce_loss"]
        l1_num, l1_den = ratio_metrics["l1_loss"]
        confidence_num, confidence_den = ratio_metrics["confidence_loss"]
        torch.testing.assert_close(ce_num / ce_den, ce, rtol=0, atol=1e-6)
        torch.testing.assert_close(
            l1_num / l1_den, l1, rtol=0, atol=1e-6, check_dtype=False
        )
        torch.testing.assert_close(
            confidence_num / confidence_den,
            conf,
            rtol=0,
            atol=1e-6,
            check_dtype=False,
        )

    def test_dspark_requires_target_hidden_states_for_l1_or_confidence(self):
        model = _make_dspark_model(
            self.logits,
            self.anchors,
            self.keep_mask,
            dspark_ce_loss_alpha=0.1,
            dspark_l1_loss_alpha=0.9,
            dspark_confidence_head_alpha=0.0,
        )
        with self.assertRaisesRegex(ValueError, "target_last_hidden_states"):
            model(
                input_ids=self.input_ids,
                hidden_states=self.hidden_states,
                loss_mask=self.loss_mask,
            )

    def test_dspark_chunking_matches_full_loss_and_gradient(self):
        torch.manual_seed(99)
        head = nn.Linear(4, self.logits.shape[-1], bias=False).double()
        full = _make_dspark_model(
            self.logits,
            self.anchors,
            self.keep_mask,
            draft_model=_LearnableDSparkDraft(4).double(),
            lm_head=head,
            dspark_ce_loss_alpha=0.1,
            dspark_l1_loss_alpha=0.9,
            dspark_confidence_head_alpha=1.0,
            objective_chunk_blocks=0,
        )
        chunked = _make_dspark_model(
            self.logits,
            self.anchors,
            self.keep_mask,
            draft_model=_LearnableDSparkDraft(4).double(),
            lm_head=copy.deepcopy(head),
            dspark_ce_loss_alpha=0.1,
            dspark_l1_loss_alpha=0.9,
            dspark_confidence_head_alpha=1.0,
            objective_chunk_blocks=1,
        )
        chunked.load_state_dict(full.state_dict())
        target_hidden = torch.randn_like(self.hidden_states)

        full_loss, _full_acc, full_metrics = full(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
            target_last_hidden_states=target_hidden,
        )
        chunked_loss, _chunked_acc, chunked_metrics = chunked(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
            target_last_hidden_states=target_hidden,
        )
        torch.testing.assert_close(chunked_loss, full_loss, rtol=1e-6, atol=1e-8)
        for name, full_pair in full_metrics["ratio_metrics"].items():
            chunked_pair = chunked_metrics["ratio_metrics"][name]
            torch.testing.assert_close(chunked_pair[0], full_pair[0])
            torch.testing.assert_close(chunked_pair[1], full_pair[1])

        full_loss.backward()
        chunked_loss.backward()
        torch.testing.assert_close(
            chunked.draft_model.signal.grad,
            full.draft_model.signal.grad,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_dspark_ce_only_chunking_recomputes(self):
        torch.manual_seed(101)
        head = nn.Linear(4, self.logits.shape[-1], bias=False).double()
        full = _make_dspark_model(
            self.logits,
            self.anchors,
            self.keep_mask,
            draft_model=_LearnableDSparkDraft(4).double(),
            lm_head=head,
            dspark_ce_loss_alpha=1.0,
            dspark_l1_loss_alpha=0.0,
            dspark_confidence_head_alpha=0.0,
            objective_chunk_blocks=0,
        )
        chunked = _make_dspark_model(
            self.logits,
            self.anchors,
            self.keep_mask,
            draft_model=_LearnableDSparkDraft(4).double(),
            lm_head=copy.deepcopy(head),
            dspark_ce_loss_alpha=1.0,
            dspark_l1_loss_alpha=0.0,
            dspark_confidence_head_alpha=0.0,
            objective_chunk_blocks=1,
        )
        chunked.load_state_dict(full.state_dict())

        full_loss, _full_acc, _full_metrics = full(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
        )
        chunked_loss, _chunked_acc, _chunked_metrics = chunked(
            input_ids=self.input_ids,
            hidden_states=self.hidden_states,
            loss_mask=self.loss_mask,
        )
        torch.testing.assert_close(chunked_loss, full_loss, rtol=1e-6, atol=1e-8)
        full_loss.backward()
        chunked_loss.backward()
        torch.testing.assert_close(
            chunked.draft_model.signal.grad,
            full.draft_model.signal.grad,
            rtol=1e-6,
            atol=1e-7,
        )

    def test_dspark_sampler_keeps_sparse_high_index_anchor(self):
        model = _make_dspark_model(
            self.logits,
            self.anchors,
            self.keep_mask,
        )
        model.num_anchors = 2
        sparse_mask = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        anchors, keep = OnlineDSparkModel._sample_anchor_positions(
            model,
            seq_len=6,
            loss_mask=sparse_mask,
            device=sparse_mask.device,
        )
        self.assertEqual(anchors[0, 0].item(), 4)
        self.assertEqual(keep[0].tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
