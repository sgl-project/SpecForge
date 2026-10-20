"""Sampled-anchor walks reuse causal predictions without changing training."""

import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from specforge.algorithms.common import dflash_family_model as family_model
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.modeling.draft.domino import DominoDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import (
    DFlashTrainStrategy,
    DominoTrainStrategy,
    DSparkTrainStrategy,
    StepContext,
)

MODEL_KINDS = (
    "dflash",
    "dflash2",
    "domino",
    "domino_shifted",
    "dspark",
    "dspark_vanilla",
    "dspark_gated",
    "dspark_rnn",
)


def _make_case(kind):
    torch.manual_seed(835)
    family = kind.split("_")[0]
    draft_class, wrapper, strategy_class = {
        "dflash": (
            DFlashDraftModel,
            family_model.OnlineDFlashModel,
            DFlashTrainStrategy,
        ),
        "dflash2": (
            DFlash2DraftModel,
            family_model.OnlineDFlashModel,
            DFlashTrainStrategy,
        ),
        "domino": (
            DominoDraftModel,
            family_model.OnlineDominoModel,
            DominoTrainStrategy,
        ),
        "dspark": (
            DSparkDraftModel,
            family_model.OnlineDSparkModel,
            DSparkTrainStrategy,
        ),
    }[family]
    shifted = kind == "domino_shifted"
    config = SimpleNamespace(hidden_size=8, vocab_size=8, initializer_range=0.02)
    config.dflash_config = dict(
        selector_rank=2,
        selector_top_k=2,
        emb_dim=4,
        gru_hidden_dim=4,
        shift_label=shifted,
        markov_rank=4 if "_" in kind else 0,
        markov_head_type=kind.split("_")[-1],
    )
    # Only the backbone output is fixed; all auxiliary heads are production modules.
    draft = draft_class.__new__(draft_class)
    nn.Module.__init__(draft)
    draft.config, draft.block_size = config, 4
    draft._init_draft_head(config, config.dflash_config)
    options = (
        dict(dspark_l1_loss_alpha=0, dspark_confidence_head_alpha=0)
        if family == "dspark"
        else {}
    )
    if family == "domino":
        options["shift_label"] = shifted
    model = wrapper(
        draft,
        nn.Identity(),
        nn.Embedding(8, 8),
        0,
        block_size=4,
        attention_backend="sdpa",
        **options,
    ).double()
    model.embed_tokens.requires_grad_(False)
    anchors = torch.tensor([[0, 2, 5, 8, 11, 14], [1, 3, 6, 9, 12, 15]])
    keep = torch.ones_like(anchors, dtype=torch.bool)
    input_ids = (torch.arange(20).repeat(2, 1) % 7) + 1
    hidden = torch.full((2, 6, 4, 8), -10.0, dtype=torch.double)
    hidden[..., 0] = 4
    plans = [[3, 0, 2, 1, 0, 3], [1, 2, 0, 3, 1, 0]]
    for row, plan in enumerate(plans):
        for block, accepted in enumerate(plan):
            for step in range(accepted):
                slot = step if shifted or family == "dspark" else step + 1
                target = input_ids[row, anchors[row, block] + step + 1]
                hidden[row, block, slot, target] = 3 if family == "dflash2" else 10
    if family == "dflash2":
        selector = draft.candidate_selector
        with torch.no_grad():
            selector.predecessor_codebook.fill_(1)
            selector.successor_codebook.zero_()
            selector.successor_codebook[0].fill_(-1)
            selector.hidden_projection.weight.zero_()
            selector.hidden_projection.weight[:, 0] = 0.5
    hidden.requires_grad_()
    model._forward_draft_blocks = MethodType(
        lambda self, **kwargs: (anchors, keep, hidden.flatten(1, 2)), model
    )
    tensors = dict(
        input_ids=input_ids,
        hidden_states=torch.zeros(2, 20, 8),
        loss_mask=torch.ones(2, 20),
    )
    if family == "dspark":
        tensors["target_last_hidden_states"] = tensors["hidden_states"]
    prefixes = {
        "dflash": ("dflash/hard_label",),
        "dflash2": ("dflash/hard_label", "dflash2/selector"),
        "domino": ("domino/final", "domino/base"),
        "dspark": ("dspark/hard_label",),
    }[family]
    return SimpleNamespace(
        model=model,
        strategy=strategy_class(model),
        hidden=hidden,
        anchors=anchors,
        keep=keep,
        family=family,
        batch=TrainBatch(["a", "b"], family, tensors),
        keys=[f"{prefix}/walk_accepted_length" for prefix in prefixes],
    )


def _check_walk_values_chunking_gradients_and_gating(case):
    reference = None
    metric_reference = None
    for detailed, chunk in [(True, size) for size in (0, 1, 2, 4, 5)] + [(False, 2)]:
        case.model.objective_chunk_blocks = chunk
        case.model.zero_grad(set_to_none=True)
        case.hidden.grad = None
        with patch.object(
            family_model,
            "compute_walk_accepted_length_terms",
            wraps=family_model.compute_walk_accepted_length_terms,
        ) as walk:
            output = case.strategy.forward_loss(
                case.batch, StepContext(collect_detailed_metrics=detailed)
            )
            output.loss.backward()
            assert walk.call_count == (len(case.keys) if detailed else 0)
        gradients = {
            name: p.grad.detach().clone()
            for name, p in case.model.named_parameters()
            if p.grad is not None
        }
        current = (output.loss.detach(), case.hidden.grad, gradients)
        if reference is None:
            reference = current
        else:
            torch.testing.assert_close(current, reference)
        for key in case.keys:
            if detailed:
                expected = (
                    (12.0, 12.0)
                    if case.family == "dflash2" and key == case.keys[0]
                    else (25.0, 10.0)
                )
                if key == "domino/final/walk_accepted_length":
                    # The nonzero correction head changes the planned base prefixes.
                    expected = (20.0, 12.0)
                assert (
                    tuple(value.item() for value in output.ratio_metrics[key])
                    == expected
                )
            else:
                assert key not in output.ratio_metrics
        assert "ratio_metrics" not in output.metrics
        if case.family in ("dflash", "dflash2"):
            if detailed:
                # Walk metrics must coexist with main's prefix diagnostics and
                # additive counts, independently of the metric chunk size.
                assert "dflash/hard_label/unary_greedy_prefix_acceptance" in (
                    output.ratio_metrics
                )
                assert output.sum_metrics["dflash/hard_label/block_count"] == 12
                if case.family == "dflash2":
                    assert "dflash2/selector/greedy_prefix_acceptance" in (
                        output.ratio_metrics
                    )
                metric_values = (output.ratio_metrics, output.sum_metrics)
                if metric_reference is None:
                    metric_reference = metric_values
                else:
                    torch.testing.assert_close(metric_values, metric_reference)
            else:
                assert output.sum_metrics == {}


@torch.no_grad()
def _check_causal_heads_match_serving_prefixes(case, dtype):
    model, draft = case.model, case.model.draft_model
    model.to(dtype=dtype)
    hidden = torch.randn(2, 6, 4, 8, dtype=dtype)
    anchor_ids = torch.randint(1, 8, (2, 6))
    block_ids = torch.zeros(12, 4, dtype=torch.long)
    block_ids[:, 0] = anchor_ids.flatten()
    if case.family == "domino":
        target = SimpleNamespace(
            lm_head=model.lm_head,
            model=SimpleNamespace(embed_tokens=model.embed_tokens),
        )
        generated = draft._sample_draft_tokens(
            target, hidden.flatten(0, 1), block_ids
        ).reshape(2, 6, 3)
    elif draft.markov_head is None:
        generated = hidden.argmax(-1)
    else:
        generated, _ = draft.markov_head.sample_block_tokens(
            hidden.flatten(0, 1),
            first_prev_token_ids=anchor_ids.flatten(),
            hidden_states=hidden.flatten(0, 1),
        )
        generated = generated.reshape(2, 6, 4)
    for corrupt_at in range(generated.shape[-1] + 1):
        labels = generated.clone()
        if corrupt_at < labels.shape[-1]:
            labels[..., corrupt_at] = (labels[..., corrupt_at] + 1) % 8
        if case.family == "domino":
            clean = torch.cat((anchor_ids.unsqueeze(-1), labels), dim=-1)
            if model.shift_label:
                targets = torch.cat((labels, torch.zeros_like(labels[..., :1])), dim=-1)
                prev = clean
            else:
                targets = prev = clean
            weights = torch.ones_like(targets, dtype=torch.double)
            terms = model._domino_objective_chunk_terms(
                hidden,
                prev,
                targets,
                weights,
                weights,
                torch.arange(6).unsqueeze(0),
                total_blocks=6,
            )
            accepted = terms[-2]
        else:
            prev = torch.cat((anchor_ids.unsqueeze(-1), labels[..., :-1]), dim=-1)
            terms = model._dspark_objective_chunk_terms(
                hidden,
                prev,
                labels,
                torch.ones_like(labels, dtype=torch.double),
                torch.ones_like(labels, dtype=torch.bool),
                None,
                torch.arange(6).unsqueeze(0),
                total_blocks=6,
            )
            accepted = terms[-1]
        torch.testing.assert_close(accepted, torch.full((2, 6), float(corrupt_at)))


class DFlashWalkTest(unittest.TestCase):
    def test_walk_values_chunking_gradients_and_gating(self):
        for kind in MODEL_KINDS:
            with self.subTest(model=kind):
                _check_walk_values_chunking_gradients_and_gating(_make_case(kind))

    def test_sparse_padded_and_empty_walks(self):
        cases = (
            ([0, 1, 2, 3, 4, 5], [2, 0, 5, 1, 0, 3], [1] * 6, (9, 3)),
            ([0, 1, 2, 10], [3, 0, 0, 0], [1] * 4, (5, 2)),
            ([1, 3, 8, 0], [0, 4, 0, 8], [1, 0, 1, 0], (2, 2)),
            ([0, 0], [9, 9], [0, 0], (0, 0)),
        )
        for positions, counts, valid, expected in cases:
            with self.subTest(positions=positions, counts=counts, valid=valid):
                result = family_model.compute_walk_accepted_length_terms(
                    torch.tensor([counts], dtype=torch.float32),
                    torch.tensor([positions]),
                    torch.tensor([valid], dtype=torch.bool),
                )
                self.assertEqual(tuple(value.item() for value in result), expected)

    def test_prefix_stops_at_masked_holes_and_scatters_global_indices(self):
        targets = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
        valid = torch.tensor(
            [[[1, 0, 1], [0, 1, 1]], [[1, 1, 1], [0, 0, 0]]], dtype=torch.bool
        )
        result = family_model._scatter_accepted_prefix(
            targets, targets, valid, torch.tensor([[1, 3]]), 5
        )
        self.assertEqual(result.tolist(), [[0, 1, 0, 0, 0], [0, 3, 0, 0, 0]])

    def test_causal_heads_match_serving_prefixes(self):
        for kind in MODEL_KINDS:
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(model=kind, dtype=dtype):
                    case = _make_case(kind)
                    if case.family not in ("domino", "dspark"):
                        self.skipTest("Causal-head comparison applies to Domino/DSpark")
                    _check_causal_heads_match_serving_prefixes(case, dtype)


if __name__ == "__main__":
    unittest.main()
