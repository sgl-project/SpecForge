"""Supervised-row objective parity and projection-work regressions."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from specforge.core.mtp import OnlineMTPModel


@pytest.mark.parametrize("offset", [0, 1, 2])
@pytest.mark.parametrize("chunk_size", [0, 3, 4096])
@pytest.mark.parametrize("mask_kind", ["full", "sparse", "weighted", "empty"])
@pytest.mark.parametrize("frozen_head", [False, True])
def test_filtered_objective_matches_dense(offset, chunk_size, mask_kind, frozen_head):
    torch.manual_seed(17)
    batch, length, hidden_size, vocab_size = 2, 9, 7, 11
    head = nn.Linear(hidden_size, vocab_size, bias=False)
    head.requires_grad_(not frozen_head)
    reference_head = copy.deepcopy(head)
    draft = nn.Module()
    draft.mtp = nn.Module()
    draft.mtp.lm_head = head
    draft.config = SimpleNamespace(pad_token_id=0)
    model = OnlineMTPModel(draft, objective_chunk_size=chunk_size, num_steps=3)
    ids = torch.randint(vocab_size, (batch, length))
    mask = torch.ones(batch, length)
    if mask_kind == "sparse":
        mask[:, :4] = 0
        mask[0, 6:] = 0
    elif mask_kind == "weighted":
        mask = torch.randint(0, 5, (batch, length)).float() / 4
    elif mask_kind == "empty":
        mask.zero_()
    hidden = torch.randn(batch, length - 1, hidden_size, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()

    labels = F.pad(ids[:, 2 + offset :], (0, 1 + offset), value=-100)
    weights = F.pad(mask[:, 2 + offset :], (0, 1 + offset))
    logits = reference_head(reference_hidden)
    losses = F.cross_entropy(
        logits.reshape(-1, vocab_size), labels.reshape(-1), reduction="none"
    )
    expected_loss = (losses * weights.reshape(-1)).sum() / weights.sum().clamp_min(1)
    expected_corrects = (logits.argmax(-1) == labels).float() * weights
    actual_loss, corrects, denoms = model._chunked_objective(hidden, ids, mask, offset)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(corrects, expected_corrects, rtol=0, atol=0)
    torch.testing.assert_close(denoms, weights, rtol=0, atol=0)

    actual_loss.backward()
    expected_loss.backward()
    assert hidden.grad is not None
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=1e-5, atol=1e-6)
    if not frozen_head:
        assert head.weight.grad is not None
        torch.testing.assert_close(
            head.weight.grad, reference_head.weight.grad, rtol=1e-5, atol=1e-6
        )

    # Count projection work in inference mode, excluding backward recomputation.
    projected_rows = []
    handle = head.register_forward_pre_hook(
        lambda _, args: projected_rows.append(args[0].shape[0])
    )
    try:
        with torch.no_grad():
            model._chunked_objective(hidden, ids, mask, offset)
    finally:
        handle.remove()
    assert sum(projected_rows) == max(int(weights.count_nonzero()), 1)
