# coding=utf-8
"""Focused CPU regressions for Domino's auxiliary logits head."""

import unittest
from types import MethodType
from unittest.mock import patch

import torch
from torch import nn

from specforge.modeling.draft.domino import DominoDraftModel


def _bare_domino(
    *,
    hidden_size: int = 4,
    gru_hidden_size: int = 3,
    embedding_size: int = 2,
    vocab_size: int = 7,
    block_size: int = 4,
) -> DominoDraftModel:
    model = DominoDraftModel.__new__(DominoDraftModel)
    nn.Module.__init__(model)
    model.block_size = block_size
    model.shift_label = False
    model.pure_draft_prefix_len = 0
    model.prefix_gru = nn.GRU(
        input_size=hidden_size,
        hidden_size=gru_hidden_size,
        num_layers=1,
        batch_first=True,
        bias=False,
    )
    model.embed_proj = nn.Sequential(
        nn.Linear(hidden_size + gru_hidden_size, embedding_size, bias=False),
        nn.SiLU(),
        nn.Linear(embedding_size, vocab_size, bias=False),
    )
    return model


class _TargetBody(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)


class _TinyTarget(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.model = _TargetBody(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)


class TestDominoDraftModel(unittest.TestCase):
    def test_training_loss_updates_gru_and_projection(self):
        from specforge.algorithms.common.dflash_family_model import OnlineDominoModel

        torch.manual_seed(11)
        hidden_size, vocab_size, block_size = 4, 7, 4
        draft = _bare_domino(
            hidden_size=hidden_size,
            gru_hidden_size=3,
            embedding_size=2,
            vocab_size=vocab_size,
            block_size=block_size,
        )
        model = OnlineDominoModel(
            draft_model=draft,
            target_lm_head=nn.Linear(hidden_size, vocab_size, bias=False),
            target_embed_tokens=nn.Embedding(vocab_size, hidden_size),
            mask_token_id=0,
            block_size=block_size,
            attention_backend="sdpa",
            num_anchors=1,
            shift_label=False,
        )
        fixed_hidden = torch.randn(1, block_size, hidden_size)

        def fixed_draft_blocks(self, input_ids, hidden_states, loss_mask):
            del hidden_states, loss_mask
            return (
                torch.zeros(1, 1, dtype=torch.long, device=input_ids.device),
                torch.ones(1, 1, dtype=torch.bool, device=input_ids.device),
                fixed_hidden.to(input_ids.device),
            )

        model._forward_draft_blocks = MethodType(fixed_draft_blocks, model)
        loss, _accuracy, _metrics = model(
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            hidden_states=torch.zeros(1, block_size, hidden_size),
            loss_mask=torch.ones(1, block_size),
            lambda_base=0.0,
        )
        loss.backward()

        for module in (draft.prefix_gru, draft.embed_proj):
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None for gradient in gradients))
            self.assertGreater(
                sum(gradient.abs().sum().item() for gradient in gradients),
                0.0,
            )

    def test_npu_bf16_gru_gradients_reach_registered_weights(self):
        torch.manual_seed(7)
        model = _bare_domino().to(dtype=torch.bfloat16)
        inputs = torch.randn(2, 5, 4, dtype=torch.bfloat16)

        with patch(
            "specforge.modeling.draft.domino.get_device_type", return_value="npu"
        ):
            output = model._run_gru(inputs)
            output.float().square().mean().backward()

        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertFalse(hasattr(model, "_gru_fp16"))
        for parameter in model.prefix_gru.parameters():
            self.assertEqual(parameter.dtype, torch.bfloat16)
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0.0)

    def test_generation_is_sensitive_to_gru_and_projection_weights(self):
        hidden_size, vocab_size, block_size = 2, 5, 4
        model = _bare_domino(
            hidden_size=hidden_size,
            gru_hidden_size=1,
            embedding_size=1,
            vocab_size=vocab_size,
            block_size=block_size,
        )
        target = _TinyTarget(vocab_size, hidden_size)

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            for parameter in target.parameters():
                parameter.zero_()
            # The anchor and the correction-selected token feed a positive GRU
            # candidate. The projection reads only that GRU state and boosts id 3.
            target.model.embed_tokens.weight[1, 0] = 1.0
            target.model.embed_tokens.weight[3, 0] = 1.0
            model.prefix_gru.weight_ih_l0[2, 0] = 2.0
            model.embed_proj[0].weight[0, hidden_size] = 4.0
            model.embed_proj[2].weight[3, 0] = 4.0

        draft_hidden = torch.zeros(1, block_size, hidden_size)
        block_ids = torch.tensor([[1, 4, 4, 4]])
        corrected = model._sample_draft_tokens(target, draft_hidden, block_ids)
        torch.testing.assert_close(corrected, torch.tensor([[3, 3, 3]]))

        gru_weight = model.prefix_gru.weight_ih_l0.detach().clone()
        with torch.no_grad():
            model.prefix_gru.weight_ih_l0.zero_()
        no_gru_correction = model._sample_draft_tokens(target, draft_hidden, block_ids)
        torch.testing.assert_close(no_gru_correction, torch.tensor([[0, 0, 0]]))

        with torch.no_grad():
            model.prefix_gru.weight_ih_l0.copy_(gru_weight)
            model.embed_proj[2].weight.zero_()
        no_projection = model._sample_draft_tokens(target, draft_hidden, block_ids)
        torch.testing.assert_close(no_projection, torch.tensor([[0, 0, 0]]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
