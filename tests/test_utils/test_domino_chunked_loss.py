import copy
import importlib.util
import unittest
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "specforge.core.loss", REPO / "specforge" / "core" / "loss.py"
)
_loss_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_loss_module)

compute_accept_len = _loss_module.compute_accept_len
compute_domino_chunked_metrics = _loss_module.compute_domino_chunked_metrics
compute_domino_chunked_weighted_losses = (
    _loss_module.compute_domino_chunked_weighted_losses
)


def _full_loss(
    output_hidden,
    prefix_states,
    target_ids,
    weight_mask,
    lambda_base,
    lm_head,
    embed_proj,
    suffix_start,
):
    bsz, n, block_size = target_ids.shape
    base_logits = lm_head(output_hidden).view(bsz, n, block_size, -1)
    hidden = output_hidden.view(bsz, n, block_size, -1)
    correction = embed_proj(
        torch.cat([hidden[:, :, suffix_start:, :], prefix_states], dim=-1)
    )
    final_logits = torch.cat(
        [
            base_logits[:, :, :suffix_start, :],
            base_logits[:, :, suffix_start:, :] + correction,
        ],
        dim=2,
    )

    targets = target_ids.reshape(-1)
    weights = weight_mask.reshape(-1)
    denominator = weights.sum() + 1e-6
    final_loss = (
        F.cross_entropy(
            final_logits.reshape(-1, final_logits.shape[-1]),
            targets,
            reduction="none",
        )
        * weights
    ).sum() / denominator
    base_loss = (
        F.cross_entropy(
            base_logits.reshape(-1, base_logits.shape[-1]),
            targets,
            reduction="none",
        )
        * weights
    ).sum() / denominator
    loss = (1.0 - lambda_base) * final_loss + lambda_base * base_loss
    return loss, final_loss, base_loss, final_logits, base_logits


class TestDominoChunkedLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.bsz = 2
        self.num_blocks = 3
        self.block_size = 4
        self.suffix_start = 2
        self.hidden_size = 7
        self.prefix_size = 5
        self.vocab_size = 13

        self.output_hidden = torch.randn(
            self.bsz,
            self.num_blocks * self.block_size,
            self.hidden_size,
            dtype=torch.double,
        )
        self.prefix_states = torch.randn(
            self.bsz,
            self.num_blocks,
            self.block_size - self.suffix_start,
            self.prefix_size,
            dtype=torch.double,
        )
        self.target_ids = torch.randint(
            self.vocab_size,
            (self.bsz, self.num_blocks, self.block_size),
        )
        self.weight_mask = torch.rand(
            self.bsz, self.num_blocks, self.block_size, dtype=torch.double
        )
        self.weight_mask[0, 0, 0] = 0.0
        self.weight_mask[1, 2, 3] = 0.0

        self.lm_head = nn.Linear(
            self.hidden_size, self.vocab_size, bias=False, dtype=torch.double
        )
        self.lm_head.requires_grad_(False)
        self.embed_proj = nn.Sequential(
            nn.Linear(
                self.hidden_size + self.prefix_size,
                9,
                bias=False,
                dtype=torch.double,
            ),
            nn.SiLU(),
            nn.Linear(9, self.vocab_size, bias=False, dtype=torch.double),
        )

    def _run_full(self, lambda_base):
        hidden = self.output_hidden.clone().requires_grad_(True)
        prefix = self.prefix_states.clone().requires_grad_(True)
        embed_proj = copy.deepcopy(self.embed_proj)
        outputs = _full_loss(
            hidden,
            prefix,
            self.target_ids,
            self.weight_mask,
            lambda_base,
            self.lm_head,
            embed_proj,
            self.suffix_start,
        )
        outputs[0].backward()
        grads = [hidden.grad, prefix.grad]
        grads.extend(parameter.grad for parameter in embed_proj.parameters())
        return outputs, grads

    def _run_chunked(self, lambda_base, chunk_size):
        hidden = self.output_hidden.clone().requires_grad_(True)
        prefix = self.prefix_states.clone().requires_grad_(True)
        embed_proj = copy.deepcopy(self.embed_proj)
        outputs = compute_domino_chunked_weighted_losses(
            output_hidden=hidden,
            prefix_states=prefix,
            target_ids=self.target_ids,
            weight_mask=self.weight_mask,
            lambda_base=lambda_base,
            logit_chunk_size=chunk_size,
            lm_head=self.lm_head,
            embed_proj=embed_proj,
            block_size=self.block_size,
            suffix_start=self.suffix_start,
        )
        outputs[0].backward()
        grads = [hidden.grad, prefix.grad]
        grads.extend(parameter.grad for parameter in embed_proj.parameters())
        return outputs, grads, hidden, prefix, embed_proj

    def test_loss_and_gradients_match_full_logits(self):
        total_tokens = self.target_ids.numel()
        for lambda_base in (0.0, 0.4, 1.0):
            full_outputs, full_grads = self._run_full(lambda_base)
            for chunk_size in (1, 3, 5, total_tokens + 7):
                with self.subTest(lambda_base=lambda_base, chunk_size=chunk_size):
                    chunked_outputs, chunked_grads, _, _, _ = self._run_chunked(
                        lambda_base, chunk_size
                    )
                    for actual, expected in zip(chunked_outputs, full_outputs[:3]):
                        torch.testing.assert_close(
                            actual, expected, rtol=1e-10, atol=1e-10
                        )
                    for actual, expected in zip(chunked_grads, full_grads):
                        torch.testing.assert_close(
                            actual, expected, rtol=1e-10, atol=1e-10
                        )

    def test_metrics_match_full_logits(self):
        lambda_base = 0.4
        full_outputs, _ = self._run_full(lambda_base)
        chunked_outputs, _, hidden, prefix, embed_proj = self._run_chunked(
            lambda_base, chunk_size=5
        )
        eval_weight_mask = (self.weight_mask > 0).to(self.weight_mask.dtype)
        accuracy, metrics = compute_domino_chunked_metrics(
            output_hidden=hidden,
            prefix_states=prefix,
            target_ids=self.target_ids,
            eval_weight_mask=eval_weight_mask,
            final_loss=chunked_outputs[1],
            base_loss=chunked_outputs[2],
            lambda_base=lambda_base,
            logit_chunk_size=5,
            lm_head=self.lm_head,
            embed_proj=embed_proj,
            block_size=self.block_size,
            suffix_start=self.suffix_start,
        )

        _, final_loss, base_loss, final_logits, base_logits = full_outputs
        valid = self.weight_mask > 0
        targets = self.target_ids
        final_predictions = final_logits.argmax(dim=-1)
        base_predictions = base_logits.argmax(dim=-1)
        expected_accuracy = ((final_predictions == targets) & valid).sum() / valid.sum()
        expected_base_accuracy = (
            (base_predictions == targets) & valid
        ).sum() / valid.sum()
        valid_blocks = valid.any(dim=2)
        expected_accept_len = (
            (compute_accept_len(final_predictions, targets, valid) + 1.0) * valid_blocks
        ).sum() / valid_blocks.sum()
        expected_base_accept_len = (
            (compute_accept_len(base_predictions, targets, valid) + 1.0) * valid_blocks
        ).sum() / valid_blocks.sum()

        torch.testing.assert_close(expected_accuracy.to(accuracy), accuracy)
        torch.testing.assert_close(metrics["final_loss"], final_loss)
        torch.testing.assert_close(metrics["base_loss"], base_loss)
        torch.testing.assert_close(
            expected_base_accuracy.to(metrics["base_accuracy"]),
            metrics["base_accuracy"],
        )
        torch.testing.assert_close(metrics["accept_len"], expected_accept_len.float())
        torch.testing.assert_close(
            metrics["base_accept_len"], expected_base_accept_len.float()
        )

    def test_rejects_non_positive_chunk_size(self):
        with self.assertRaisesRegex(ValueError, "must be positive"):
            compute_domino_chunked_weighted_losses(
                output_hidden=self.output_hidden,
                prefix_states=self.prefix_states,
                target_ids=self.target_ids,
                weight_mask=self.weight_mask,
                lambda_base=0.5,
                logit_chunk_size=0,
                lm_head=self.lm_head,
                embed_proj=self.embed_proj,
                block_size=self.block_size,
                suffix_start=self.suffix_start,
            )


if __name__ == "__main__":
    unittest.main()
