"""DFlash packing metadata remains host-side across the wrapped model boundary."""

import unittest

import torch
from torch import nn

from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import DFlashTrainStrategy, StepContext


class _RecordingModel(nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones((), device=device))
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        return (
            self.weight * kwargs["hidden_states"].sum(),
            self.weight.new_zeros(()),
            {},
        )


def _batch(loss_mask, lengths):
    return TrainBatch(
        sample_ids=[str(i) for i in range(len(lengths))],
        strategy="dflash",
        tensors={
            "input_ids": torch.zeros_like(loss_mask, dtype=torch.long),
            "loss_mask": loss_mask,
            "hidden_states": torch.ones(1, sum(lengths), 4, device=loss_mask.device),
            "sequence_lengths": torch.tensor(lengths, dtype=torch.long),
        },
        metadata={},
    )


class PackingStrategyMetadataTest(unittest.TestCase):
    def test_host_counts_respect_document_boundaries_and_stay_python_values(self):
        model = _RecordingModel()
        mask = torch.tensor([[1, 1, 1, 1, 0, 1, 1, 1, 0]])
        DFlashTrainStrategy(model).forward_loss(_batch(mask, (3, 4, 2)), StepContext())
        self.assertEqual(model.kwargs["sequence_lengths"], (3, 4, 2))
        self.assertEqual(model.kwargs["valid_anchor_counts"], (2, 1, 0))
        self.assertEqual(model.kwargs["max_valid_anchors"], 2)
        self.assertIsInstance(model.kwargs["max_valid_anchors"], int)
        self.assertTrue(
            all(isinstance(n, int) for n in model.kwargs["valid_anchor_counts"])
        )

    def test_zero_masks_and_cross_document_pair_have_no_anchors(self):
        for mask in (torch.zeros(1, 4), torch.tensor([[0, 1, 1, 0]])):
            with self.subTest(mask=mask.tolist()):
                model = _RecordingModel()
                DFlashTrainStrategy(model).forward_loss(_batch(mask, (2, 2)))
                self.assertEqual(model.kwargs["valid_anchor_counts"], (0, 0))
                self.assertEqual(model.kwargs["max_valid_anchors"], 0)

    def test_model_rejects_no_anchor_batch_before_attention(self):
        from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel

        model = OnlineDFlashModel(
            nn.Linear(4, 4),
            nn.Linear(4, 8, bias=False),
            nn.Embedding(8, 4),
            mask_token_id=0,
            block_size=2,
            num_anchors=2,
            attention_backend="flex_attention",
        )
        for mask in (torch.zeros(1, 4), torch.tensor([[0, 1, 1, 0]])):
            with (
                self.subTest(mask=mask.tolist()),
                self.assertRaisesRegex(ValueError, "two consecutive supervised"),
            ):
                DFlashTrainStrategy(model).forward_loss(_batch(mask, (2, 2)))

    @unittest.skipUnless(
        torch.cuda.is_available(), "GPU loss-mask fallback requires CUDA"
    )
    def test_gpu_loss_mask_uses_model_fallback_without_host_counts(self):
        model = _RecordingModel("cuda")
        batch = _batch(torch.tensor([[1, 1, 0, 1, 1]], device="cuda"), (3, 2))
        DFlashTrainStrategy(model).forward_loss(batch)
        self.assertEqual(model.kwargs["sequence_lengths"], (3, 2))
        self.assertIsNone(model.kwargs["max_valid_anchors"])
        self.assertNotIn("valid_anchor_counts", model.kwargs)


if __name__ == "__main__":
    unittest.main()
