import types
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from tests.test_utils.test_dflash_losses import OnlineDFlashModel, _dflash_module


class _RecordingDraft(nn.Module):
    draft_kernel_backend = "torch"

    def forward(
        self, position_ids, noise_embedding, target_hidden, attention_mask, **kwargs
    ):
        del position_ids, target_hidden, attention_mask
        self.forward_kwargs = kwargs
        return torch.zeros_like(noise_embedding)


class TestDFlashFlexKernelOptions(unittest.TestCase):
    def _model(self, *, attention_backend, options):
        draft = _RecordingDraft()
        model = OnlineDFlashModel(
            draft_model=draft,
            target_lm_head=nn.Linear(4, 11, bias=False),
            target_embed_tokens=nn.Embedding(11, 4),
            mask_token_id=0,
            block_size=2,
            attention_backend=attention_backend,
            num_anchors=1,
            flex_kernel_options=options,
        )
        model._sample_anchor_positions = types.MethodType(
            lambda self, seq_len, loss_mask, device: (
                torch.tensor([[1]], device=device),
                torch.tensor([[True]], device=device),
            ),
            model,
        )
        model._create_noise_embed = types.MethodType(
            lambda self, input_ids, anchor_positions, block_keep_mask: torch.zeros(
                1, 2, 4, device=input_ids.device
            ),
            model,
        )
        return model, draft

    def test_forwards_options_to_flex_attention(self):
        options = {"num_stages": 2}
        model, draft = self._model(
            attention_backend="flex_attention", options=options
        )
        with patch.object(
            _dflash_module, "create_dflash_block_mask", return_value=object()
        ):
            model._forward_draft_blocks(
                input_ids=torch.tensor([[1, 2, 3, 4]]),
                hidden_states=torch.zeros(1, 4, 4),
                loss_mask=torch.ones(1, 4),
            )

        self.assertEqual(draft.forward_kwargs, {"kernel_options": options})
        self.assertIsNot(model.flex_kernel_options, options)

    def test_rejects_options_for_non_flex_backend(self):
        with self.assertRaisesRegex(ValueError, "attention_backend='flex_attention'"):
            self._model(attention_backend="sdpa", options={"num_stages": 2})


if __name__ == "__main__":
    unittest.main()
