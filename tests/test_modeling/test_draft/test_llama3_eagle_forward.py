"""CPU regression tests for the public EAGLE3 draft forward path."""

import unittest
from unittest import mock

import torch
from transformers import LlamaConfig

from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3


class LlamaEagle3ForwardTest(unittest.TestCase):
    def _model(self, norm_output):
        config = LlamaConfig(
            vocab_size=16,
            draft_vocab_size=8,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=16,
            pad_token_id=0,
            norm_output=norm_output,
        )
        return LlamaForCausalLMEagle3(config, attention_backend="sdpa").eval()

    @torch.no_grad()
    def test_forward_runs_with_both_normalization_and_cache_modes(self):
        generator = torch.Generator().manual_seed(0)
        hidden_states = torch.randn(2, 3, 24, generator=generator)
        inputs_embeds = torch.randn(2, 3, 8, generator=generator)

        for norm_output in (False, True):
            model = self._model(norm_output)
            for ttt_length in (1, 2):
                with self.subTest(norm_output=norm_output, ttt_length=ttt_length):
                    output = model(
                        hidden_states=hidden_states,
                        inputs_embeds=inputs_embeds,
                        ttt_length=ttt_length,
                    )

                    self.assertEqual(output.shape, (2, 3, 8))
                    self.assertTrue(torch.isfinite(output).all())
                    logits = model.compute_logits(output)
                    self.assertEqual(logits.shape, (2, 3, 8))
                    self.assertTrue(torch.isfinite(logits).all())

    @torch.no_grad()
    def test_forward_and_logits_apply_output_norm_exactly_once(self):
        raw_hidden = torch.arange(1, 25, dtype=torch.float32).reshape(1, 3, 8)

        for norm_output in (False, True):
            with self.subTest(norm_output=norm_output):
                model = self._model(norm_output)
                # A non-unit scale also makes accidental double normalization visible.
                model.norm.weight.fill_(2.0)
                normalized = model.norm(raw_hidden)
                expected_logits = model.lm_head(normalized)

                with (
                    mock.patch.object(
                        model.midlayer, "forward", return_value=raw_hidden
                    ),
                    mock.patch.object(
                        model.norm, "forward", wraps=model.norm.forward
                    ) as normalize,
                ):
                    output = model(
                        hidden_states=torch.ones(1, 3, 24),
                        inputs_embeds=torch.ones(1, 3, 8),
                    )
                    self.assertEqual(normalize.call_count, int(norm_output))
                    logits = model.compute_logits(output)
                    self.assertEqual(normalize.call_count, 1)

                torch.testing.assert_close(
                    output, normalized if norm_output else raw_hidden
                )
                torch.testing.assert_close(logits, expected_logits)


if __name__ == "__main__":
    unittest.main()
