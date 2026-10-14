import unittest
from itertools import product
from unittest import mock

import torch
from torch import nn

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.algorithms.model_providers import build_domino_model, build_dspark_model
from specforge.config import Config
from tests.test_modeling.test_dspark_causal import build_model as dspark_model
from tests.test_utils.test_domino_optimizations import model as domino_model


def config_payload(strategy, **options):
    return {
        "model": {"target_model_path": "unused"},
        "data": {"hidden_states_path": "unused"},
        "training": {"strategy": strategy, **options},
    }


class VocabularyProjectionTest(unittest.TestCase):
    def test_shared_options_reach_both_models_independently(self):
        for strategy, builder, reference in (
            ("domino", build_domino_model, domino_model(device="cpu")),
            ("dspark", build_dspark_model, dspark_model()),
        ):
            common = {
                "draft_model": reference.draft_model,
                "target_lm_head": reference.lm_head,
                "target_embed_tokens": reference.embed_tokens,
                "mask_token_id": reference.mask_token_id,
                "block_size": reference.block_size,
            }
            defaults = Config.model_validate(config_payload(strategy))
            self.assertFalse(defaults.training.cache_vocab_projection)
            self.assertFalse(defaults.training.flatten_vocab_projection)
            for cache, flatten in product((False, True), repeat=2):
                with self.subTest(strategy=strategy, cache=cache, flatten=flatten):
                    config = Config.model_validate(
                        config_payload(
                            strategy,
                            cache_vocab_projection=cache,
                            flatten_vocab_projection=flatten,
                        )
                    )
                    with mock.patch(
                        "specforge.algorithms.model_providers._build_dflash_family_model",
                        side_effect=lambda config, draft, tokenizer, factory: factory(
                            common
                        ),
                    ):
                        model = builder(config, reference.draft_model, None, None, None)
                    self.assertIs(model.cache_vocab_projection, cache)
                    self.assertIs(model.flatten_vocab_projection, flatten)

    def test_shared_options_reject_unsupported_strategies(self):
        for strategy, option in product(
            ("dflash", "eagle3"),
            ("cache_vocab_projection", "flatten_vocab_projection"),
        ):
            with self.subTest(strategy=strategy, option=option):
                with self.assertRaisesRegex(ValueError, "to be domino or dspark"):
                    Config.model_validate(config_payload(strategy, **{option: True}))

    def test_flatten_preserves_noncontiguous_inputs_and_gradients(self):
        for trainable, bias in product((False, True), repeat=2):
            with self.subTest(trainable=trainable, bias=bias):
                torch.manual_seed(42)
                head = nn.Linear(4, 7, bias=bias).double().requires_grad_(trainable)
                owner = type("ProjectionOwner", (), {"lm_head": head})()
                values = torch.randn(2, 6, 4, dtype=torch.double)[
                    :, ::2
                ].requires_grad_()
                self.assertFalse(values.is_contiguous())
                reference = values.detach().clone().requires_grad_()
                expected = head(reference)
                actual = OnlineDFlashModel._project_vocab_logits(
                    owner, values, flatten=True
                )
                parameters = tuple(head.parameters()) if trainable else ()
                expected_grad = torch.autograd.grad(
                    expected.square().sum(), (reference, *parameters)
                )
                actual_grad = torch.autograd.grad(
                    actual.square().sum(), (values, *parameters)
                )
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(actual_grad, expected_grad)

    def test_custom_head_keeps_input_layout(self):
        class SequenceHead(nn.Module):
            def forward(self, hidden):
                # This operation depends on the sequence axis.
                return hidden - hidden.mean(dim=1, keepdim=True)

        owner = type("ProjectionOwner", (), {"lm_head": SequenceHead()})()
        hidden = torch.randn(2, 3, 4)
        actual = OnlineDFlashModel._project_vocab_logits(owner, hidden, flatten=True)
        torch.testing.assert_close(actual, owner.lm_head(hidden))

    def test_both_objectives_use_shared_flattening(self):
        for strategy, flatten in product(("domino", "dspark"), (False, True)):
            with self.subTest(strategy=strategy, flatten=flatten):
                model = (
                    domino_model(device="cpu", dtype=torch.float32, chunk=0)
                    if strategy == "domino"
                    else dspark_model()
                )
                model.objective_chunk_blocks = 0
                model.flatten_vocab_projection = flatten
                tokens = torch.randint(3, 60, (2, 32))
                hidden = torch.randn(2, 32, model.draft_model.config.hidden_size * 2)
                observed = []
                handle = model.lm_head.register_forward_pre_hook(
                    lambda module, args: observed.append(
                        (args[0].ndim, torch.is_grad_enabled())
                    )
                )
                kwargs = {}
                if strategy == "dspark":
                    kwargs["target_last_hidden_states"] = torch.randn(2, 32, 128)
                try:
                    model(tokens, hidden, torch.ones_like(tokens), **kwargs)
                finally:
                    handle.remove()
                self.assertTrue(observed)
                self.assertTrue(
                    all(rank == (2 if flatten else 3) for rank, _ in observed)
                )
                if strategy == "dspark":
                    self.assertEqual({grad for _, grad in observed}, {False, True})


if __name__ == "__main__":
    unittest.main()
