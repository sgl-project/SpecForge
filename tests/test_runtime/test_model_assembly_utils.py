"""Import-light tests for shared training model assembly helpers."""

import types
import unittest
from unittest import mock

from specforge.config import Config
from specforge.training.model_utils import resolve_mask_token_id


class _Tokenizer:
    def __init__(
        self,
        *,
        size=32,
        mask_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        unk_token_id=None,
    ):
        self.size = size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id

    def __len__(self):
        return self.size


def _draft(mask_token_id=None):
    config = types.SimpleNamespace(dflash_config={"mask_token_id": mask_token_id})
    return types.SimpleNamespace(config=config)


class TestResolveMaskTokenId(unittest.TestCase):
    def test_priority_is_explicit_draft_tokenizer_unused_then_fallback(self):
        tokenizer = _Tokenizer(size=30, mask_token_id=7, pad_token_id=3)
        self.assertEqual(
            resolve_mask_token_id(
                explicit=12,
                tokenizer=tokenizer,
                draft_model=_draft(16),
                embedding_vocab_size=32,
            ),
            12,
        )
        self.assertEqual(
            resolve_mask_token_id(
                explicit=None,
                tokenizer=tokenizer,
                draft_model=_draft(16),
                embedding_vocab_size=32,
            ),
            16,
        )
        self.assertEqual(
            resolve_mask_token_id(
                explicit=None,
                tokenizer=tokenizer,
                embedding_vocab_size=32,
            ),
            7,
        )
        tokenizer.mask_token_id = None
        self.assertEqual(
            resolve_mask_token_id(
                explicit=None,
                tokenizer=tokenizer,
                embedding_vocab_size=32,
            ),
            30,
        )
        tokenizer.size = 32
        self.assertEqual(
            resolve_mask_token_id(
                explicit=None,
                tokenizer=tokenizer,
                embedding_vocab_size=32,
            ),
            3,
        )

    def test_candidates_must_fit_embedding_table(self):
        tokenizer = _Tokenizer(size=32, pad_token_id=33)
        with self.assertRaisesRegex(ValueError, "outside draft embedding"):
            resolve_mask_token_id(
                explicit=33,
                tokenizer=tokenizer,
                embedding_vocab_size=32,
            )
        with self.assertRaisesRegex(ValueError, "unable to resolve"):
            resolve_mask_token_id(
                explicit=None,
                tokenizer=tokenizer,
                embedding_vocab_size=32,
            )


def _dflash_config(**training_overrides):
    training = {
        "strategy": "dflash",
        "total_steps": 10,
        "draft_kernel_backend": "liger",
        "linear_cross_entropy_backend": "liger",
        "compact_zero_weight_ce_rows": True,
        "flex_kernel_options": {"num_stages": 2},
        "adamw_backend": "fused",
        **training_overrides,
    }
    return Config.model_validate(
        {
            "model": {
                "target_model_path": "target",
                "draft_model_config": "draft.json",
                "mask_token_id": 7,
            },
            "data": {"hidden_states_path": "features"},
            "training": training,
        }
    )


class TestDFlashOptimizationWiring(unittest.TestCase):
    def test_draft_model_receives_configured_kernel_backend(self):
        from specforge.algorithms.model_providers import build_registered_draft

        cfg = _dflash_config()
        draft_config = types.SimpleNamespace(
            architectures=["DFlashDraftModel"],
            _attn_implementation=None,
        )
        draft_model = mock.Mock()
        draft_model.to.return_value = draft_model
        with mock.patch(
            "specforge.modeling.auto.AutoDraftModel.from_config",
            return_value=draft_model,
        ) as factory:
            result = build_registered_draft(cfg, draft_config)

        self.assertIs(result, draft_model)
        self.assertEqual(factory.call_args.kwargs["draft_kernel_backend"], "liger")

    def test_training_model_and_optimizer_receive_optimized_backends(self):
        from specforge.algorithms.model_providers import build_dflash_model
        from specforge.training.assembly import _ConfiguredOptimizerFactory

        cfg = _dflash_config()
        draft_model = mock.Mock(block_size=16, target_layer_ids=[1, 2, 3])
        draft_model.config = types.SimpleNamespace(dflash_config={}, vocab_size=32)
        draft_model.to.return_value = draft_model
        target = types.SimpleNamespace(lm_head=object(), embed_tokens=object())
        built_model = mock.Mock()
        built_model.to.return_value = built_model
        with (
            mock.patch(
                "specforge.modeling.target.target_utils."
                "TargetEmbeddingsAndHead.from_pretrained",
                return_value=target,
            ),
            mock.patch(
                "specforge.algorithms.common.dflash_family_model.OnlineDFlashModel",
                return_value=built_model,
            ) as model_factory,
        ):
            parts = build_dflash_model(cfg, draft_model, None, None, _Tokenizer())

        self.assertIs(parts.model, built_model)
        kwargs = model_factory.call_args.kwargs
        self.assertEqual(kwargs["flex_kernel_options"], {"num_stages": 2})
        self.assertEqual(kwargs["draft_kernel_backend"], "liger")
        self.assertEqual(kwargs["linear_cross_entropy_backend"], "liger")
        self.assertTrue(kwargs["compact_zero_weight_ce_rows"])

        with mock.patch("specforge.optimizer.BF16Optimizer") as optimizer:
            _ConfiguredOptimizerFactory(cfg)(mock.sentinel.module)
        self.assertEqual(optimizer.call_args.kwargs["adamw_backend"], "fused")


if __name__ == "__main__":
    unittest.main(verbosity=2)
