import copy
import tempfile
import unittest

import torch
from torch import nn
from torch.testing import assert_close

from specforge.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
    OnlineDominoModel,
    OnlineDSparkModel,
)
from specforge.modeling.draft.dflash import (
    DFlashDraftModel,
    dflash_sparse_resume_contract,
)
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.modeling.draft.domino import DominoDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from tests.test_modeling.test_layers.test_indexed_attention import sparse_config


class SparseDraftTest(unittest.TestCase):
    def cfg(self, kind, mode="joint"):
        cfg = sparse_config(kind, "gathered", mode)
        cfg.is_causal = True
        cfg.dflash_config.update(
            mask_token_id=31,
            markov_rank=4,
            enable_confidence_head=False,
            conv_kernel_size=2,
            conv_group_size=4,
            selector_rank=8,
            selector_top_k=4,
            emb_dim=8,
            gru_hidden_dim=8,
        )
        return cfg

    def inputs(self, device="cpu"):
        return dict(
            position_ids=torch.arange(9, device=device)[None],
            noise_embedding=torch.randn(2, 4, 16, device=device),
            target_hidden=torch.randn(2, 5, 16, device=device),
            return_auxiliary_losses=True,
        )

    def test_family_forward_and_checkpoint_recompute(self):
        for kind in ("dsa", "qsa"):
            for cls in (
                DFlashDraftModel,
                DFlash2DraftModel,
                DominoDraftModel,
                DSparkDraftModel,
            ):
                with self.subTest(kind=kind, model=cls.__name__):
                    torch.manual_seed(31)
                    model = cls(self.cfg(kind))
                    checkpointed = copy.deepcopy(model)
                    checkpointed.gradient_checkpointing_enable()
                    inputs = self.inputs()
                    results = []
                    for m in (model, checkpointed):
                        hidden, aux = m(**inputs)
                        (hidden.square().mean() + aux).backward()
                        results.append((hidden, aux))
                    assert_close(*results)
                    for (n, p), (_, q) in zip(
                        model.named_parameters(), checkpointed.named_parameters()
                    ):
                        if p.grad is not None:
                            assert_close(p.grad, q.grad, msg=n)

    def test_online_loss_terms_preserve_indexer_gradients(self):
        for kind in ("dsa", "qsa"):
            for cls, wrapper in (
                (DFlashDraftModel, OnlineDFlashModel),
                (DFlash2DraftModel, OnlineDFlashModel),
                (DSparkDraftModel, OnlineDSparkModel),
                (DominoDraftModel, OnlineDominoModel),
            ):
                with self.subTest(kind=kind, cls=cls.__name__):
                    torch.manual_seed(19)
                    draft = cls(self.cfg(kind))
                    model = wrapper(
                        draft,
                        nn.Linear(16, 32, bias=False).requires_grad_(False),
                        nn.Embedding(32, 16).requires_grad_(False),
                        mask_token_id=31,
                        block_size=4,
                        attention_backend="sdpa",
                        num_anchors=2,
                        objective_chunk_blocks=1,
                    )
                    batch = dict(
                        input_ids=torch.randint(0, 30, (2, 12)),
                        hidden_states=torch.randn(2, 12, 16),
                        loss_mask=torch.ones(2, 12),
                    )
                    if wrapper is not OnlineDominoModel:
                        batch["target_last_hidden_states"] = torch.randn(2, 12, 16)
                    loss, _, metrics = model(**batch)
                    if "loss_terms" in metrics:
                        numerator, denominator = metrics["loss_terms"]
                        assert_close(loss, numerator / denominator)
                        numerator.backward()
                    else:
                        loss.backward()
                    total = 0
                    for name, p in draft.named_parameters():
                        if ".indexer." in name:
                            self.assertIsNotNone(p.grad, name)
                            self.assertTrue(torch.isfinite(p.grad).all(), name)
                            total += p.grad.abs().sum().item()
                    self.assertGreater(total, 0)

    def test_warmup_freezes_backbone_and_resume_records_selection_settings(self):
        for kind in ("qsa", "dsa"):
            model = DFlashDraftModel(self.cfg(kind, "warmup"))
            for name, p in model.named_parameters():
                self.assertEqual(p.requires_grad, ".indexer." in name, name)
            _, aux = model(**self.inputs())
            aux.backward()
            contract = dflash_sparse_resume_contract(model)
            self.assertIn("draft_sparse_attention_semantics", contract)
            model.config.sparse_attention["loss_weight"] = 2
            self.assertNotEqual(contract, dflash_sparse_resume_contract(model))

    def test_save_load_preserves_outputs_and_frozen_mode(self):
        for kind in ("dsa", "qsa"):
            model = DFlashDraftModel(self.cfg(kind, "frozen")).eval()
            inputs = self.inputs()
            inputs.pop("return_auxiliary_losses")
            expected = model(**inputs)
            with tempfile.TemporaryDirectory() as folder:
                model.save_pretrained(folder)
                restored = DFlashDraftModel.from_pretrained(
                    folder, attn_implementation="sdpa"
                ).eval()
            assert_close(expected, restored(**inputs), atol=0, rtol=0)
            self.assertFalse(
                any(
                    p.requires_grad
                    for n, p in restored.named_parameters()
                    if ".indexer." in n
                )
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_bfloat16_cuda_training(self):
        for kind in ("dsa", "qsa"):
            model = DFlashDraftModel(self.cfg(kind)).cuda().bfloat16()
            inputs = self.inputs("cuda")
            inputs["noise_embedding"] = inputs["noise_embedding"].bfloat16()
            inputs["target_hidden"] = inputs["target_hidden"].bfloat16()
            hidden, aux = model(**inputs)
            (hidden.float().square().mean() + aux).backward()
            for n, p in model.named_parameters():
                self.assertIsNotNone(p.grad, n)
                self.assertTrue(torch.isfinite(p.grad).all(), n)


if __name__ == "__main__":
    unittest.main()
