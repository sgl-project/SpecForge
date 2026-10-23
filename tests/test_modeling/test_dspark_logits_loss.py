import unittest

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.modeling.draft.dspark import DSparkDraftModel


def build_model(device="cpu"):
    config = Qwen3Config(
        architectures=["DSparkDraftModel"],
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        num_target_layers=4,
        head_dim=32,
        vocab_size=64,
        block_size=4,
        layer_types=["full_attention"] * 2,
        is_causal=True,
        dflash_config={
            "projector_type": "dspark",
            "target_layer_ids": [0, 2],
            "mask_token_id": 63,
            "markov_head_type": "vanilla",
            "markov_rank": 8,
            "enable_confidence_head": False,
            "confidence_head_alpha": 0.0,
        },
        attn_implementation="sdpa",
    )
    draft = DSparkDraftModel(config)
    head = nn.Linear(128, 64, bias=False).requires_grad_(False)
    embedding = nn.Embedding(64, 128).requires_grad_(False)
    return OnlineDSparkModel(
        draft,
        head,
        embedding,
        mask_token_id=63,
        block_size=4,
        attention_backend="sdpa",
        num_anchors=2,
        dspark_confidence_head_alpha=0.0,
        objective_chunk_blocks=1,
    ).to(device)


class DSparkLogitsLossTest(unittest.TestCase):

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_eager_and_fused_return_matching_terms(self):
        from specforge.core.fused_loss import ce_l1_loss

        torch.manual_seed(31)
        model = build_model("cuda")
        self.assertTrue(model.fused_logits_loss)
        logits = torch.randn(2, 3, 64, device="cuda", requires_grad=True)
        teacher = torch.randn_like(logits, requires_grad=True)
        labels = torch.randint(64, (2, 3), device="cuda")
        reference = ce_l1_loss(logits, labels, teacher_logits=teacher, top_k=3)
        reference_grad = torch.autograd.grad(
            (reference.cross_entropy + reference.teacher_l1).sum()
            + reference.topk_values.sum(),
            logits,
        )[0]
        for enabled, expected in ((False, "eager"), (True, "fused")):
            with self.subTest(backend=expected):
                model.fused_logits_loss = enabled
                self.assertEqual(model._logits_loss_backend(logits), expected)
                actual = model._logits_loss_terms(
                    logits, labels, teacher_logits=teacher, top_k=3
                )
                torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)
                grad, teacher_grad = torch.autograd.grad(
                    (actual.cross_entropy + actual.teacher_l1).sum()
                    + actual.topk_values.sum(),
                    (logits, teacher),
                    allow_unused=True,
                )
                torch.testing.assert_close(grad, reference_grad, rtol=1e-4, atol=1e-5)
                self.assertIsNone(teacher_grad)
        # Unsupported devices fall back to eager, including BF16.
        self.assertEqual(model._logits_loss_backend(logits.cpu().bfloat16()), "eager")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_fused_logits_preserves_markov_and_confidence_gradients(self):
        from specforge.modeling.draft.dspark import AcceptRatePredictor

        torch.manual_seed(42)
        model = build_model(device="cuda").bfloat16()
        model.draft_model.confidence_head = AcceptRatePredictor(128).cuda().bfloat16()
        model.draft_model.confidence_head_with_markov = False
        model.dspark_confidence_head_alpha = 1.0
        model.fused_logits_loss = True
        batch = dict(
            input_ids=torch.randint(0, 62, (2, 12), device="cuda"),
            hidden_states=torch.randn(2, 12, 256, device="cuda", dtype=torch.bfloat16),
            loss_mask=torch.ones(2, 12, device="cuda"),
            target_last_hidden_states=torch.randn(
                2, 12, 128, device="cuda", dtype=torch.bfloat16
            ),
        )
        for l1_alpha in (0.0, 0.9):
            for chunk, checkpoint in ((0, False), (0, True), (1, False), (1, True)):
                with self.subTest(
                    l1_alpha=l1_alpha, chunk=chunk, checkpoint=checkpoint
                ):
                    model.dspark_l1_loss_alpha = l1_alpha
                    model.objective_chunk_blocks = chunk
                    model.objective_checkpointing = checkpoint
                    outputs = []
                    for fused in (False, True):
                        model.fused_logits_loss = fused
                        model.zero_grad(set_to_none=True)
                        torch.manual_seed(17)
                        loss, accuracy, metrics = model(**batch)
                        loss.backward()
                        grads = {
                            n: p.grad.detach().float().clone()
                            for n, p in model.named_parameters()
                            if p.requires_grad
                        }
                        outputs.append((loss.detach(), accuracy, metrics, grads))
                    reference, actual = outputs
                    torch.testing.assert_close(
                        actual[:2], reference[:2], rtol=1e-4, atol=1e-5
                    )
                    torch.testing.assert_close(
                        actual[2], reference[2], rtol=1e-4, atol=1e-5
                    )
                    for name, grad in reference[3].items():
                        error = (actual[3][name] - grad).norm() / grad.norm().clamp_min(
                            1e-12
                        )
                        self.assertLess(error.item(), 0.01, name)
