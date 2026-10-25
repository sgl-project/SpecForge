"""D-PAL formulas, detached credit and checkpointed teacher projection."""

import unittest
from functools import partial

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.algorithms.dflash.providers import resume_contract
from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.modeling.draft.dflash import DFlashDraftModel

LOSSES = ("dpal-renyi", "dpal-kl", "dpal-la", "dpal-ce-tv")


class DPALTest(unittest.TestCase):
    def test_formulas_detached_credit_and_chunk_normalization(self):
        torch.manual_seed(42)
        logits = torch.randn(2, 3, 3, 5, requires_grad=True)
        teacher = torch.randn(2, 7, 5, requires_grad=True)
        indices = torch.tensor([[[0, 0, 1], [1, 2, 3], [3, 5, 6]]] * 2)
        ids = torch.randint(5, (2, 3, 3))
        for kind in LOSSES:
            model = OnlineDFlashModel(
                nn.Identity(),
                nn.Identity(),
                nn.Embedding(5, 5),
                mask_token_id=4,
                block_size=3,
                attention_backend="sdpa",
                loss_type=kind,
                dpal_rho=0.3,
            )
            for empty in (0, 1, 2):
                mask = torch.tensor(
                    [
                        [[0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]],
                    ]
                )
                if empty:
                    mask[-empty:] = 0
                p = (
                    teacher.detach()
                    .softmax(-1)[:, None]
                    .expand(-1, 3, -1, -1)
                    .gather(2, indices[..., None].expand(-1, -1, -1, 5))
                )
                logq = logits.log_softmax(-1)
                q = logq.exp()
                overlap = torch.minimum(p, q).sum(-1)
                actor = {
                    "dpal-renyi": -2 * (p.sqrt() * q.sqrt()).sum(-1).log(),
                    "dpal-kl": (p * (p.log() - logq)).sum(-1),
                    "dpal-la": -overlap.log(),
                    "dpal-ce-tv": 0.1 * -logq.gather(-1, ids[..., None]).squeeze(-1)
                    + 0.9 * 0.5 * (p - q).abs().sum(-1),
                }[kind]
                credit = torch.zeros_like(mask)
                for b in range(2):
                    for a in range(3):
                        reach = 1.0
                        for t in range(3):
                            if mask[b, a, t]:
                                reach = reach * (0.3 + 0.7 * overlap[b, a, t].detach())
                                for j in range(t + 1):
                                    if mask[b, a, j]:
                                        credit[b, a, j] += reach
                counts = (mask > 0).any(-1).sum(-1)
                expected_num = (
                    actor * credit / counts.clamp_min(1)[:, None, None]
                ).sum()
                expected_den = (counts > 0).sum().float()
                expected_grad = torch.autograd.grad(expected_num, logits)[0]
                for chunk in (0, 1, 2):
                    with self.subTest(kind=kind, empty=empty, chunk=chunk):
                        terms = checkpointed_chunk_reduce(
                            partial(
                                model._dflash_objective_chunk_terms,
                                target_logits=teacher.detach(),
                            ),
                            logits,
                            ids,
                            mask,
                            ids,
                            model._sequence_anchor_scale(mask),
                            indices,
                            chunk_size=chunk,
                            dim=1,
                        )
                        torch.testing.assert_close(terms[0], expected_num)
                        torch.testing.assert_close(terms[2], expected_den)
                        torch.testing.assert_close(
                            torch.autograd.grad(terms[0], logits)[0], expected_grad
                        )
                        self.assertIsNone(teacher.grad)

    def test_forward_kl_zero_teacher_mass_matches_ce(self):
        from specforge.core.dpal import dpal_actor_and_acceptance

        logits = torch.tensor([[0.2, -0.7, 0.9]], requires_grad=True)
        teacher = torch.tensor(
            [[float("-inf"), 0.0, float("-inf")]], requires_grad=True
        )
        actor, acceptance = dpal_actor_and_acceptance(logits, teacher, "dpal-kl")
        torch.testing.assert_close(actor, -logits.log_softmax(-1)[:, 1])
        actor.sum().backward()
        expected = logits.detach().softmax(-1)
        expected[:, 1] -= 1
        torch.testing.assert_close(logits.grad, expected)
        self.assertFalse(acceptance.requires_grad)
        self.assertIsNone(teacher.grad)

    def test_real_forward_backward_projects_teacher_once(self):
        self._check_forward("cpu", torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_bf16_cuda_forward_backward(self):
        self._check_forward("cuda", torch.bfloat16)

    def _check_forward(self, device, dtype):
        for kind in LOSSES:
            torch.manual_seed(42)
            cfg = Qwen3Config(
                hidden_size=32,
                intermediate_size=64,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_hidden_layers=3,
                num_target_layers=36,
                head_dim=8,
                max_position_embeddings=128,
                vocab_size=64,
                layer_types=["full_attention"] * 3,
                dflash_config={"block_size": 16, "target_layer_ids": [1, 17, 33]},
            )
            cfg._attn_implementation = "sdpa"
            model = OnlineDFlashModel(
                DFlashDraftModel(cfg),
                nn.Linear(32, 64, bias=False).requires_grad_(False),
                nn.Embedding(64, 32).requires_grad_(False),
                mask_token_id=63,
                block_size=16,
                num_anchors=3,
                attention_backend="sdpa",
                loss_type=kind,
                dpal_rho=0.3,
                teacher_metrics=False,
            ).to(device=device, dtype=dtype)
            contract = resume_contract(None, model.draft_model, model)
            self.assertEqual(contract["dflash_dpal_rho"], 0.3)
            self.assertEqual(contract["dflash_loss_type"], kind)
            ids = torch.randint(63, (2, 48), device=device)
            features = torch.randn(2, 48, 96, device=device, dtype=dtype)
            teacher = torch.randn(
                2, 48, 32, device=device, dtype=dtype, requires_grad=True
            )
            mask = torch.ones_like(ids, dtype=torch.float)
            mask[1] = 0
            reference = None
            for chunk in (0, 1, 2):
                for detailed in (False, True):
                    with self.subTest(kind=kind, chunk=chunk, detailed=detailed):
                        projections = []
                        hook = model.lm_head.register_forward_pre_hook(
                            lambda module, args: (
                                projections.append(tuple(args[0].shape))
                                if args[0] is teacher
                                else None
                            )
                        )
                        model.zero_grad(set_to_none=True)
                        model.objective_chunk_blocks = chunk
                        torch.manual_seed(7)
                        try:
                            loss, _, metrics = model(
                                ids,
                                features,
                                mask,
                                target_last_hidden_states=teacher,
                                collect_detailed_metrics=detailed,
                            )
                            loss.backward()
                        finally:
                            hook.remove()
                        grads = torch.cat(
                            [
                                p.grad.flatten()
                                for p in model.draft_model.parameters()
                                if p.grad is not None
                            ]
                        )
                        self.assertTrue(
                            torch.isfinite(loss) and torch.isfinite(grads).all()
                        )
                        self.assertGreater(grads.abs().sum().item(), 0)
                        self.assertEqual(projections, [(2, 48, 32)])
                        self.assertIsNone(teacher.grad)
                        self.assertIn("dpal_loss", metrics["ratio_metrics"])
                        if reference is None:
                            reference = (loss.detach(), grads.detach().clone())
                        tolerance = (
                            dict(atol=2e-6, rtol=2e-5)
                            if dtype == torch.float32
                            else dict(atol=2e-3, rtol=2e-2)
                        )
                        torch.testing.assert_close(loss, reference[0], **tolerance)
                        torch.testing.assert_close(grads, reference[1], **tolerance)
            with self.assertRaisesRegex(ValueError, "target_last_hidden_states"):
                model(ids, features, mask, collect_detailed_metrics=False)

    def test_selector_and_lk_combinations_are_rejected(self):
        for kind in LOSSES:
            for selector, lk in ((nn.Identity(), None), (None, "tv")):
                draft = nn.Identity()
                draft.candidate_selector = selector
                with (
                    self.subTest(kind=kind),
                    self.assertRaisesRegex(ValueError, "D-PAL"),
                ):
                    OnlineDFlashModel(
                        draft,
                        nn.Identity(),
                        nn.Embedding(5, 5),
                        mask_token_id=4,
                        loss_type=kind,
                        lk_loss_type=lk,
                    )


if __name__ == "__main__":
    unittest.main()
