"""Opt-in CUDA correctness tests for DFlash's real Liger draft kernels."""

import copy
import os
import unittest
from unittest import mock

import torch
import torch.nn.functional as F
from transformers import Qwen3Config

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.optimizer import BF16Optimizer


def _draft_config():
    config = Qwen3Config(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=257,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
    )
    config.block_size = 4
    config.num_target_layers = 4
    config.dflash_config = {"mask_token_id": 0, "target_layer_ids": [1]}
    config._attn_implementation = "eager"
    return config


def _inputs():
    generator = torch.Generator(device="cuda").manual_seed(71)
    batch_size, context_length, draft_length, hidden_size = 2, 9, 8, 128
    noise = torch.randn(
        batch_size,
        draft_length,
        hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    target = torch.randn(
        batch_size,
        context_length,
        hidden_size,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    positions = torch.arange(
        context_length + draft_length, device="cuda", dtype=torch.long
    ).expand(batch_size, -1)
    return noise, target, positions


def _forward(model, inputs):
    noise, target, positions = inputs
    return model(
        position_ids=positions,
        noise_embedding=noise,
        target_hidden=target,
        attention_mask=None,
    )


@unittest.skipUnless(
    os.environ.get("SPECFORGE_RUN_LIGER_TESTS") == "1" and torch.cuda.is_available(),
    "set SPECFORGE_RUN_LIGER_TESTS=1 on an exclusive CUDA GPU",
)
class TestDFlashDraftKernelsCuda(unittest.TestCase):
    def test_bf16_full_draft_output_and_gradients_match_torch(self):
        torch.manual_seed(19)
        reference = (
            DFlashDraftModel(_draft_config(), draft_kernel_backend="torch")
            .cuda()
            .to(torch.bfloat16)
        )
        fused = (
            DFlashDraftModel(_draft_config(), draft_kernel_backend="liger")
            .cuda()
            .to(torch.bfloat16)
        )
        fused.load_state_dict(reference.state_dict())
        self.assertEqual(tuple(fused.state_dict()), tuple(reference.state_dict()))

        inputs = _inputs()
        reference_output = _forward(reference, inputs)
        fused_output = _forward(fused, inputs)
        reference_output.float().square().mean().backward()
        fused_output.float().square().mean().backward()

        self.assertTrue(
            torch.allclose(reference_output, fused_output, rtol=0.02, atol=0.02)
        )
        reference_grad = torch.cat(
            [parameter.grad.float().flatten() for parameter in reference.parameters()]
        )
        fused_grad = torch.cat(
            [parameter.grad.float().flatten() for parameter in fused.parameters()]
        )
        cosine = F.cosine_similarity(reference_grad, fused_grad, dim=0)
        relative_l2 = (fused_grad - reference_grad).norm() / reference_grad.norm()
        self.assertGreater(cosine.item(), 0.999)
        self.assertLess(relative_l2.item(), 0.03)
        self.assertTrue(torch.isfinite(fused_grad).all())

    def test_liger_model_continues_after_bf16_optimizer_restore(self):
        torch.manual_seed(29)
        model = (
            DFlashDraftModel(_draft_config(), draft_kernel_backend="liger")
            .cuda()
            .to(torch.bfloat16)
        )
        optimizer = BF16Optimizer(
            model,
            lr=1e-3,
            max_grad_norm=1.0,
            total_steps=4,
            warmup_ratio=0.0,
        )
        inputs = _inputs()

        def step(current_model, current_optimizer):
            loss = _forward(current_model, inputs).float().square().mean()
            loss.backward()
            current_optimizer.step()
            return loss.detach()

        step(model, optimizer)
        model_state = copy.deepcopy(model.state_dict())
        optimizer_state = copy.deepcopy(optimizer.state_dict())

        resumed = (
            DFlashDraftModel(_draft_config(), draft_kernel_backend="liger")
            .cuda()
            .to(torch.bfloat16)
        )
        resumed.load_state_dict(model_state)
        resumed_optimizer = BF16Optimizer(
            resumed,
            lr=1e-3,
            max_grad_norm=1.0,
            total_steps=4,
            warmup_ratio=0.0,
        )
        with mock.patch("specforge.optimizer.print_on_rank0"):
            resumed_optimizer.load_state_dict(optimizer_state)

        uninterrupted_loss = step(model, optimizer)
        resumed_loss = step(resumed, resumed_optimizer)
        self.assertTrue(torch.equal(uninterrupted_loss, resumed_loss))
        for expected, actual in zip(model.parameters(), resumed.parameters()):
            self.assertTrue(torch.equal(expected, actual))
        for expected, actual in zip(
            optimizer.fp32_params, resumed_optimizer.fp32_params
        ):
            self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
