"""MTP SDPA and chunked head loss preserve loss and gradients."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch

from specforge.algorithms.objectives import build_objective_model
from specforge.algorithms.objectives.protocols.mtp import prepare_supervised_rows
from specforge.modeling import build_draft_model
from tests.test_algorithms.test_objectives import training_batch
from tests.test_checkpoint.test_native_presets import native_configs


def owned_mtp_config():
    return replace(
        next(native_configs(small=True)),
        target_bindings=(),
        target_logit_transform=None,
    )


def gradients(model):
    return {name: param.grad for name, param in model.named_parameters()}


class MTPBackendObjectiveTest(unittest.TestCase):
    def test_sdpa_chunked_loss_and_gradients(self):
        cases = [("cpu", torch.float32)]
        if torch.cuda.is_available():
            cases += [("cuda", torch.float32), ("cuda", torch.bfloat16)]
        for device, dtype in cases:
            rtol, atol = (3e-2, 3e-3) if dtype == torch.bfloat16 else (3e-4, 2e-6)
            config = owned_mtp_config()
            torch.manual_seed(31)
            reference = build_draft_model(config, device=device, dtype=dtype)
            batch = training_batch(config, device)
            batch = {
                k: v.to(dtype) if v.is_floating_point() else v for k, v in batch.items()
            }
            batch["position_ids"] = torch.arange(9, device=device)[None] + 17
            batch["attention_mask"][0, -2:] = 0
            batch["loss_mask"][:, 3:5] = 0
            selected = prepare_supervised_rows(batch, config)["mtp_supervised_rows"]
            dense_batch = batch | {
                "mtp_supervised_rows": torch.arange(8, device=device)
            }
            expected = build_objective_model(
                reference, options={"objective_chunk_size": 0}
            )(dense_batch, step=0, total_steps=None)
            expected.loss.backward()
            for chunk_size in (0, 1, 3, 4096):
                with self.subTest(device=device, dtype=dtype, chunk_size=chunk_size):
                    model = build_draft_model(
                        config, device=device, dtype=dtype, backend="sdpa"
                    )
                    model.load_state_dict(reference.state_dict())
                    runtime = build_objective_model(
                        model, options={"objective_chunk_size": chunk_size}
                    )
                    with (
                        patch.object(
                            model.lm_head, "forward", wraps=model.lm_head.forward
                        ) as head,
                        patch.object(
                            torch.nn.functional,
                            "scaled_dot_product_attention",
                            wraps=torch.nn.functional.scaled_dot_product_attention,
                        ) as sdpa,
                    ):
                        actual = runtime(batch, step=0, total_steps=None)
                        self.assertTrue(sdpa.called)
                        self.assertEqual(
                            sum(c.args[0].shape[0] for c in head.call_args_list),
                            selected.numel(),
                        )
                        forward_calls = head.call_count
                        actual.loss.backward()
                        self.assertEqual(
                            head.call_count,
                            forward_calls * (2 if runtime.execution.checkpoint else 1),
                        )
                        self.assertLessEqual(
                            max(call.args[0].shape[0] for call in head.call_args_list),
                            chunk_size or 8,
                        )
                    torch.testing.assert_close(
                        actual.loss, expected.loss, rtol=rtol, atol=atol
                    )
                    torch.testing.assert_close(
                        gradients(model), gradients(reference), rtol=rtol, atol=atol
                    )
                    torch.testing.assert_close(actual.ratios, expected.ratios)

    def test_frozen_backbone_and_empty_loss_mask(self):
        model = build_draft_model(owned_mtp_config()).requires_grad_(False)
        model.lm_head.requires_grad_(True)
        batch = training_batch(model.config.draft)
        runtime = build_objective_model(model, options={"objective_chunk_size": 2})
        runtime(batch, step=0, total_steps=None).loss.backward()
        self.assertIsNotNone(model.lm_head.weight.grad)
        batch["loss_mask"].zero_()
        self.assertEqual(runtime(batch, step=0, total_steps=None).loss.item(), 0)

    def test_host_rows_and_empty_backward(self):
        from specforge.runtime.contracts import TrainBatch
        from specforge.training.strategies.base import StepContext
        from specforge.training.strategies.canonical import CanonicalTrainStrategy

        config = owned_mtp_config()
        for empty in (False, True):
            model = build_draft_model(config)
            runtime = build_objective_model(model)
            batch = training_batch(config)
            batch["loss_mask"][:, ::2] = 0
            if empty:
                batch["loss_mask"].zero_()
            host = prepare_supervised_rows(batch, config)
            expected = runtime(batch, step=0, total_steps=None)
            strategy = CanonicalTrainStrategy(
                runtime, name="mtp", objective="mtp", required_features=batch
            )
            with patch.object(runtime, "forward", wraps=runtime.forward) as forward:
                actual = strategy.forward_loss(
                    TrainBatch(sample_ids=["test"], strategy="mtp", tensors=batch),
                    StepContext(),
                )
            torch.testing.assert_close(
                forward.call_args.args[0]["mtp_supervised_rows"],
                host["mtp_supervised_rows"],
            )
            torch.testing.assert_close(actual.loss, expected.loss)
            actual.loss.backward()
            if empty:
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertEqual(parameter.grad.count_nonzero().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_host_rows_avoid_device_nonzero(self):
        config = owned_mtp_config()
        batch = training_batch(config)
        batch.update(prepare_supervised_rows(batch, config))
        batch = {key: value.cuda() for key, value in batch.items()}
        runtime = build_objective_model(build_draft_model(config, device="cuda"))
        previous = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            with patch.object(
                torch.Tensor, "nonzero", side_effect=AssertionError("device row lookup")
            ):
                runtime(batch, step=0, total_steps=None).loss.backward()
        finally:
            torch.cuda.set_sync_debug_mode(previous)

    def test_chunk_sizes_validate(self):
        from specforge.algorithms.objectives.protocols.mtp import MTPObjective

        for value in (-1, True, 0.5):
            with self.assertRaises(ValueError):
                MTPObjective(objective_chunk_size=value)

    def test_fp32_ce_against_legacy_bf16_and_fp64(self):
        from specforge.algorithms.objectives.losses import cross_entropy

        logits = (
            torch.randn(4, 248320, generator=torch.Generator().manual_seed(31))
            .bfloat16()
            .requires_grad_()
        )
        reference_logits = logits.detach().double().requires_grad_()
        labels = torch.tensor([1, 123, 45000, 248319])
        actual = cross_entropy(logits, labels)
        reference = torch.nn.functional.cross_entropy(
            reference_logits, labels, reduction="none"
        )
        legacy = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(actual.double(), reference, rtol=1e-6, atol=2e-6)
        torch.testing.assert_close(actual, legacy.float(), rtol=0.01, atol=0.01)
        self.assertLess(
            (actual - reference).abs().max(), (legacy - reference).abs().max()
        )
        actual_grad = torch.autograd.grad(actual.mean(), logits)[0].double()
        legacy_grad = torch.autograd.grad(legacy.mean(), logits)[0].double()
        reference_grad = torch.autograd.grad(reference.mean(), reference_logits)[0]
        # The FP32 CE backward is rounded back to the BF16 input dtype.
        torch.testing.assert_close(actual_grad, reference_grad, rtol=0.004, atol=1e-9)
        # Historical BF16 log-softmax changes small probabilities substantially;
        # compare the aggregate gradient rather than impose elementwise parity.
        relative_change = torch.linalg.vector_norm(
            actual_grad - legacy_grad
        ) / torch.linalg.vector_norm(reference_grad)
        self.assertLess(relative_change, 0.01)
        self.assertLess(
            torch.linalg.vector_norm(actual_grad - reference_grad),
            torch.linalg.vector_norm(legacy_grad - reference_grad),
        )


if __name__ == "__main__":
    unittest.main()
