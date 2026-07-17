import copy
import unittest
from unittest import mock

import torch

from specforge.optimizer import BF16Optimizer

CUDA = torch.cuda.is_available()


def _optimizer(model, *, backend="torch"):
    return BF16Optimizer(
        model,
        lr=1e-3,
        max_grad_norm=0.25,
        warmup_ratio=0.0,
        total_steps=8,
        adamw_backend=backend,
    )


class TestBF16OptimizerBackend(unittest.TestCase):
    def test_default_preserves_torch_adamw_and_legacy_resume(self):
        model = torch.nn.Linear(2, 2)
        optimizer = _optimizer(model)
        group = optimizer.optimizer.param_groups[0]
        self.assertIsNone(group["foreach"])
        self.assertIsNone(group["fused"])

        legacy_state = optimizer.state_dict()
        legacy_state.pop("adamw_backend")
        resumed = _optimizer(torch.nn.Linear(2, 2))
        with mock.patch("specforge.optimizer.print_on_rank0"):
            resumed.load_state_dict(legacy_state)
        self.assertEqual(resumed.adamw_backend, "torch")

    def test_cpu_and_unknown_fused_backend_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one CUDA device"):
            _optimizer(torch.nn.Linear(2, 2), backend="fused")
        with self.assertRaisesRegex(ValueError, "adamw_backend"):
            _optimizer(torch.nn.Linear(2, 2), backend="unknown")

    def test_checkpoint_backend_must_match(self):
        optimizer = _optimizer(torch.nn.Linear(2, 2))
        state = optimizer.state_dict()
        state["adamw_backend"] = "fused"
        with self.assertRaisesRegex(ValueError, "does not match"):
            optimizer.load_state_dict(state)

    @unittest.skipUnless(CUDA, "fused AdamW requires CUDA")
    def test_fused_step_and_checkpoint_resume_match_torch(self):
        torch.manual_seed(91)
        reference_model = torch.nn.Linear(
            128, 128, device="cuda:0", dtype=torch.bfloat16
        )
        fused_model = copy.deepcopy(reference_model)
        reference = _optimizer(reference_model)
        fused = _optimizer(fused_model, backend="fused")

        for _ in range(3):
            for reference_param, fused_param in zip(
                reference_model.parameters(), fused_model.parameters()
            ):
                gradient = torch.randn_like(reference_param)
                reference_param.grad = gradient.clone()
                fused_param.grad = gradient.clone()
            reference_norm = reference.step()
            fused_norm = fused.step()
            torch.testing.assert_close(reference_norm, fused_norm)

        for reference_param, fused_param in zip(
            reference_model.parameters(), fused_model.parameters()
        ):
            torch.testing.assert_close(reference_param, fused_param, rtol=0, atol=0)
        for reference_master, fused_master in zip(
            reference.fp32_params, fused.fp32_params
        ):
            torch.testing.assert_close(
                reference_master, fused_master, rtol=1e-7, atol=1e-7
            )

        checkpoint = copy.deepcopy(fused.state_dict())
        resumed_model = copy.deepcopy(fused_model)
        resumed = _optimizer(resumed_model, backend="fused")
        with mock.patch("specforge.optimizer.print_on_rank0"):
            resumed.load_state_dict(checkpoint)

        for fused_param, resumed_param in zip(
            fused_model.parameters(), resumed_model.parameters()
        ):
            gradient = torch.randn_like(fused_param)
            fused_param.grad = gradient.clone()
            resumed_param.grad = gradient.clone()
        fused.step()
        resumed.step()
        torch.cuda.synchronize()

        for expected, actual in zip(
            fused_model.parameters(), resumed_model.parameters()
        ):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        for expected, actual in zip(fused.fp32_params, resumed.fp32_params):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
