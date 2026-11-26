import math
import unittest

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from specforge.lr_scheduler import CosineAnnealingWarmupLR, _WarmupScheduler
from specforge.optimizer import BF16Optimizer

BASE_LR = 1e-3


def _expected_cosine(total_steps, warmup_steps, base_lr=BASE_LR):
    """Closed-form LR after each of ``total_steps`` scheduler positions."""
    decay_steps = total_steps - warmup_steps
    expected = []
    for step in range(total_steps):
        if step < warmup_steps:
            expected.append((step + 1) / warmup_steps * base_lr)
            continue
        progress = min(step - warmup_steps, decay_steps) / decay_steps
        expected.append(base_lr * (1 + math.cos(math.pi * progress)) / 2)
    return expected


def _observed_lrs(scheduler, optimizer, steps):
    observed = [optimizer.param_groups[0]["lr"]]
    for _ in range(steps):
        scheduler.step()
        observed.append(optimizer.param_groups[0]["lr"])
    return observed


def _sgd():
    return torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=BASE_LR)


def _optimizer(*, scheduler="cosine", total_steps=4, warmup_ratio=0.0):
    model = torch.nn.Linear(2, 2, bias=False)
    optimizer = BF16Optimizer(
        model,
        lr=1e-3,
        max_grad_norm=1.0,
        total_steps=total_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler=scheduler,
    )
    return model, optimizer


class TestLearningRateScheduler(unittest.TestCase):
    def test_constant_scheduler_keeps_base_lr_without_warmup(self):
        model, optimizer = _optimizer(scheduler="constant")
        observed = [optimizer.get_learning_rate()]
        for _ in range(4):
            model.weight.grad = torch.ones_like(model.weight)
            optimizer.step()
            observed.append(optimizer.get_learning_rate())
        self.assertEqual(observed, [1e-3] * 5)

    def test_constant_scheduler_supports_linear_warmup(self):
        _model, optimizer = _optimizer(
            scheduler="constant", total_steps=4, warmup_ratio=0.5
        )
        self.assertAlmostEqual(optimizer.get_learning_rate(), 5e-4)
        optimizer.scheduler.step()
        self.assertAlmostEqual(optimizer.get_learning_rate(), 1e-3)
        optimizer.scheduler.step()
        self.assertAlmostEqual(optimizer.get_learning_rate(), 1e-3)

    def test_unknown_scheduler_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported lr_scheduler"):
            _optimizer(scheduler="linear")

    def test_resume_rejects_scheduler_change(self):
        _model, cosine = _optimizer(scheduler="cosine")
        _other_model, constant = _optimizer(scheduler="constant")
        with self.assertRaisesRegex(ValueError, "checkpoint optimizer used"):
            constant.load_state_dict(cosine.state_dict())

    def test_legacy_checkpoint_defaults_to_cosine(self):
        _model, cosine = _optimizer(scheduler="cosine")
        state = cosine.state_dict()
        state.pop("lr_scheduler_type")
        self.assertEqual(state.get("lr_scheduler_type", "cosine"), "cosine")


class TestCosineAnnealingWarmupLR(unittest.TestCase):
    def assertLrsAlmostEqual(self, observed, expected):
        self.assertEqual(len(observed), len(expected))
        for step, (got, want) in enumerate(zip(observed, expected)):
            self.assertAlmostEqual(got, want, places=12, msg=f"step {step}")

    def test_cosine_without_warmup_starts_at_peak_and_follows_closed_form(self):
        _model, optimizer = _optimizer(scheduler="cosine", total_steps=10)
        observed = _observed_lrs(optimizer.scheduler, optimizer.optimizer, 9)
        self.assertAlmostEqual(observed[0], BASE_LR, places=12)
        self.assertLrsAlmostEqual(observed, _expected_cosine(10, 0))

    def test_cosine_with_warmup_follows_closed_form_and_never_exceeds_peak(self):
        _model, optimizer = _optimizer(
            scheduler="cosine", total_steps=10, warmup_ratio=0.2
        )
        observed = _observed_lrs(optimizer.scheduler, optimizer.optimizer, 9)
        self.assertLrsAlmostEqual(observed, _expected_cosine(10, 2))
        self.assertLessEqual(max(observed), BASE_LR + 1e-15)

    def test_cosine_rejects_warmup_covering_every_step(self):
        with self.assertRaisesRegex(ValueError, "warmup_steps"):
            _optimizer(scheduler="cosine", total_steps=4, warmup_ratio=1.0)

    def test_cosine_rejects_non_positive_total_steps(self):
        with self.assertRaisesRegex(ValueError, "total_steps"):
            CosineAnnealingWarmupLR(_sgd(), total_steps=0)

    def test_cosine_resume_matches_uninterrupted_schedule(self):
        reference_opt = _sgd()
        reference = CosineAnnealingWarmupLR(reference_opt, 10, warmup_steps=2)
        expected = _observed_lrs(reference, reference_opt, 9)

        first_opt = _sgd()
        first = CosineAnnealingWarmupLR(first_opt, 10, warmup_steps=2)
        before = _observed_lrs(first, first_opt, 4)

        resumed_opt = _sgd()
        resumed = CosineAnnealingWarmupLR(resumed_opt, 10, warmup_steps=2)
        resumed.load_state_dict(first.state_dict())
        after = _observed_lrs(resumed, resumed_opt, 5)

        self.assertLrsAlmostEqual(before + after[1:], expected)

    def test_cosine_loads_legacy_torch_cosine_state(self):
        legacy_opt = _sgd()
        legacy = _WarmupScheduler(
            legacy_opt, 2, CosineAnnealingLR(legacy_opt, 8, eta_min=0.0)
        )
        for _ in range(4):
            legacy.step()
        legacy_state = legacy.state_dict()
        self.assertEqual(legacy_state["after_scheduler_type"], "CosineAnnealingLR")

        resumed_opt = _sgd()
        resumed = CosineAnnealingWarmupLR(resumed_opt, 10, warmup_steps=2)
        resumed.load_state_dict(legacy_state)
        after = _observed_lrs(resumed, resumed_opt, 5)

        # The first value is the checkpoint's own (possibly inflated) _last_lr;
        # every later step comes from the closed form.
        self.assertLrsAlmostEqual(after[1:], _expected_cosine(10, 2)[5:])


if __name__ == "__main__":
    unittest.main()
