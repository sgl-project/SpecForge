import unittest

import torch

from specforge.optimizer import BF16Optimizer


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


if __name__ == "__main__":
    unittest.main()
