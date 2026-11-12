import os
import tempfile
import unittest

import torch
import torch.distributed as dist
from torch import nn

from specforge.optimizer import BF16Optimizer


def _set_grads(model: nn.Module, value: float) -> None:
    for p in model.parameters():
        p.grad = torch.full_like(p, value)


def _params(model: nn.Module):
    return [p.detach().clone() for p in model.parameters()]


class TestSpikeGuard(unittest.TestCase):
    _created_pg = False

    @classmethod
    def setUpClass(cls):
        # BF16Optimizer.load_state_dict logs via rank0, which needs a group.
        if dist.is_available() and not dist.is_initialized():
            store = dist.FileStore(
                os.path.join(tempfile.mkdtemp(prefix="spike_pg_"), "store"), 1
            )
            dist.init_process_group("gloo", store=store, rank=0, world_size=1)
            cls._created_pg = True

    @classmethod
    def tearDownClass(cls):
        if cls._created_pg:
            dist.destroy_process_group()

    def _optimizer(self, model: nn.Module, ratio: float) -> BF16Optimizer:
        return BF16Optimizer(
            model,
            lr=1e-2,
            max_grad_norm=1.0,
            total_steps=100,
            warmup_ratio=0.0,
            lr_scheduler="cosine",
            spike_guard_ratio=ratio,
            spike_guard_window=8,
            spike_guard_min_history=3,
        )

    def test_spike_skips_the_update_but_keeps_the_schedule(self):
        torch.manual_seed(0)
        model = nn.Linear(4, 4)
        opt = self._optimizer(model, ratio=10.0)
        for _ in range(3):
            _set_grads(model, 0.01)
            opt.step()
        before = _params(model)
        lr_before = opt.get_learning_rate()

        _set_grads(model, 100.0)  # thousands of times the recent norm
        norm = opt.step()

        self.assertTrue(opt.last_step_skipped)
        self.assertEqual(opt.skipped_steps, 1)
        self.assertGreater(float(norm), 0.0)
        for p, b in zip(model.parameters(), before):
            self.assertTrue(torch.equal(p.detach(), b))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        # The cosine schedule moved on: the skipped step keeps its place.
        self.assertNotEqual(opt.get_learning_rate(), lr_before)

        # The spike did not enter the baseline, so the next ordinary step updates.
        _set_grads(model, 0.01)
        opt.step()
        self.assertFalse(opt.last_step_skipped)
        self.assertEqual(opt.skipped_steps, 1)
        self.assertTrue(
            any(not torch.equal(p.detach(), b) for p, b in zip(model.parameters(), before))
        )
        state = opt.state_dict()["spike_guard"]
        self.assertEqual(state["skipped_steps"], 1)
        self.assertEqual(len(state["grad_norm_history"]), 4)

    def test_off_by_default_and_silent_until_history(self):
        model = nn.Linear(4, 4)
        opt = self._optimizer(model, ratio=0.0)
        for _ in range(3):
            _set_grads(model, 0.01)
            opt.step()
        _set_grads(model, 100.0)
        opt.step()
        self.assertFalse(opt.last_step_skipped)
        self.assertEqual(opt.skipped_steps, 0)

        model = nn.Linear(4, 4)
        opt = self._optimizer(model, ratio=10.0)
        _set_grads(model, 0.01)
        opt.step()  # one update: below spike_guard_min_history
        before = _params(model)
        _set_grads(model, 100.0)
        opt.step()
        self.assertFalse(opt.last_step_skipped)
        self.assertTrue(
            any(not torch.equal(p.detach(), b) for p, b in zip(model.parameters(), before))
        )

    def test_state_round_trips_through_checkpoints(self):
        model = nn.Linear(4, 4)
        opt = self._optimizer(model, ratio=10.0)
        for _ in range(3):
            _set_grads(model, 0.01)
            opt.step()
        _set_grads(model, 100.0)
        opt.step()
        state = opt.state_dict()

        resumed = self._optimizer(nn.Linear(4, 4), ratio=10.0)
        resumed.load_state_dict(state)
        self.assertEqual(resumed.skipped_steps, 1)
        self.assertEqual(len(resumed._grad_norm_history), 3)


if __name__ == "__main__":
    unittest.main()
