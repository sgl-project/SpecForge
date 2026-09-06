from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from specforge.training.strategies.base import DFlashTrainStrategy


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _TinyDraft(nn.Module):
    """Stands in for a DFlash2 draft: a parameter plus the selector schedule attributes."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.selector_loss_alpha = 0.7
        self.selector_warmup_ratio = 0.0
        self.selector_ramp_ratio = 0.0

    def forward(self, x):
        return self.linear(x)


class DFlashStrategySelectorAttrTest(unittest.TestCase):
    def test_reads_selector_alpha_on_plain_module(self):
        strategy = DFlashTrainStrategy(_TinyDraft())
        self.assertAlmostEqual(strategy._selector_loss_alpha(None), 0.7)

    def test_reads_selector_alpha_through_ddp_wrapper(self):
        # NO_SHARD wraps the draft in DistributedDataParallel; the wrapper does not
        # forward attribute access, so the strategy has to look through it.
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        try:
            wrapped = DDP(_TinyDraft())
            self.assertFalse(hasattr(wrapped, "selector_loss_alpha"))
            strategy = DFlashTrainStrategy(wrapped)
            self.assertAlmostEqual(strategy._selector_loss_alpha(None), 0.7)
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
