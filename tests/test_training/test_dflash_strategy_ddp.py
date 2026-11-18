"""DDP exposes the same protocol requirements to the generic strategy."""

import gc
import unittest
from tempfile import TemporaryDirectory

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tests.test_algorithms.test_objective_contracts import build_runtime
from tests.test_runtime._fixtures import build_strategy


class CanonicalDDPStrategyTest(unittest.TestCase):
    def test_protocol_and_host_preparation_survive_ddp(self):
        _, runtime = build_runtime("DFlash2DraftModel")
        plain = build_strategy(runtime)
        with TemporaryDirectory() as root:
            dist.init_process_group(
                "gloo", init_method=f"file://{root}/store", rank=0, world_size=1
            )
            try:
                wrapped = DDP(runtime)
                strategy = build_strategy(wrapped)
                self.assertEqual(strategy.required_features, plain.required_features)
                self.assertIs(strategy.host_inputs, plain.host_inputs)
                torch.testing.assert_close(
                    strategy.checkpoint_state_filter(runtime.state_dict()),
                    plain.checkpoint_state_filter(runtime.state_dict()),
                )
                del strategy, wrapped
                gc.collect()
            finally:
                dist.destroy_process_group()
