"""A runner sees the same objective model through DDP as it sees bare."""

import gc
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from specforge.runtime.contracts import TrainBatch
from tests.test_algorithms.test_objective_contracts import build_runtime
from tests.test_algorithms.test_objectives import training_batch
from tests.test_runtime._fixtures import build_runner, forward_program


class CanonicalDDPProgramTest(unittest.TestCase):
    def test_protocol_and_host_preparation_survive_ddp(self):
        config, runtime = build_runtime("DFlash2DraftModel")
        batch = TrainBatch(["a"], runtime.objective_name, training_batch(config))
        backend = SimpleNamespace(
            state_dict=lambda: {
                "model": runtime.state_dict(),
                "optimizer": None,
                "rng": None,
            },
            optimizer_state_is_replicated=True,
        )

        def handed(model):
            """Shapes of the tensors the objective model's forward is handed."""
            with mock.patch.object(runtime, "forward") as forward, torch.no_grad():
                forward_program(model, batch)
            (tensors,) = forward.call_args.args
            return {name: value.shape for name, value in tensors.items()}

        plain = build_runner(runtime, backend)
        with TemporaryDirectory() as root:
            dist.init_process_group(
                "gloo", init_method=f"file://{root}/store", rank=0, world_size=1
            )
            try:
                wrapped = DDP(runtime)
                runner = build_runner(wrapped, backend)
                self.assertEqual(runner.spec, plain.spec)
                through_ddp = handed(wrapped)
                self.assertIn("anchor_capacity", through_ddp)
                self.assertEqual(through_ddp, handed(runtime))
                torch.testing.assert_close(
                    runner.checkpoint_state().draft, plain.checkpoint_state().draft
                )
                del runner, wrapped
                gc.collect()
            finally:
                dist.destroy_process_group()
