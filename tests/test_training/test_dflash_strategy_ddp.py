"""The canonical objective owns loss options under DDP and keeps the wrapped forward."""

import tempfile
import unittest

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from specforge.algorithms.objectives import build_objective_model
from specforge.modeling import build_draft_model
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.canonical import CanonicalTrainStrategy
from tests.test_algorithms.test_objectives import configurations, training_batch


class CanonicalDDPTest(unittest.TestCase):
    def test_selector_warmup_and_gradient_through_wrapper(self):
        with tempfile.TemporaryDirectory() as path:
            torch.distributed.init_process_group(
                "gloo", init_method=f"file://{path}/group", rank=0, world_size=1
            )
            try:
                config = configurations()[1]
                model = build_objective_model(
                    build_draft_model(config),
                    options=dict(
                        num_anchors=2,
                        teacher_metrics=False,
                        fused_logits_loss=False,
                        selector_warmup_steps=3,
                    ),
                )
                wrapped = DDP(model)
                strategy = CanonicalTrainStrategy(
                    wrapped,
                    name="dflash",
                    objective="dflash2",
                    required_features=model.registration.inputs,
                )
                self.assertIs(strategy.model, wrapped)
                for _ in range(2):
                    wrapped.zero_grad(set_to_none=True)
                    output = strategy.forward_loss(
                        TrainBatch(
                            sample_ids=["sample"],
                            strategy="dflash",
                            tensors=training_batch(config),
                        )
                    )
                    output.loss.backward()
                    for name, parameter in model.named_parameters():
                        self.assertEqual(
                            parameter.grad is not None, parameter.requires_grad, name
                        )
            finally:
                torch.distributed.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
