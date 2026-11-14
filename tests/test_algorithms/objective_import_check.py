"""Check the eager objective path without Triton or Liger installed."""

import importlib.abc
import sys

import torch


class NoKernels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in ("triton", "liger_kernel"):
            raise ModuleNotFoundError(fullname)


sys.meta_path.insert(0, NoKernels())
from specforge.core import fused_loss as legacy_losses  # noqa: E402
from specforge.losses import fused as shared_losses  # noqa: E402

assert legacy_losses is shared_losses
assert "specforge.algorithms.objectives" not in sys.modules
assert "transformers" not in sys.modules

from specforge.algorithms.objectives import build_objective_model  # noqa: E402
from specforge.modeling import build_draft_model  # noqa: E402
from tests.test_config.draft_fixtures import draft_config  # noqa: E402

config = draft_config(target_bindings=(), target_logit_transform=None)
model = build_draft_model(config)
runtime = build_objective_model(model, options={"num_anchors": 1})
result = runtime(
    dict(
        input_ids=torch.arange(8)[None],
        loss_mask=torch.ones(1, 8),
        hidden_states=torch.randn(
            1, 8, config.spec_config.target_features.input_width()
        ),
    ),
    step=0,
    total_steps=None,
)
result.loss.backward()
assert "specforge.core.loss" not in sys.modules
assert "specforge.algorithms.peagle.model" not in sys.modules

from specforge.algorithms.objectives.implementations import fused_loss  # noqa: E402

assert fused_loss is shared_losses
