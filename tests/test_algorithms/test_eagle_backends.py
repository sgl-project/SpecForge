"""Canonical EAGLE3 runs a configured objective step on each declared backend."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from specforge.algorithms.objectives import build_objective_model
from specforge.algorithms.objectives.registry import (
    protocol_registrations,
    resolve_objective,
)
from specforge.modeling import build_draft_model
from tests.test_algorithms.test_objectives import eagle_config, training_batch
from tests.test_modeling.test_canonical_eagle3 import install_mapping

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def objective_model(backend, **selection):
    config = eagle_config()
    torch.manual_seed(0)
    model = build_draft_model(config, backend=backend, device=DEVICE)
    install_mapping(model)
    torch.manual_seed(1)
    head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).to(DEVICE)
    runtime = build_objective_model(model, target_head=head, **selection)
    runtime.attention_backend = backend  # set by build_bundle in a real run
    return config, runtime


def forward_loss(config, runtime):
    torch.manual_seed(2)
    batch = training_batch(config, device=DEVICE)
    with torch.no_grad():
        return runtime(batch, step=0, total_steps=1).loss


class EagleBackendTest(unittest.TestCase):
    def test_dense_backends_agree_through_the_objective(self):
        declared = protocol_registrations()["eagle3"].training.attention_backends
        self.assertEqual(declared, {"eager", "sdpa", "flex_attention", "fa", "usp"})
        # Compiled FlexAttention only runs on CUDA.
        backends = ["eager", "sdpa"] + ["flex_attention"] * torch.cuda.is_available()
        losses = {
            backend: forward_loss(*objective_model(backend)) for backend in backends
        }
        for backend in backends[1:]:
            with self.subTest(backend=backend):
                torch.testing.assert_close(
                    losses[backend], losses["eager"], rtol=1e-4, atol=1e-5
                )

    def test_flex_runs_through_the_cached_rollout(self):
        # The uncached branch builds a tensor visibility without active
        # queries, which Flex preparation rejects; the rollout supplies both.
        config, runtime = objective_model("sdpa")
        runtime.attention_backend = "flex_attention"
        model = runtime.draft_model
        original, requested = model.new_rollout, []

        def new_rollout(backend, **groups):
            requested.append(backend)
            return original("sdpa", **groups)  # stands in for Flex off CUDA

        with patch.object(model, "new_rollout", new_rollout):
            loss = forward_loss(config, runtime)
        self.assertEqual(requested, ["flex_attention"])
        torch.testing.assert_close(
            loss, forward_loss(*objective_model("sdpa")), rtol=1e-5, atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
