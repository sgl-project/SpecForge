# coding=utf-8
"""Numerical parity across the canonical colocated/disaggregated builders.

Both runs use the same offline tensors, initial model/head state, optimizer and
public ``Trainer.fit()`` lifecycle.  Only the data plane changes: file-backed
local refs versus producer-ingested ``disagg://`` refs.
"""

import os
import tempfile
import unittest

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry
from tests.test_runtime import _fixtures as fx

CUDA = torch.cuda.is_available()
ALGORITHM = builtin_algorithm_registry().resolve("eagle3")


def _optimizer_factory(module):
    from specforge.optimizer import BF16Optimizer

    return BF16Optimizer(
        module,
        lr=1e-3,
        max_grad_norm=0.5,
        warmup_ratio=0.0,
        total_steps=2,
    )


def _draft_state(trainer):
    state = trainer.backend.state_dict()["model"]
    return trainer.core.strategy.checkpoint_state_filter(state)


@unittest.skipUnless(CUDA, "colocated/disaggregated parity requires CUDA")
class TestColocatedVsDisaggEquiv(unittest.TestCase):
    def test_current_builders_produce_identical_losses_and_weights(self):
        fx.build_single_rank_distributed(port="29571")
        from specforge.launch import build_disagg_offline_runtime, build_offline_runtime
        from specforge.runtime.data_plane.disagg_ingest import ingest_offline_features
        from specforge.runtime.data_plane.disaggregated import SharedDirFeatureStore

        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            with tempfile.TemporaryDirectory(prefix="colocated_disagg_") as work:
                feature_dir = fx.write_offline_files(
                    os.path.join(work, "features"), n=2, seq=16, seed=13
                )
                local_model_dir = os.path.join(work, "local-model")
                disagg_model_dir = os.path.join(work, "disagg-model")
                os.makedirs(local_model_dir)
                os.makedirs(disagg_model_dir)

                torch.manual_seed(0)
                local_model, local_head = fx.build_eagle3(local_model_dir, ttt=3)
                disagg_model, disagg_head = fx.build_eagle3(disagg_model_dir, ttt=3)
                disagg_model.load_state_dict(local_model.state_dict())
                disagg_head.load_state_dict(local_head.state_dict())

                shared_store = SharedDirFeatureStore(
                    os.path.join(work, "shared"),
                    store_id="parity",
                    retain_on_release=True,
                )
                disagg_refs = ingest_offline_features(
                    shared_store,
                    feature_dir,
                    run_id="parity",
                    ttt_length=3,
                    max_len=512,
                )

                local_losses = []
                local_trainer = build_offline_runtime(
                    algorithm=ALGORITHM,
                    hidden_states_path=feature_dir,
                    draft_model=local_model,
                    target_head=local_head,
                    optimizer_factory=_optimizer_factory,
                    run_id="local",
                    output_dir=os.path.join(work, "local-output"),
                    ttt_length=3,
                    max_len=512,
                    batch_size=1,
                    max_steps=2,
                    logger=lambda metrics, step: local_losses.append(
                        (step, metrics["loss"])
                    ),
                    log_interval=1,
                )
                disagg_losses = []
                disagg_trainer = build_disagg_offline_runtime(
                    algorithm=ALGORITHM,
                    feature_store=shared_store,
                    refs=disagg_refs,
                    draft_model=disagg_model,
                    target_head=disagg_head,
                    optimizer_factory=_optimizer_factory,
                    run_id="disaggregated",
                    output_dir=os.path.join(work, "disagg-output"),
                    ttt_length=3,
                    max_len=512,
                    batch_size=1,
                    max_steps=2,
                    logger=lambda metrics, step: disagg_losses.append(
                        (step, metrics["loss"])
                    ),
                    log_interval=1,
                )

                self.assertEqual(local_trainer.fit(), 2)
                local_state = _draft_state(local_trainer)
                self.assertEqual(disagg_trainer.fit(), 2)
                disagg_state = _draft_state(disagg_trainer)

                self.assertEqual(local_losses, disagg_losses)
                self.assertEqual(set(local_state), set(disagg_state))
                for name in sorted(local_state):
                    self.assertTrue(
                        torch.equal(local_state[name], disagg_state[name]),
                        f"draft weight {name!r} diverged across data planes",
                    )
        finally:
            torch.use_deterministic_algorithms(
                previous_deterministic, warn_only=previous_warn_only
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
