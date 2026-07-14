# coding=utf-8
"""Trainer-boundary numerical parity using only current training APIs.

One model takes a step through the offline builder and public ``Trainer.fit()``.
An identically initialized model takes the same step through the canonical
``TrainerCore``/``FSDPTrainingBackend`` seam.  Loss, gradient norm and updated
draft weights must agree.
"""

import os
import tempfile
import unittest

import torch

from tests.test_runtime import _fixtures as fx

CUDA = torch.cuda.is_available()


def _optimizer_factory(module):
    from specforge.optimizer import BF16Optimizer

    return BF16Optimizer(
        module,
        lr=1e-4,
        max_grad_norm=0.5,
        warmup_ratio=0.0,
        total_steps=1,
    )


@unittest.skipUnless(CUDA, "trainer-boundary parity requires CUDA")
class TestEquivTrainerSplit(unittest.TestCase):
    def test_trainer_fit_matches_current_core_step(self):
        fx.build_single_rank_distributed(port="29562")
        from specforge.launch import build_offline_runtime
        from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
        from specforge.training.controller import TrainerCore
        from specforge.training.strategies.registry import resolve_strategy

        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            with tempfile.TemporaryDirectory(prefix="equiv_trainer_") as work:
                feature_dir = fx.write_offline_files(
                    os.path.join(work, "features"), n=1, seq=16, seed=19
                )
                fit_dir = os.path.join(work, "fit-model")
                core_dir = os.path.join(work, "core-model")
                os.makedirs(fit_dir)
                os.makedirs(core_dir)

                torch.manual_seed(0)
                fit_model, fit_head = fx.build_eagle3(fit_dir, ttt=3)
                core_model, core_head = fx.build_eagle3(core_dir, ttt=3)
                core_model.load_state_dict(fit_model.state_dict())
                core_head.load_state_dict(fit_head.state_dict())

                logged = []
                trainer = build_offline_runtime(
                    strategy="eagle3",
                    hidden_states_path=feature_dir,
                    draft_model=fit_model,
                    target_head=fit_head,
                    optimizer_factory=_optimizer_factory,
                    run_id="fit-path",
                    output_dir=os.path.join(work, "fit-output"),
                    ttt_length=3,
                    max_len=512,
                    batch_size=1,
                    max_steps=1,
                    logger=lambda metrics, step: logged.append((step, dict(metrics))),
                    log_interval=1,
                )

                backend = FSDPTrainingBackend(
                    ParallelConfig.from_distributed(),
                    optimizer_factory=_optimizer_factory,
                )
                wrapped = backend.prepare_model(
                    core_model,
                    wrap=True,
                    optimizer_target=core_model.draft_model,
                )
                spec = resolve_strategy("eagle3")
                core = TrainerCore(
                    spec.make_strategy(wrapped, target_head=core_head),
                    backend,
                    accumulation_steps=1,
                )
                batch = next(
                    iter(
                        fx.build_offline_eagle3_loader(
                            feature_dir,
                            batch_size=1,
                            run_id="core-data",
                            ttt_length=3,
                            max_len=512,
                        )
                    )
                )

                self.assertEqual(trainer.fit(), 1)
                core_result = core.train_step(batch)
                self.assertTrue(core_result.optimizer_stepped)
                self.assertEqual([step for step, _ in logged], [1])
                self.assertEqual(logged[0][1]["loss"], core_result.loss)
                self.assertAlmostEqual(
                    logged[0][1]["grad_norm"], core_result.grad_norm, places=3
                )

                fit_state = trainer.core.strategy.checkpoint_state_filter(
                    trainer.backend.state_dict()["model"]
                )
                core_state = core.strategy.checkpoint_state_filter(
                    backend.state_dict()["model"]
                )
                self.assertEqual(set(fit_state), set(core_state))
                for name in sorted(fit_state):
                    self.assertTrue(
                        torch.equal(fit_state[name], core_state[name]),
                        f"draft weight {name!r} changed at the Trainer boundary",
                    )
        finally:
            torch.use_deterministic_algorithms(
                previous_deterministic, warn_only=previous_warn_only
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
