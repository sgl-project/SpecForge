# coding=utf-8
"""Offline EAGLE3 feature-to-loss parity through the canonical runtime.

The reference computes the objective directly from one normalized ``TrainBatch``.
The compared path owns the same feature files through ``build_offline_runtime``
and the public no-argument ``Trainer.fit()`` lifecycle.  No legacy dataset or
training entrypoint participates in this gate.
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
        lr=1e-3,
        max_grad_norm=0.5,
        warmup_ratio=0.0,
        total_steps=1,
    )


@torch.no_grad()
def _direct_offline_loss(model, target_head, batch) -> float:
    """Compute the EAGLE3 objective below the strategy/runtime boundary."""
    tensors = batch.tensors
    input_ids, target, loss_mask = target_head.preprocess(
        tensors["input_ids"], tensors["target"], tensors["loss_mask"]
    )
    target = target_head(target.cuda())
    plosses, *_ = model(
        input_ids=input_ids.cuda(),
        attention_mask=tensors["attention_mask"].cuda(),
        loss_mask=loss_mask.cuda(),
        target=target,
        hidden_states=tensors["hidden_state"].cuda(),
    )
    return float(sum(0.8**index * loss for index, loss in enumerate(plosses)).item())


@unittest.skipUnless(CUDA, "offline EAGLE3 parity requires CUDA")
class TestEquivOfflineEagle3(unittest.TestCase):
    def test_feature_files_match_direct_objective_through_trainer_fit(self):
        fx.build_single_rank_distributed(port="29555")
        from specforge.launch import build_offline_runtime

        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            with tempfile.TemporaryDirectory(prefix="equiv_offline_") as work:
                feature_dir = fx.write_offline_files(
                    os.path.join(work, "features"), n=2, seq=16, seed=7
                )
                torch.manual_seed(0)
                model, target_head = fx.build_eagle3(work, ttt=3)
                model.eval()
                reference_batch = next(
                    iter(
                        fx.build_offline_eagle3_loader(
                            feature_dir,
                            batch_size=2,
                            run_id="offline-reference",
                            ttt_length=3,
                            max_len=512,
                        )
                    )
                )
                expected = _direct_offline_loss(model, target_head, reference_batch)

                logged = []
                trainer = build_offline_runtime(
                    strategy="eagle3",
                    hidden_states_path=feature_dir,
                    draft_model=model,
                    target_head=target_head,
                    optimizer_factory=_optimizer_factory,
                    run_id="offline-canonical",
                    output_dir=os.path.join(work, "output"),
                    ttt_length=3,
                    max_len=512,
                    batch_size=2,
                    max_steps=1,
                    logger=lambda metrics, step: logged.append((step, metrics["loss"])),
                    log_interval=1,
                )

                self.assertEqual(trainer.fit(), 1)
                self.assertEqual([step for step, _ in logged], [1])
                self.assertEqual(expected, logged[0][1])
        finally:
            torch.use_deterministic_algorithms(
                previous_deterministic, warn_only=previous_warn_only
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
