# coding=utf-8
"""Online EAGLE3 capture-to-loss parity through the canonical runtime.

The reference uses the policy-driven target capture and registered strategy
directly.  The compared path sends the same prompt through the current rollout
stream, feature store, builder and public ``Trainer.fit()`` lifecycle.
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


@unittest.skipUnless(CUDA, "online EAGLE3 parity requires CUDA")
class TestEquivOnlineEagle3(unittest.TestCase):
    def test_policy_capture_matches_rollout_stream_and_trainer_fit(self):
        fx.build_single_rank_distributed(port="29565")
        from specforge.inference.adapters.policy import (
            EAGLE3_FEATURE_SCHEMA,
            PolicyFeatureAdapter,
        )
        from specforge.inference.capture import CaptureConfig
        from specforge.launch import build_online_runtime
        from specforge.runtime.contracts import PromptTask, TrainBatch
        from specforge.training.strategies.registry import resolve_strategy

        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            with tempfile.TemporaryDirectory(prefix="equiv_online_") as work:
                torch.manual_seed(0)
                target, _target_dir, aux_layer_ids = fx.build_hf_target(
                    work, hidden=fx.H, layers=8, vocab=fx.V
                )
                model, _target_head = fx.build_eagle3(work, ttt=3)
                model.eval()

                generator = torch.Generator().manual_seed(11)
                input_ids = torch.randint(0, fx.V, (12,), generator=generator).tolist()
                loss_mask = [1] * len(input_ids)
                task = PromptTask(
                    task_id="prompt-0",
                    run_id="online-reference",
                    source_id="synthetic",
                    payload={"input_ids": input_ids, "loss_mask": loss_mask},
                    max_length=len(input_ids),
                )
                capture = CaptureConfig.from_strategy(
                    required_features=EAGLE3_FEATURE_SCHEMA.names,
                    aux_hidden_state_layer_ids=tuple(aux_layer_ids),
                    target_repr="logits",
                    target_hidden_size=fx.H,
                    target_vocab_size=fx.V,
                    draft_vocab_size=fx.D,
                )
                adapter = PolicyFeatureAdapter(
                    target, schema=EAGLE3_FEATURE_SCHEMA, device="cuda"
                )
                features = adapter.generate_features([task], capture=capture)
                # RolloutWorker verifies and removes this out-of-band capture
                # record before FeatureStore.put; mirror that exact boundary.
                for feature in features:
                    self.assertEqual(
                        feature.pop("__aux_layer_ids__"), tuple(aux_layer_ids)
                    )
                spec = resolve_strategy("eagle3")
                tensors = spec.make_online_collate()(features)
                direct_batch = TrainBatch(
                    sample_ids=[task.task_id],
                    strategy=spec.name,
                    tensors=tensors,
                    metadata={"target_repr": "logits", "ttt_length": 3},
                )
                with torch.no_grad():
                    expected = float(
                        spec.make_strategy(model, target_head=None)
                        .forward_loss(direct_batch)
                        .loss.item()
                    )

                logged = []
                trainer = build_online_runtime(
                    strategy="eagle3",
                    target_model=target,
                    prompts=[
                        {
                            "task_id": task.task_id,
                            "payload": {
                                "input_ids": input_ids,
                                "loss_mask": loss_mask,
                            },
                        }
                    ],
                    draft_model=model,
                    optimizer_factory=_optimizer_factory,
                    run_id="online-canonical",
                    output_dir=os.path.join(work, "output"),
                    target_hidden_size=fx.H,
                    target_vocab_size=fx.V,
                    draft_vocab_size=fx.D,
                    target_repr="logits",
                    aux_hidden_state_layer_ids=aux_layer_ids,
                    device="cuda",
                    batch_size=1,
                    max_steps=1,
                    dataset_size=1,
                    logger=lambda metrics, step: logged.append((step, metrics["loss"])),
                    log_interval=1,
                )

                self.assertEqual(trainer.fit(), 1)
                self.assertEqual([step for step, _ in logged], [1])
                self.assertAlmostEqual(expected, logged[0][1], places=3)
        finally:
            torch.use_deterministic_algorithms(
                previous_deterministic, warn_only=previous_warn_only
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
