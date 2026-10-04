"""Packed offline loader -> FSDP -> optimizer/eval/checkpoint GPU smoke test."""

import math
import numbers
import tempfile
import unittest
from pathlib import Path

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.eagle3.data import DataCollatorWithPacking
from tests.test_runtime import _fixtures as fx


def _write_variable_length_features(directory, lengths):
    directory.mkdir()
    generator = torch.Generator().manual_seed(17)
    for index, length in enumerate(lengths):
        torch.save(
            {
                "input_ids": torch.randint(0, fx.V, (length,), generator=generator),
                "loss_mask": torch.ones(length, dtype=torch.long),
                "hidden_state": torch.randn(
                    1, length, fx.H, generator=generator
                ).bfloat16(),
                "aux_hidden_state": torch.randn(
                    1, length, 3 * fx.H, generator=generator
                ).bfloat16(),
            },
            directory / f"{index:04d}.ckpt",
        )
    return str(directory)


@unittest.skipUnless(
    torch.cuda.is_available(), "packed trainer lifecycle requires CUDA"
)
class SequencePackingLifecycleTest(unittest.TestCase):
    def test_packed_training_preserves_steps_evaluation_and_checkpoints(self):
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge.launch import build_offline_runtime
        from specforge.optimizer import BF16Optimizer
        from specforge.training.checkpoint import STATE_FILE

        torch.manual_seed(17)
        fx.build_single_rank_distributed(port="29687")
        logged = []
        with tempfile.TemporaryDirectory(prefix="packed_lifecycle_") as workdir:
            work = Path(workdir)
            train_path = _write_variable_length_features(
                work / "train", (8, 24, 12, 16) * 2
            )
            eval_path = _write_variable_length_features(work / "eval", (8, 24, 12))
            model, target_head = fx.build_eagle3(workdir, ttt=3)
            output = work / "output"

            def optimizer_factory(draft_module):
                return BF16Optimizer(
                    draft_module,
                    lr=1e-3,
                    max_grad_norm=0.5,
                    warmup_ratio=0.0,
                    total_steps=2,
                )

            trainer = build_offline_runtime(
                algorithm=builtin_algorithm_registry().resolve("eagle3"),
                hidden_states_path=train_path,
                eval_hidden_states_path=eval_path,
                draft_model=model,
                target_head=target_head,
                optimizer_factory=optimizer_factory,
                run_id="packing-lifecycle",
                output_dir=str(output),
                ttt_length=3,
                max_len=32,
                batch_size=2,
                sequence_packing=True,
                accumulation_steps=2,
                num_epochs=1,
                max_steps=2,
                eval_interval=1,
                save_interval=1,
                log_interval=1,
                logger=lambda metrics, step: logged.append((dict(metrics), step)),
            )
            self.assertIsInstance(trainer.core.strategy.trainable_module(), FSDP)
            self.assertIsInstance(trainer._loader.collate_fn, DataCollatorWithPacking)
            eval_loader = trainer._controller.eval_data_factory()
            self.assertIsInstance(eval_loader.collate_fn, DataCollatorWithPacking)
            self.assertEqual([len(b.sample_ids) for b in eval_loader], [2, 1])

            self.assertEqual(trainer.fit(), 2)
            self.assertEqual(trainer.global_step, 2)
            self.assertEqual(trainer.micro_step, 4)
            self.assertEqual(trainer.last_checkpoint_step, 2)
            self.assertEqual(
                [step for metrics, step in logged if "eval/avg_loss" in metrics], [1, 2]
            )
            self.assertTrue(any("loss" in metrics for metrics, _ in logged))
            for metrics, _ in logged:
                for name, value in metrics.items():
                    if isinstance(value, numbers.Real):
                        self.assertTrue(math.isfinite(value), f"{name}={value}")
            for step in (1, 2):
                checkpoint = output / f"packing-lifecycle-step{step}" / STATE_FILE
                self.assertTrue(checkpoint.is_file())
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                self.assertEqual(state["global_step"], step)
                self.assertEqual(state["epoch_samples"], 4 * step)
                self.assertTrue(state["draft_state_dict"])
            self.assertTrue((output / "packing-lifecycle-latest").is_dir())


if __name__ == "__main__":
    unittest.main()
