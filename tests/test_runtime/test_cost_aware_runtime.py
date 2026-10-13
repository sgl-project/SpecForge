# coding=utf-8
"""Cost-aware scheduling through the offline trainer runtime (CUDA, 1 rank):
DSpark checkpoint/resume contract and EAGLE3 at batch size one."""

import os
import tempfile
import unittest

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.runtime.data_plane.offline_index import INDEX_FILENAME
from specforge.runtime.data_plane.window_plan import SchedulingPolicy

CUDA = torch.cuda.is_available()


def _optimizer_factory(total_steps):
    from specforge.optimizer import BF16Optimizer

    return lambda module: BF16Optimizer(
        module,
        lr=1e-3,
        max_grad_norm=0.5,
        warmup_ratio=0.0,
        total_steps=total_steps,
    )


@unittest.skipUnless(CUDA, "the offline launcher fixtures build on CUDA")
class TestCostAwareOfflineRuntime(unittest.TestCase):
    def test_dspark_resume_requires_the_same_scheduling_inputs(self):
        from specforge.launch import build_offline_runtime
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29583")
        algorithm = builtin_algorithm_registry().resolve("dspark")
        hidden, seq = 64, 12
        workdir = tempfile.mkdtemp(prefix="las_resume_")
        _, width = fx.build_dspark(
            os.path.join(workdir, "m0"),
            hidden=hidden,
            block_size=4,
            num_anchors=2,
            attention_backend="sdpa",
        )
        feature_dir = fx.write_offline_files_dspark(
            os.path.join(workdir, "features"),
            n=8,
            seq=seq,
            captured_width=width,
            target_hidden=hidden,
        )
        output_dir = os.path.join(workdir, "out")
        builds = [0]

        def build(**overrides):
            builds[0] += 1
            model, _ = fx.build_dspark(
                os.path.join(workdir, f"m{builds[0]}"),
                hidden=hidden,
                block_size=4,
                num_anchors=2,
                attention_backend="sdpa",
            )
            # Resume through the direct builder needs the provider-bound step
            # runtime (algorithm options + resume contract), as the CLI passes.
            runtime = algorithm.providers.step.bind_runtime(
                None, model.draft_model, model, model_provenance={"fixture": "dspark"}
            )
            kwargs = dict(
                algorithm=algorithm,
                hidden_states_path=feature_dir,
                draft_model=model,
                target_head=None,
                strategy_kwargs=runtime,
                optimizer_factory=_optimizer_factory(4),
                run_id="las",
                output_dir=output_dir,
                max_len=seq,
                batch_size=2,
                accumulation_steps=2,
                num_epochs=2,
                max_steps=4,
                total_steps=4,
                save_interval=1,
                scheduling=SchedulingPolicy("anchors", num_anchors=2),
            )
            kwargs.update(overrides)
            return build_offline_runtime(**kwargs)

        trainer = build(max_steps=2)
        self.assertEqual(trainer.fit(), 2)
        checkpoint = os.path.join(output_dir, "las-step2")
        self.assertTrue(os.path.isdir(checkpoint))
        self.assertTrue(os.path.exists(os.path.join(output_dir, INDEX_FILENAME)))

        # Same inputs: the resume is accepted and finishes the remaining steps
        # (``fit`` returns the final global step).
        self.assertEqual(build(resume_from=checkpoint).fit(), 4)
        self.assertTrue(os.path.isdir(os.path.join(output_dir, "las-step4")))
        # A different max_len changes the costs: rejected before training.
        with self.assertRaisesRegex(ValueError, "cost_aware_scheduling"):
            build(resume_from=checkpoint, max_len=seq - 1)
        # Scheduling switched off would reorder the samples: rejected too.
        with self.assertRaisesRegex(ValueError, "cost_aware_scheduling"):
            build(resume_from=checkpoint, scheduling=None)

    def test_eagle3_at_batch_size_one_trains_with_the_length_key(self):
        from specforge.launch import build_offline_runtime
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29584")
        algorithm = builtin_algorithm_registry().resolve("eagle3")
        workdir = tempfile.mkdtemp(prefix="las_eagle3_")
        model, target_head = fx.build_eagle3(workdir, ttt=3)
        feature_dir = fx.write_offline_files(
            os.path.join(workdir, "features"), n=6, seq=16
        )
        trainer = build_offline_runtime(
            algorithm=algorithm,
            hidden_states_path=feature_dir,
            draft_model=model,
            target_head=target_head,
            optimizer_factory=_optimizer_factory(3),
            run_id="las-eagle3",
            output_dir=os.path.join(workdir, "out"),
            ttt_length=3,
            max_len=16,
            batch_size=1,
            accumulation_steps=2,
            num_epochs=1,
            max_steps=3,
            total_steps=3,
            scheduling=SchedulingPolicy("length"),
        )
        self.assertEqual(trainer.fit(), 3)


if __name__ == "__main__":
    unittest.main()
