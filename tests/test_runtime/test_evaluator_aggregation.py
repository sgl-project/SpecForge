# coding=utf-8
"""Phase D: Evaluator per-position aggregation + CheckpointManager rotation/pointers.

The evaluator correctness that scalar-averaging gets wrong: per-position accuracy
must be aggregated over the whole eval pass (sum corrects / sum denoms) BEFORE the
geometric sum, so ``simulated_acc_len`` does not depend on the eval batch size. No
CUDA required (fabricated per-step metrics); run anywhere torch imports.
"""

import os
import tempfile
import unittest

import torch

from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import StepOutput


def _batch():
    return TrainBatch(
        sample_ids=["x"],
        strategy="eagle3",
        tensors={"loss_mask": torch.ones(4, dtype=torch.long)},
        metadata={},
    )


def _step_output(loss, corrects, denoms):
    t = lambda xs: [torch.tensor(float(x)) for x in xs]  # noqa: E731
    return StepOutput(
        loss=torch.tensor(float(loss)),
        metrics={
            "acc_corrects": t(corrects),
            "acc_denoms": t(denoms),
            "metric_loss_denoms": t(denoms),
        },
    )


class TestEvaluatorAggregation(unittest.TestCase):
    def _run(self, outputs):
        from specforge.eval import Evaluator

        it = iter(outputs)
        return Evaluator().run(lambda b: next(it), [_batch() for _ in outputs])

    def test_aggregates_counts_before_geometric_sum(self):
        # pos0: (3+1)/(4+1)=0.8 ; pos1: (2+0)/(4+1)=0.4  -- NOT the mean of ratios.
        m = self._run(
            [
                _step_output(2.0, corrects=[3, 2], denoms=[4, 4]),
                _step_output(8.0, corrects=[1, 0], denoms=[1, 1]),
            ]
        )
        self.assertAlmostEqual(m["eval/avg_acc"], 0.8, places=6)
        # simulated_acc_len = 0.8 + 0.8*0.4 = 1.12
        self.assertAlmostEqual(m["eval/simulated_acc_len"], 1.12, places=6)
        # token-weighted loss: (2*8 + 8*2) / (8+2) = 3.2
        self.assertAlmostEqual(m["eval/avg_loss"], 3.2, places=6)

    def test_naive_average_would_differ(self):
        # The mean of per-batch pos0 accuracy is (0.75 + 1.0)/2 = 0.875; the correct
        # count-aggregated value is 0.8. Guard against a regression to the mean.
        m = self._run(
            [
                _step_output(1.0, corrects=[3], denoms=[4]),
                _step_output(1.0, corrects=[1], denoms=[1]),
            ]
        )
        self.assertAlmostEqual(m["eval/avg_acc"], 0.8, places=6)
        self.assertNotAlmostEqual(m["eval/avg_acc"], 0.875, places=3)

    def test_batch_size_invariance(self):
        split = self._run(
            [
                _step_output(1.0, corrects=[3, 2], denoms=[4, 4]),
                _step_output(1.0, corrects=[1, 0], denoms=[1, 1]),
            ]
        )
        combined = self._run([_step_output(1.0, corrects=[4, 2], denoms=[5, 5])])
        self.assertAlmostEqual(
            split["eval/simulated_acc_len"],
            combined["eval/simulated_acc_len"],
            places=6,
        )
        self.assertAlmostEqual(
            split["eval/avg_acc"], combined["eval/avg_acc"], places=6
        )

    def test_scalar_strategy_degenerates_gracefully(self):
        from specforge.eval import Evaluator

        outs = [
            StepOutput(loss=torch.tensor(1.0), metrics={"accuracy": torch.tensor(0.5)}),
            StepOutput(loss=torch.tensor(1.0), metrics={"accuracy": torch.tensor(0.7)}),
        ]
        it = iter(outs)
        m = Evaluator().run(lambda b: next(it), [_batch(), _batch()])
        self.assertAlmostEqual(m["eval/avg_acc"], 0.6, places=6)
        self.assertAlmostEqual(m["eval/simulated_acc_len"], 0.6, places=6)


class TestCheckpointManager(unittest.TestCase):
    def _state(self, step):
        return {"draft_state_dict": {"w": torch.zeros(2)}, "global_step": step}

    def test_rotation_pointers_and_best(self):
        from specforge.training.checkpoint import CheckpointManager

        out = tempfile.mkdtemp(prefix="ckpt_mgr_")
        mgr = CheckpointManager(out, "run", max_checkpoints=2)

        mgr.save(self._state(1), 1)
        mgr.save(self._state(2), 2)
        mgr.update_best(2, {"eval/simulated_acc_len": 5.0})
        mgr.save(self._state(3), 3)  # rotate: drop step1
        mgr.save(self._state(4), 4)  # rotate: step2 would drop but is best -> kept

        def ck(step):
            return mgr.checkpoint_dir(step)

        self.assertFalse(os.path.exists(ck(1)), "oldest non-best should rotate away")
        self.assertTrue(os.path.exists(ck(2)), "best checkpoint must be protected")
        self.assertTrue(os.path.exists(ck(3)))
        self.assertTrue(os.path.exists(ck(4)))

        # latest -> step4, best -> step2
        self.assertEqual(os.path.realpath(mgr.latest_dir()), os.path.realpath(ck(4)))
        self.assertEqual(
            os.path.realpath(os.path.join(out, "best")), os.path.realpath(ck(2))
        )

        loaded = mgr.load(step=2)
        self.assertEqual(loaded["global_step"], 2)
        self.assertEqual(mgr.load()["global_step"], 4)  # latest


if __name__ == "__main__":
    unittest.main(verbosity=2)
