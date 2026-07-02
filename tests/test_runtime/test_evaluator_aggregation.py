# coding=utf-8
"""Phase D: Evaluator per-position aggregation + CheckpointManager rotation/pointers.

The evaluator correctness that scalar-averaging gets wrong: per-position accuracy
must be aggregated over the whole eval pass (sum corrects / sum denoms) BEFORE the
geometric sum, so ``simulated_acc_len`` does not depend on the eval batch size.
The DP gate (2-process gloo) additionally pins the rank axis: scalar accuracy is
reduced across ranks, and a rank with an EMPTY eval shard issues the same
collectives as its peers (a ragged shard must not desynchronize the group). No
CUDA required (fabricated per-step metrics); run anywhere torch imports.
"""

import json
import os
import tempfile
import unittest
from datetime import timedelta

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

        # equal token counts (loss_mask of 4 each) -> the token-weighted
        # aggregate equals the plain mean here.
        outs = [
            StepOutput(loss=torch.tensor(1.0), metrics={"accuracy": torch.tensor(0.5)}),
            StepOutput(loss=torch.tensor(1.0), metrics={"accuracy": torch.tensor(0.7)}),
        ]
        it = iter(outs)
        m = Evaluator().run(lambda b: next(it), [_batch(), _batch()])
        self.assertAlmostEqual(m["eval/avg_acc"], 0.6, places=6)
        self.assertAlmostEqual(m["eval/simulated_acc_len"], 0.6, places=6)

    def test_scalar_accuracy_is_batch_size_invariant(self):
        from specforge.eval import Evaluator

        # A batch's scalar accuracy is a per-token mean, so the aggregate must
        # weight by token count: (3-of-4) + (1-of-2) regrouped as one 4-of-6
        # batch must give the same number — a plain mean of per-batch means
        # would not (the ragged batch would skew it: (0.75+0.5)/2 = 0.625).
        split = [
            _scalar_out(1.0, 3 / 4, tokens=4),
            _scalar_out(1.0, 1 / 2, tokens=2),
        ]
        combined = [_scalar_out(1.0, 4 / 6, tokens=6)]
        for outs, n in ((split, 2), (combined, 1)):
            it = iter(outs)
            m = Evaluator().run(lambda b: next(it), [_batch() for _ in range(n)])
            self.assertAlmostEqual(m["eval/avg_acc"], 4 / 6, places=6)
            self.assertAlmostEqual(m["eval/simulated_acc_len"], 4 / 6, places=6)

    def test_reports_per_position_acceptance(self):
        # The roadmap asks for per-position acceptance, not just the folded sum.
        m = self._run([_step_output(1.0, corrects=[3, 2], denoms=[4, 4])])
        self.assertAlmostEqual(m["eval/per_position_acc"][0], 0.75, places=6)
        self.assertAlmostEqual(m["eval/per_position_acc"][1], 0.5, places=6)
        self.assertEqual(len(m["eval/per_position_acc"]), 2)


def _scalar_out(loss, acc, tokens):
    return StepOutput(
        loss=torch.tensor(float(loss)),
        metrics={
            "accuracy": torch.tensor(float(acc)),
            "metric_loss_denoms": [torch.tensor(float(tokens))],
        },
    )


def _dp_worker(rank, world_size, port, results_dir):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    from specforge.eval import Evaluator

    # Scenario 1 — scalar accuracy, uneven shards: every sum must be reduced.
    if rank == 0:
        outs = [_scalar_out(2.0, 0.5, tokens=8)]
    else:
        outs = [_scalar_out(8.0, 0.7, tokens=2), _scalar_out(4.0, 0.9, tokens=10)]
    it = iter(outs)
    scalar = Evaluator().run(lambda b: next(it), [_batch() for _ in outs])

    # Scenario 2 — ragged per-position shards: rank 1's shard is EMPTY, so the
    # collective schedule must be decided globally or the group desynchronizes.
    outs2 = [_step_output(1.0, corrects=[3, 2], denoms=[4, 4])] if rank == 0 else []
    it2 = iter(outs2)
    ragged = Evaluator().run(lambda b: next(it2), [_batch() for _ in outs2])

    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as fh:
        json.dump({"scalar": scalar, "ragged": ragged}, fh)
    dist.destroy_process_group()


class TestEvaluatorDataParallel(unittest.TestCase):
    def test_dp_reduction_and_ragged_shards(self):
        import torch.multiprocessing as mp

        results_dir = tempfile.mkdtemp(prefix="eval_dp_")
        mp.spawn(_dp_worker, args=(2, "29591", results_dir), nprocs=2, join=True)

        results = []
        for r in range(2):
            with open(os.path.join(results_dir, f"rank{r}.json")) as fh:
                results.append(json.load(fh))
        # identical metrics on every rank
        self.assertEqual(results[0], results[1])
        scalar, ragged = results[0]["scalar"], results[0]["ragged"]
        # global scalar accuracy = token-weighted over ALL 3 batches (8/2/10
        # tokens) — not a rank-shard number and not a mean of per-batch means:
        # (0.5*8 + 0.7*2 + 0.9*10) / 20 = 0.72
        self.assertAlmostEqual(scalar["eval/avg_acc"], 0.72, places=6)
        # global token-weighted loss: (2*8 + 8*2 + 4*10) / 20 = 3.6
        self.assertAlmostEqual(scalar["eval/avg_loss"], 3.6, places=6)
        # the ragged pass completed (no hang) with rank0's counts as the total
        self.assertAlmostEqual(ragged["eval/avg_acc"], 0.75, places=6)
        self.assertAlmostEqual(ragged["eval/simulated_acc_len"], 1.125, places=6)


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

    def test_best_survives_restart(self):
        from specforge.training.checkpoint import CheckpointManager

        out = tempfile.mkdtemp(prefix="ckpt_mgr_rehydrate_")
        mgr = CheckpointManager(out, "run", max_checkpoints=2)
        mgr.save(self._state(1), 1)
        mgr.update_best(1, {"eval/simulated_acc_len": 5.0})

        # A fresh process: a NEW manager over the same output_dir rehydrates the
        # best record from best_meta.json instead of starting blind.
        mgr2 = CheckpointManager(out, "run", max_checkpoints=2)
        self.assertEqual(mgr2.best_step, 1)
        self.assertEqual(mgr2.best_score, 5.0)

        # rotation in the resumed process still protects the pre-restart best...
        mgr2.save(self._state(2), 2)
        mgr2.save(self._state(3), 3)
        mgr2.save(self._state(4), 4)
        self.assertTrue(os.path.exists(mgr2.checkpoint_dir(1)), "best rotated away")
        self.assertFalse(os.path.exists(mgr2.checkpoint_dir(2)))
        # ...and a worse post-restart score cannot overwrite it.
        self.assertFalse(mgr2.update_best(4, {"eval/simulated_acc_len": 4.0}))
        self.assertEqual(mgr2.best_step, 1)

    def test_read_resume_state_merges_rank_file(self):
        from specforge.training.checkpoint import CheckpointManager

        out = tempfile.mkdtemp(prefix="ckpt_mgr_rank_")
        mgr = CheckpointManager(out, "run")
        ckpt_dir = mgr.save(
            {"draft_state_dict": {}, "global_step": 7, "world_size": 1},
            7,
            rank_state={
                "optimizer": {"lr": 1.0},
                "rng": {"cpu": torch.get_rng_state()},
            },
        )
        state = CheckpointManager.read_resume_state(ckpt_dir)
        self.assertEqual(state["global_step"], 7)
        self.assertEqual(state["optimizer_state_dict"], {"lr": 1.0})
        self.assertIn("rng_state", state)
        # file:// URI form resolves identically
        state2 = CheckpointManager.read_resume_state(f"file://{ckpt_dir}")
        self.assertEqual(state2["global_step"], 7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
