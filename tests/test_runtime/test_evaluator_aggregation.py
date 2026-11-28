# coding=utf-8
"""Evaluator aggregation gates: correct/denom counts are summed over the whole
eval pass (all batches, all DP ranks) before any ratio or geometric sum, scalar
accuracy is weighted by its own denominator, and zero batches globally returns
``{}``. Plus CheckpointManager rotation/pointer/resume checks. CPU-only
(fabricated per-step metrics); the DP gate spawns a 2-process gloo group."""

import json
import os
import socket
import tempfile
import unittest
from contextlib import closing
from datetime import timedelta

import torch

from specforge.algorithms.objectives.base import ObjectiveResult
from specforge.runtime.contracts import TrainBatch
from tests.test_runtime._fixtures import FakeProgram, fake_spec


def _batch():
    return TrainBatch(
        sample_ids=["x"],
        strategy="eagle3",
        tensors={"loss_mask": torch.ones(4, dtype=torch.long)},
        metadata={},
    )


class _ScriptedProgram(FakeProgram):
    """Hands out prepared results, one for each batch it is asked to evaluate."""

    def __init__(self, results):
        super().__init__()
        self.results = iter(results)
        self.linear = torch.nn.Linear(1, 1)

    def forward(
        self, tensors, *, step, total_steps, collect_detailed_metrics, **replay
    ):
        return next(self.results)


def _evaluate(results, batches=()):
    """Pool the results as one evaluation pass of this rank, through the step runner."""
    from specforge.eval import Evaluator
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
    from specforge.training.contracts import StepContext
    from specforge.training.runner import DataParallelRunner

    runner = DataParallelRunner(
        _ScriptedProgram(results), FSDPTrainingBackend(ParallelConfig()), fake_spec()
    )
    if batches == ():
        batches = [_batch() for _ in results]
    return Evaluator().run(runner, batches, StepContext())


def _pairs(name, numerators, denominators):
    return {
        f"{name}_{depth}": (torch.tensor(float(n)), torch.tensor(float(d)))
        for depth, (n, d) in enumerate(zip(numerators, denominators))
    }


def _per_depth_result(loss, corrects, denoms, acceptance_rates=None, plosses=None):
    """Per-depth accuracy counts, with ``loss`` and the per-depth rates as means.

    The loss is a mean over the summed ``denoms``; a depth's acceptance rate
    and loss are means over that depth's denominator.
    """
    ratios = _pairs("acc", corrects, denoms)
    for name, means in (("acceptance_rate", acceptance_rates), ("ploss", plosses)):
        if means is not None:
            sums = [mean * denom for mean, denom in zip(means, denoms)]
            ratios.update(_pairs(name, sums, denoms))
    tokens = float(sum(denoms))
    return ObjectiveResult(
        torch.tensor(loss * tokens), torch.tensor(tokens), ratios=ratios
    )


def _scalar_result(loss, acc, tokens, denom):
    # DFlash/Domino-shaped result: a mean loss over ``tokens`` and a mean
    # accuracy over ``denom``, its own count, which need not equal the
    # loss-token count.
    return _additive_scalar_result(loss * tokens, tokens, acc * denom, denom)


def _additive_scalar_result(loss_num, loss_den, accuracy_num, accuracy_den):
    return ObjectiveResult(
        torch.tensor(float(loss_num)),
        torch.tensor(float(loss_den)),
        ratios={
            "acc": (
                torch.tensor(float(accuracy_num)),
                torch.tensor(float(accuracy_den)),
            )
        },
    )


def _chain_result(hits):
    """A block-shaped result: one fully supervised chain per row of ``hits``."""
    from specforge.algorithms.objectives.execution import accepted_draft_tokens

    hit = torch.tensor(hits, dtype=torch.bool)
    supervised = torch.ones_like(hit)
    return ObjectiveResult(
        torch.tensor(1.0),
        torch.tensor(1.0),
        ratios={
            "acc": (hit.sum().float(), supervised.sum().float()),
            "accepted_draft_tokens": accepted_draft_tokens(
                hit, supervised, torch.ones(hit.shape[1], dtype=torch.bool)
            ),
        },
    )


class TestEvaluatorAggregation(unittest.TestCase):
    def _run(self, results):
        return _evaluate(results)

    def test_scalar_acceptance_length_is_the_greedy_prefix(self):
        # Token accuracy ranks these two drafts the other way round: the
        # verifier rejects the first at its first token.
        late = _evaluate([_chain_result([[False, True, True, True]])])
        early = _evaluate([_chain_result([[True, True, False, False]])])
        self.assertEqual((late["eval/avg_acc"], early["eval/avg_acc"]), (0.75, 0.5))
        self.assertEqual(late["eval/simulated_acc_len"], 0.0)
        self.assertEqual(early["eval/simulated_acc_len"], 2.0)

    def test_scalar_acceptance_length_pools_chains_over_batches(self):
        hits = [[True, False, True], [True, True, True], [False, True, True]]
        split = _evaluate([_chain_result([row]) for row in hits])
        combined = _evaluate([_chain_result(hits)])
        self.assertEqual(split, combined)
        self.assertEqual(split["eval/simulated_acc_len"], 4 / 3)

    def test_aggregates_counts_before_geometric_sum(self):
        # pos0: (3+1)/(4+1)=0.8 ; pos1: (2+0)/(4+1)=0.4  -- NOT the mean of ratios.
        m = self._run(
            [
                _per_depth_result(2.0, corrects=[3, 2], denoms=[4, 4]),
                _per_depth_result(8.0, corrects=[1, 0], denoms=[1, 1]),
            ]
        )
        self.assertAlmostEqual(m["eval/avg_acc"], 0.8, places=6)
        # simulated_acc_len = 0.8 + 0.8*0.4 = 1.12
        self.assertAlmostEqual(m["eval/simulated_acc_len"], 1.12, places=6)
        # token-weighted loss: (2*8 + 8*2) / (8+2) = 3.2
        self.assertAlmostEqual(m["eval/avg_loss"], 3.2, places=6)

    def test_naive_average_would_differ(self):
        # Mean of per-batch pos0 accuracy is (0.75+1.0)/2 = 0.875; the correct
        # count-aggregated value is 0.8.
        m = self._run(
            [
                _per_depth_result(1.0, corrects=[3], denoms=[4]),
                _per_depth_result(1.0, corrects=[1], denoms=[1]),
            ]
        )
        self.assertAlmostEqual(m["eval/avg_acc"], 0.8, places=6)
        self.assertNotAlmostEqual(m["eval/avg_acc"], 0.875, places=3)

    def test_batch_size_invariance(self):
        split = self._run(
            [
                _per_depth_result(1.0, corrects=[3, 2], denoms=[4, 4]),
                _per_depth_result(1.0, corrects=[1, 0], denoms=[1, 1]),
            ]
        )
        combined = self._run([_per_depth_result(1.0, corrects=[4, 2], denoms=[5, 5])])
        self.assertAlmostEqual(
            split["eval/simulated_acc_len"],
            combined["eval/simulated_acc_len"],
            places=6,
        )
        self.assertAlmostEqual(
            split["eval/avg_acc"], combined["eval/avg_acc"], places=6
        )

    def test_zero_batches_returns_empty(self):
        # Fabricated zero metrics would poison best-checkpoint tracking.
        self.assertEqual(_evaluate([]), {})
        self.assertEqual(_evaluate([], batches=None), {})

    def test_counts_accumulate_in_float64(self):
        # float32 accumulation stalls at 2**24 (x+1 == x), which would report a
        # perfect 1.0 here; float64 keeps the counts exact.
        big = 2**24
        m = self._run(
            [_per_depth_result(1.0, corrects=[big], denoms=[big])]
            + [_per_depth_result(1.0, corrects=[0], denoms=[1]) for _ in range(3)]
        )
        self.assertLess(m["eval/avg_acc"], 1.0)
        self.assertAlmostEqual(m["eval/avg_acc"], big / (big + 3), places=12)

    def test_scalar_accuracy_degenerates_gracefully(self):
        # equal counts -> the count-weighted accuracy equals the plain mean.
        results = [
            _scalar_result(1.0, 0.5, tokens=4, denom=4),
            _scalar_result(1.0, 0.7, tokens=4, denom=4),
        ]
        m = _evaluate(results)
        self.assertAlmostEqual(m["eval/avg_acc"], 0.6, places=6)
        # Without chain counts there is no acceptance length to report.
        self.assertNotIn("eval/simulated_acc_len", m)

    def test_scalar_accuracy_weighted_by_accuracy_denom(self):
        # Realistic DFlash shapes: the accuracy denominator (anchor-block eval
        # positions) differs from the loss-token count. Invariance must come
        # from the accuracy's own denominator: 3-of-4 + 1-of-2 == 4-of-6
        # regardless of grouping.
        split = [
            _scalar_result(1.0, 3 / 4, tokens=10, denom=4),
            _scalar_result(1.0, 1 / 2, tokens=50, denom=2),
        ]
        combined = [_scalar_result(1.0, 4 / 6, tokens=60, denom=6)]
        for results in (split, combined):
            m = _evaluate(results)
            self.assertAlmostEqual(m["eval/avg_acc"], 4 / 6, places=6)
            # loss-token weighting would skew to (0.75*10 + 0.5*50)/60 ~ 0.542
            self.assertNotAlmostEqual(m["eval/avg_acc"], 32.5 / 60, places=2)

    def test_additive_scalar_terms_are_partition_invariant(self):
        split = self._run(
            [
                _additive_scalar_result(2, 1, 1, 1),
                _additive_scalar_result(30, 3, 1, 3),
            ]
        )
        combined = self._run([_additive_scalar_result(32, 4, 2, 4)])

        self.assertEqual(split, combined)
        self.assertEqual(split["eval/avg_loss"], 8.0)
        self.assertEqual(split["eval/avg_acc"], 0.5)

    def test_reports_per_position_acceptance(self):
        m = self._run([_per_depth_result(1.0, corrects=[3, 2], denoms=[4, 4])])
        self.assertAlmostEqual(m["eval/per_position_acc"][0], 0.75, places=6)
        self.assertAlmostEqual(m["eval/per_position_acc"][1], 0.5, places=6)
        self.assertEqual(len(m["eval/per_position_acc"]), 2)

    def test_emits_token_weighted_acceptance_rate_and_ploss(self):
        # eval/acceptance_rate_i and eval/ploss_i are token-weighted means over
        # the eval set (batch weights 8 and 2 here), emitted only when the
        # objective reports them.
        m = self._run(
            [
                _per_depth_result(
                    2.0,
                    corrects=[3, 2],
                    denoms=[4, 4],
                    acceptance_rates=[0.8, 0.6],
                    plosses=[1.0, 2.0],
                ),
                _per_depth_result(
                    8.0,
                    corrects=[1, 0],
                    denoms=[1, 1],
                    acceptance_rates=[0.4, 0.2],
                    plosses=[3.0, 4.0],
                ),
            ]
        )
        self.assertAlmostEqual(m["eval/acceptance_rate_0"], 0.72, places=6)
        self.assertAlmostEqual(m["eval/acceptance_rate_1"], 0.52, places=6)
        self.assertAlmostEqual(m["eval/expected_acceptance_0"], 0.72, places=6)
        self.assertAlmostEqual(m["eval/expected_acceptance_1"], 0.52, places=6)
        self.assertAlmostEqual(m["eval/ploss_0"], 1.4, places=6)
        self.assertAlmostEqual(m["eval/ploss_1"], 2.4, places=6)
        m2 = self._run([_per_depth_result(1.0, corrects=[1], denoms=[2])])
        self.assertNotIn("eval/acceptance_rate_0", m2)
        self.assertNotIn("eval/expected_acceptance_0", m2)
        self.assertNotIn("eval/ploss_0", m2)


def _free_port():
    # A fixed rendezvous port collides with parallel suite runs.
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _dp_worker(rank, world_size, port, results_dir):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    # Scenario 1 — scalar accuracy, uneven shards, accuracy denominator != loss
    # tokens: both weight sets must be reduced across ranks independently.
    if rank == 0:
        results = [_scalar_result(2.0, 0.5, tokens=8, denom=4)]
    else:
        results = [
            _scalar_result(8.0, 0.7, tokens=2, denom=1),
            _scalar_result(4.0, 0.9, tokens=10, denom=5),
        ]
    scalar = _evaluate(results)

    # Scenario 2 — ragged per-position shards: rank 1's shard is EMPTY. This
    # gates the pooling's collectives only; a collective forward (FSDP) would
    # additionally need equal per-rank batch counts — out of scope.
    results2 = (
        [_per_depth_result(1.0, corrects=[3, 2], denoms=[4, 4])] if rank == 0 else []
    )
    ragged = _evaluate(results2)

    # Scenario 3 — zero batches on EVERY rank: the {} verdict is global too.
    empty = _evaluate([])

    # Scenario 4 — ranks with different numbers of positions: the buffer every
    # rank pads to is as wide as the widest rank, not as wide as their sum.
    widths = _evaluate(
        [_per_depth_result(1.0, corrects=[1] * (2 + rank), denoms=[2] * (2 + rank))]
    )

    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as fh:
        json.dump(
            {"scalar": scalar, "ragged": ragged, "empty": empty, "widths": widths}, fh
        )
    dist.destroy_process_group()


class TestEvaluatorDataParallel(unittest.TestCase):
    def test_dp_reduction_and_ragged_shards(self):
        import torch.multiprocessing as mp

        results_dir = tempfile.mkdtemp(prefix="eval_dp_")
        mp.spawn(_dp_worker, args=(2, _free_port(), results_dir), nprocs=2, join=True)

        results = []
        for r in range(2):
            with open(os.path.join(results_dir, f"rank{r}.json")) as fh:
                results.append(json.load(fh))
        # identical metrics on every rank
        self.assertEqual(results[0], results[1])
        scalar, ragged = results[0]["scalar"], results[0]["ragged"]
        # global denom-weighted accuracy over ALL 3 batches (denoms 4/1/5):
        # (0.5*4 + 0.7*1 + 0.9*5) / 10 = 0.72 — not a rank-shard number.
        self.assertAlmostEqual(scalar["eval/avg_acc"], 0.72, places=6)
        # global token-weighted loss: (2*8 + 8*2 + 4*10) / 20 = 3.6
        self.assertAlmostEqual(scalar["eval/avg_loss"], 3.6, places=6)
        # the ragged pass completed (no hang) with rank0's counts as the total
        self.assertAlmostEqual(ragged["eval/avg_acc"], 0.75, places=6)
        self.assertAlmostEqual(ragged["eval/simulated_acc_len"], 1.125, places=6)
        self.assertEqual(results[0]["empty"], {})
        # two positions from both ranks, a third from rank 1 alone
        self.assertEqual(results[0]["widths"]["eval/per_position_acc"], [0.5, 0.5, 0.5])


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

        # {run_id}-latest -> step4, {run_id}-best -> step2
        self.assertEqual(os.path.realpath(mgr.latest_dir()), os.path.realpath(ck(4)))
        self.assertEqual(
            os.path.realpath(os.path.join(out, "run-best")), os.path.realpath(ck(2))
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
        # best record from {run_id}.best_meta.json instead of starting blind.
        mgr2 = CheckpointManager(out, "run", max_checkpoints=2)
        self.assertEqual(mgr2.best_step, 1)
        self.assertEqual(mgr2.best_score, 5.0)

        # rotation in the resumed process still protects the pre-restart best...
        mgr2.save(self._state(2), 2)
        mgr2.save(self._state(3), 3)
        mgr2.save(self._state(4), 4)
        self.assertTrue(os.path.exists(mgr2.checkpoint_dir(1)), "best rotated away")
        self.assertFalse(os.path.exists(mgr2.checkpoint_dir(2)))
        # ...and is_better (the caller's gate for update_best) rejects a worse
        # score, empty metrics, and the zero-batch-eval {}.
        self.assertFalse(mgr2.is_better({"eval/simulated_acc_len": 4.0}))
        self.assertFalse(mgr2.is_better({}))
        self.assertFalse(mgr2.is_better(None))
        self.assertTrue(mgr2.is_better({"eval/simulated_acc_len": 6.0}))
        self.assertEqual(mgr2.best_step, 1)

    def test_read_resume_state_carries_backend(self):
        from specforge.training.checkpoint import CheckpointManager

        out = tempfile.mkdtemp(prefix="ckpt_mgr_rank_")
        mgr = CheckpointManager(out, "run")
        ckpt_dir = mgr.save(
            {"draft_state_dict": {}, "global_step": 7, "world_size": 1},
            7,
            rank_state={
                "optimizer": {"lr": 1.0},
                "rng": {"torch": torch.get_rng_state(), "cuda": None},
            },
        )
        state = CheckpointManager.read_resume_state(ckpt_dir)
        self.assertEqual(state["global_step"], 7)
        # the rank file rides through untouched under 'backend'
        self.assertEqual(state["backend"]["optimizer"], {"lr": 1.0})
        self.assertIn("rng", state["backend"])
        self.assertNotIn("optimizer_state_dict", state)
        self.assertNotIn("rng_state", state)
        # file:// URI form resolves identically
        state2 = CheckpointManager.read_resume_state(f"file://{ckpt_dir}")
        self.assertEqual(state2["global_step"], 7)

    def test_read_resume_state_requires_rank_file(self):
        from specforge.training.checkpoint import CheckpointManager

        out = tempfile.mkdtemp(prefix="ckpt_mgr_norank_")
        mgr = CheckpointManager(out, "run")
        ckpt_dir = mgr.save({"draft_state_dict": {}, "global_step": 3}, 3)
        with self.assertRaises(ValueError):
            CheckpointManager.read_resume_state(ckpt_dir)
        state = CheckpointManager.read_resume_state(ckpt_dir, require_full_state=False)
        self.assertEqual(state["backend"], {})
        self.assertEqual(state["global_step"], 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
