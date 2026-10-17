# coding=utf-8
"""Pure planner tests for cost-aware scheduling (CPU only)."""

import random
import unittest

from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.window_plan import (
    VALID_ANCHORS_METADATA_KEY,
    SchedulingPolicy,
    count_valid_anchors,
    interleave_rounds,
    plan_window,
    ref_cost,
    round_robin_rounds,
)


def _ref(sid, num_tokens=0, valid_anchors=None):
    metadata = {}
    if valid_anchors is not None:
        metadata[VALID_ANCHORS_METADATA_KEY] = valid_anchors
    return SampleRef(
        sample_id=sid,
        run_id="run0",
        source_task_id=None,
        feature_store_uri=f"mooncake://run0/{sid}",
        feature_keys={},
        feature_specs={},
        strategy="dspark",
        num_tokens=num_tokens,
        metadata=metadata,
    )


class CountValidAnchorsTest(unittest.TestCase):
    def test_counts_consecutive_supervised_pairs(self):
        self.assertEqual(count_valid_anchors([0, 1, 1, 1, 0, 1, 1]), 3)
        self.assertEqual(count_valid_anchors([1]), 0)
        self.assertEqual(count_valid_anchors([]), 0)

    def test_tensor_matches_sequence(self):
        import torch

        mask = torch.tensor([0, 1, 1, 0, 1, 1, 1], dtype=torch.float32)
        self.assertEqual(count_valid_anchors(mask), count_valid_anchors(mask.tolist()))
        self.assertEqual(count_valid_anchors(mask), 3)
        self.assertEqual(count_valid_anchors(mask[:1]), 0)


class SchedulingPolicyTest(unittest.TestCase):
    def test_keys(self):
        self.assertEqual(SchedulingPolicy("length").sample_cost(100, None), (100.0,))
        anchors = SchedulingPolicy("anchors", num_anchors=512)
        self.assertEqual(anchors.sample_cost(8000, 700), (512.0,))
        self.assertEqual(anchors.sample_cost(8000, 300), (300.0,))
        self.assertIsNone(anchors.sample_cost(8000, None))
        self.assertIsNone(anchors.sample_cost(None, 300))

    def test_validation_and_contract(self):
        with self.assertRaises(ValueError):
            SchedulingPolicy("auto")
        with self.assertRaises(ValueError):
            SchedulingPolicy("length", num_anchors=0)
        with self.assertRaises(ValueError):
            SchedulingPolicy("length", window_rounds=-1)
        with self.assertRaises(ValueError):
            SchedulingPolicy("cost")
        contract = SchedulingPolicy("anchors", window_rounds=4).contract()
        self.assertEqual(
            contract, {"key": "anchors", "num_anchors": 512, "window_rounds": 4}
        )
        self.assertEqual(SchedulingPolicy("anchors").with_key("length").key, "length")
        self.assertTrue(SchedulingPolicy("anchors").uses_anchors)
        self.assertFalse(SchedulingPolicy("length").uses_anchors)


class RefCostTest(unittest.TestCase):
    def test_reads_num_tokens_and_metadata(self):
        policy = SchedulingPolicy("anchors", num_anchors=512)
        self.assertEqual(ref_cost(policy, _ref("a", 4000, 600)), (512.0,))
        self.assertIsNone(ref_cost(policy, _ref("a", 4000)))
        self.assertEqual(
            ref_cost(SchedulingPolicy("length"), _ref("a", 4000)), (4000.0,)
        )
        self.assertEqual(
            ref_cost(SchedulingPolicy("length"), _ref("a", 9000), max_len=8192),
            (8192.0,),
        )
        self.assertIsNone(ref_cost(SchedulingPolicy("length"), _ref("a")))


class PlanWindowTest(unittest.TestCase):
    def test_conserves_samples_and_groups_similar_costs(self):
        rng = random.Random(0)
        costs = [(float(rng.randint(1, 1000)),) for _ in range(24)]
        rounds = plan_window(costs, dp_size=3, batch_size=2)
        self.assertEqual(len(rounds), 4)
        flat = [i for rnd in rounds for slice_ in rnd for i in slice_]
        self.assertEqual(sorted(flat), list(range(24)))
        for rnd in rounds:
            self.assertEqual([len(slice_) for slice_ in rnd], [2, 2, 2])
        # Rounds are descending: every sample of round r costs at least as much
        # as every sample of round r + 1, and inside a round rank k's slice
        # costs at least rank k + 1's.
        round_costs = [
            [costs[i][0] for slice_ in rnd for i in slice_] for rnd in rounds
        ]
        for earlier, later in zip(round_costs, round_costs[1:]):
            self.assertGreaterEqual(min(earlier), max(later))
        for rnd in rounds:
            for left, right in zip(rnd, rnd[1:]):
                self.assertGreaterEqual(
                    min(costs[i][0] for i in left), max(costs[i][0] for i in right)
                )

    def test_ties_follow_tiebreak(self):
        costs = [(5.0,)] * 4
        rounds = plan_window(costs, 2, 1, tiebreak=["d", "a", "c", "b"])
        self.assertEqual(rounds, [[[1], [3]], [[2], [0]]])
        self.assertEqual(plan_window(costs, 2, 1), [[[0], [1]], [[2], [3]]])

    def test_rng_permutes_rounds(self):
        costs = [(float(i),) for i in range(16)]
        base = plan_window(costs, 2, 2)
        shuffled = plan_window(costs, 2, 2, rng=random.Random(3))
        self.assertNotEqual(base, shuffled)
        self.assertEqual(sorted(map(str, base)), sorted(map(str, shuffled)))

    def test_rejects_windows_without_whole_rounds(self):
        with self.assertRaises(ValueError):
            plan_window([(1.0,)] * 5, dp_size=2, batch_size=1)
        with self.assertRaises(ValueError):
            plan_window([(1.0,)] * 10, dp_size=2, batch_size=3)
        with self.assertRaises(ValueError):
            plan_window([(1.0,)] * 4, dp_size=2, batch_size=1, tiebreak=["a"])

    def test_interleave_reproduces_rank_slices_under_striding(self):
        costs = [(float(i),) for i in range(12)]
        rounds = plan_window(costs, dp_size=2, batch_size=3)
        order = interleave_rounds(rounds, 2)
        self.assertEqual(sorted(order), list(range(12)))
        for rank in range(2):
            mine = order[rank::2]
            chunks = [mine[i : i + 3] for i in range(0, len(mine), 3)]
            self.assertEqual(chunks, [rnd[rank] for rnd in rounds])

    def test_round_robin_matches_index_modulo_dp(self):
        self.assertEqual(
            round_robin_rounds(8, dp_size=2, batch_size=2),
            [[[0, 2], [1, 3]], [[4, 6], [5, 7]]],
        )
        self.assertEqual(
            interleave_rounds(round_robin_rounds(8, 2, 2), 2), list(range(8))
        )


if __name__ == "__main__":
    unittest.main()
