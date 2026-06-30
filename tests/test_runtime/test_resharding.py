# coding=utf-8
"""Resharding contract: consumer re-partitions a stable committed pool (M6).

The numerical correctness of resharding tensors across a different rank layout is
the GPU gate (``test_equiv_4rank.py``). These CPU tests lock down the *control*
side: partitioning is a stable, consumer-side function of ``sample_id``, so the
same pool re-distributes cleanly when the consumer's DP width changes — and no
sample is leased twice or dropped across a reshard.
"""

import unittest

from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue, dp_partition


def _ref(sid):
    return SampleRef(
        sample_id=sid,
        run_id="r",
        source_task_id=None,
        feature_store_uri=f"mem://st/{sid}",
        feature_keys={"x": f"{sid}/x"},
        feature_specs={},
        strategy="eagle3",
    )


class TestDpPartition(unittest.TestCase):
    def test_stable_and_in_range(self):
        for sid in (f"s{i}" for i in range(100)):
            p = dp_partition(sid, 4)
            self.assertEqual(p, dp_partition(sid, 4))  # stable
            self.assertTrue(0 <= p < 4)

    def test_single_partition_is_zero(self):
        self.assertEqual(dp_partition("anything", 1), 0)

    def test_roughly_balanced(self):
        counts = [0, 0, 0, 0]
        for i in range(4000):
            counts[dp_partition(f"sample-{i}", 4)] += 1
        # within ~20% of even (1000 each) — hash balance, not exact
        for c in counts:
            self.assertTrue(800 < c < 1200, counts)


class TestQueueResharding(unittest.TestCase):
    def test_two_shards_partition_the_pool_disjointly(self):
        q = SampleRefQueue()
        q.put([_ref(f"s{i}") for i in range(50)])
        shard0 = q.get(100, partition=(0, 2))
        shard1 = q.get(100, partition=(1, 2))
        ids0 = {r.sample_id for r in shard0}
        ids1 = {r.sample_id for r in shard1}
        self.assertEqual(ids0 & ids1, set())  # disjoint
        self.assertEqual(ids0 | ids1, {f"s{i}" for i in range(50)})  # complete
        for sid in ids0:
            self.assertEqual(dp_partition(sid, 2), 0)

    def test_reshard_to_wider_layout_consumes_remainder_once(self):
        # Start consuming under width 2 (one shard), then the leftover pool is
        # consumed under width 3 — every sample is leased exactly once total.
        q = SampleRefQueue()
        all_ids = {f"s{i}" for i in range(60)}
        q.put([_ref(s) for s in all_ids])
        leased = set()

        first = q.get(100, partition=(0, 2))  # width-2 shard 0
        leased |= {r.sample_id for r in first}
        # ... reshard to width 3: the remaining (still-pending) refs go to 3 shards
        for idx in range(3):
            got = q.get(100, partition=(idx, 3))
            new = {r.sample_id for r in got}
            self.assertEqual(new & leased, set())  # never re-leased
            leased |= new
        self.assertEqual(leased, all_ids)  # nothing lost
        self.assertEqual(q.depth(), 0)

    def test_partitioned_get_does_not_block_on_empty_shard(self):
        q = SampleRefQueue()
        q.put([_ref("s0")])  # only sample maps to some partition
        p = dp_partition("s0", 4)
        empty_idx = (p + 1) % 4
        # a shard with no matching ref returns [] immediately, even with timeout
        self.assertEqual(q.get(10, timeout_s=0.05, partition=(empty_idx, 4)), [])
        self.assertEqual(len(q.get(10, partition=(p, 4))), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
