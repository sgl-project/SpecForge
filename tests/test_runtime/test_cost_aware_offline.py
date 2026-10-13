# coding=utf-8
"""Offline cost-aware scheduling: sampler regrouping and the feature index."""

import json
import os
import tempfile
import unittest
from unittest import mock

import torch

from specforge.launch import _distributed_sampler_indices, _shard_offline_refs
from specforge.runtime.data_plane.offline_index import (
    INDEX_FILENAME,
    costs_digest,
    index_fingerprint,
    offline_sample_costs,
    offline_window_plan,
)
from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
from specforge.runtime.data_plane.window_plan import OfflineWindowPlan, SchedulingPolicy


def _write_features(root, lengths, with_mask=True):
    for index, length in enumerate(lengths):
        record = {"input_ids": torch.arange(length, dtype=torch.long)}
        if with_mask:
            mask = torch.ones(length)
            mask[: length // 2] = 0  # first half is prompt
            record["loss_mask"] = mask
        record["hidden_state"] = torch.zeros(length, 4)
        record["aux_hidden_state"] = torch.zeros(length, 12)
        torch.save(record, os.path.join(root, f"{index:04d}.ckpt"))


class SamplerPlanTest(unittest.TestCase):
    def test_plan_regroups_each_window_without_changing_its_members(self):
        size, dp, batch, window = 40, 2, 2, 4  # window = 16 samples
        costs = [(float(i % 17),) for i in range(size)]
        plan = OfflineWindowPlan(costs, batch, window)
        base = [
            _distributed_sampler_indices(size, dp_rank=r, dp_size=dp, seed=7, epoch=0)
            for r in range(dp)
        ]
        planned = [
            _distributed_sampler_indices(
                size, dp_rank=r, dp_size=dp, seed=7, epoch=0, plan=plan
            )
            for r in range(dp)
        ]
        # same shards per rank count, same multiset overall
        self.assertEqual([len(x) for x in planned], [len(x) for x in base])
        self.assertEqual(sorted(sum(planned, [])), sorted(sum(base, [])))
        window_samples = dp * batch * window
        for w in range(0, size, window_samples):
            base_members = set()
            planned_members = set()
            lo, hi = w // dp, (w + window_samples) // dp
            for r in range(dp):
                base_members.update(base[r][lo:hi])
                planned_members.update(planned[r][lo:hi])
            self.assertEqual(base_members, planned_members)
        # within a round every rank's micro-batch is a contiguous cost slice
        # and ranks are ordered: rank 0's cheapest >= rank 1's dearest
        for round_index in range(len(planned[0]) // batch):
            slices = [
                planned[r][round_index * batch : (round_index + 1) * batch]
                for r in range(dp)
            ]
            self.assertGreaterEqual(
                min(costs[i][0] for i in slices[0]),
                max(costs[i][0] for i in slices[1]),
            )

    def test_dropped_tail_keeps_its_original_order(self):
        size, dp, batch = 10, 2, 3
        plan = OfflineWindowPlan([(float(i),) for i in range(size)], batch, 2)
        kwargs = dict(dp_size=dp, seed=1, epoch=0, shuffle=False)
        base = [
            _distributed_sampler_indices(size, dp_rank=r, **kwargs) for r in range(dp)
        ]
        planned = [
            _distributed_sampler_indices(size, dp_rank=r, plan=plan, **kwargs)
            for r in range(dp)
        ]
        # one full round (positions 0-5) is planned: sorted desc 5..0 ->
        # rank 0 [5,4,3], rank 1 [2,1,0]; positions 6-9 are the per-rank
        # drop_last tails and stay exactly where the baseline leaves them.
        self.assertEqual(planned[0], [5, 4, 3, 6, 8])
        self.assertEqual(planned[1], [2, 1, 0, 7, 9])
        for rank in range(dp):
            kept = len(base[rank]) - len(base[rank]) % batch
            self.assertEqual(planned[rank][kept:], base[rank][kept:])

    def test_windows_follow_the_cross_epoch_accumulation_counter(self):
        for size, dp, batch, accum, epochs in (
            (12, 2, 1, 4, 2),
            (40, 2, 2, 4, 3),
            (26, 3, 2, 5, 4),
        ):
            with self.subTest(size=size, dp=dp, batch=batch, accum=accum):
                costs = [(float((i * 7) % 11),) for i in range(size)]
                plan = OfflineWindowPlan(costs, batch, accum)
                base_steps = []
                planned_steps = []
                for epoch in range(epochs):
                    for variant, steps in ((None, base_steps), (plan, planned_steps)):
                        shards = [
                            _distributed_sampler_indices(
                                size,
                                dp_rank=r,
                                dp_size=dp,
                                seed=5,
                                epoch=epoch,
                                plan=variant,
                            )
                            for r in range(dp)
                        ]
                        per_rank = len(shards[0]) - len(shards[0]) % batch  # drop_last
                        for rank in range(dp):
                            steps.append(
                                [
                                    shards[rank][i : i + batch]
                                    for i in range(0, per_rank, batch)
                                ]
                            )

                # Flatten every rank's micro-batches across epochs, then cut
                # the stream into optimizer steps of ``accum`` micro-batches.
                def step_sets(steps):
                    streams = [[] for _ in range(dp)]
                    for block_index, block in enumerate(steps):
                        streams[block_index % dp].extend(block)
                    count = len(streams[0]) // accum
                    return [
                        sorted(
                            i
                            for rank in range(dp)
                            for mb in streams[rank][s * accum : (s + 1) * accum]
                            for i in mb
                        )
                        for s in range(count)
                    ]

                base_sets = step_sets(base_steps)
                planned_sets = step_sets(planned_steps)
                self.assertGreater(len(base_sets), 1)
                self.assertEqual(planned_sets, base_sets)

    def test_plan_is_deterministic_per_seed_and_epoch(self):
        size = 32
        plan = OfflineWindowPlan([(float(i % 5),) for i in range(size)], 2, 2)
        kwargs = dict(dp_rank=0, dp_size=2, plan=plan)
        a = _distributed_sampler_indices(size, seed=3, epoch=1, **kwargs)
        b = _distributed_sampler_indices(size, seed=3, epoch=1, **kwargs)
        c = _distributed_sampler_indices(size, seed=3, epoch=2, **kwargs)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_shard_offline_refs_threads_the_plan(self):
        refs = [f"r{i}" for i in range(8)]
        plan = OfflineWindowPlan([(float(i),) for i in range(8)], 1, 4)
        rank0 = _shard_offline_refs(
            refs,
            use_usp_preprocess=False,
            shuffle=False,
            dp_rank=0,
            dp_size=2,
            plan=plan,
        )
        rank1 = _shard_offline_refs(
            refs,
            use_usp_preprocess=False,
            shuffle=False,
            dp_rank=1,
            dp_size=2,
            plan=plan,
        )
        self.assertEqual(sorted(rank0 + rank1), refs)
        # window of 8, rounds of 2 (batch 1): sorted desc r7..r0 -> rank 0 gets
        # the dearer member of each round (round order shuffled by the rng).
        pairs = sorted(zip(rank0, rank1))
        self.assertEqual(
            pairs, [("r1", "r0"), ("r3", "r2"), ("r5", "r4"), ("r7", "r6")]
        )


class OfflineIndexTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="las_features_")
        self.cache = tempfile.mkdtemp(prefix="las_cache_")

    def _refs(self):
        return OfflineManifestReader(
            self.root,
            run_id="idx",
            strategy="dspark",
            feature_keys=("input_ids", "loss_mask", "hidden_state"),
            max_len=8,
        ).read()

    def test_scans_files_once_and_reuses_the_cache(self):
        _write_features(self.root, [4, 10, 6])
        refs = self._refs()
        policy = SchedulingPolicy("anchors", num_anchors=512)
        costs, applied, fingerprint = offline_sample_costs(
            refs, policy, max_len=8, cache_dir=self.cache
        )
        # loss mask: first half prompt -> valid anchors = supervised pairs
        # len 4: mask 0011 -> 1 ; len 10 clamped to 8: mask 00000111 -> 2 ;
        # len 6: 000111 -> 2
        self.assertEqual(costs, [(1.0,), (2.0,), (2.0,)])
        self.assertEqual(applied.key, "anchors")
        self.assertIsNotNone(fingerprint)
        cache_path = os.path.join(self.cache, INDEX_FILENAME)
        with open(cache_path) as handle:
            payload = json.load(handle)
        self.assertEqual(payload["fingerprint"], fingerprint)
        with mock.patch(
            "specforge.runtime.data_plane.offline_index.scan_feature_files",
            side_effect=AssertionError("cache not used"),
        ):
            again, _, fp2 = offline_sample_costs(
                refs, SchedulingPolicy("length"), max_len=8, cache_dir=self.cache
            )
        self.assertEqual(fp2, fingerprint)
        self.assertEqual(again, [(4.0,), (8.0,), (6.0,)])

    def test_stale_cache_is_rebuilt(self):
        _write_features(self.root, [4, 6])
        refs = self._refs()
        offline_sample_costs(
            refs, SchedulingPolicy("length"), max_len=8, cache_dir=self.cache
        )
        _write_features(self.root, [4, 6, 5])
        refs = self._refs()
        costs, _, _ = offline_sample_costs(
            refs, SchedulingPolicy("length"), max_len=8, cache_dir=self.cache
        )
        self.assertEqual(costs, [(4.0,), (6.0,), (5.0,)])

    def test_anchor_key_degrades_to_length_without_loss_mask(self):
        _write_features(self.root, [4, 6], with_mask=False)
        refs = OfflineManifestReader(
            self.root,
            run_id="idx",
            strategy="dspark",
            feature_keys=("input_ids",),
            max_len=8,
        ).read()
        costs, applied, _ = offline_sample_costs(
            refs, SchedulingPolicy("anchors"), max_len=8, cache_dir=self.cache
        )
        self.assertEqual(applied.key, "length")
        self.assertEqual(costs, [(4.0,), (6.0,)])

    def test_refs_with_metadata_need_no_scan(self):
        from specforge.runtime.contracts import SampleRef
        from specforge.runtime.data_plane.window_plan import VALID_ANCHORS_METADATA_KEY

        refs = [
            SampleRef(
                sample_id=f"m{i}",
                run_id="m",
                source_task_id=None,
                feature_store_uri=f"disagg://m/{i}",
                feature_keys={},
                feature_specs={},
                strategy="dspark",
                num_tokens=100 * (i + 1),
                metadata={VALID_ANCHORS_METADATA_KEY: 50 * (i + 1)},
            )
            for i in range(3)
        ]
        with mock.patch(
            "specforge.runtime.data_plane.offline_index.scan_feature_files",
            side_effect=AssertionError("no scan expected"),
        ):
            costs, applied, fingerprint = offline_sample_costs(
                refs, SchedulingPolicy("anchors", num_anchors=120), max_len=250
            )
        self.assertIsNone(fingerprint)
        self.assertEqual(costs, [(50.0,), (100.0,), (120.0,)])
        # an older producer wrote no anchor count: the whole set uses length
        bare = [r.__class__(**{**r.__dict__, "metadata": {}}) for r in refs]
        costs, applied, _ = offline_sample_costs(
            bare, SchedulingPolicy("anchors"), max_len=250
        )
        self.assertEqual(applied.key, "length")
        self.assertEqual(costs, [(100.0,), (200.0,), (250.0,)])
        no_length = [r.__class__(**{**r.__dict__, "num_tokens": 0}) for r in bare]
        with self.assertRaisesRegex(ValueError, "neither num_tokens"):
            offline_sample_costs(no_length, SchedulingPolicy("length"), max_len=250)

    def test_window_plan_contract_records_effective_window(self):
        _write_features(self.root, [4, 6, 5, 7])
        refs = self._refs()
        plan, contract = offline_window_plan(
            refs,
            SchedulingPolicy("length"),
            batch_size=2,
            accumulation_steps=3,
            max_len=8,
            cache_dir=self.cache,
        )
        self.assertEqual(plan.batch_size, 2)
        self.assertEqual(plan.window_rounds, 3)
        self.assertEqual(contract["window_rounds"], 3)
        self.assertEqual(contract["key"], "length")
        self.assertEqual(contract["max_len"], 8)
        self.assertEqual(contract["costs_digest"], costs_digest(plan.costs))

    def test_contract_changes_with_anything_that_reorders_samples(self):
        from specforge.runtime.contracts import SampleRef
        from specforge.runtime.data_plane.window_plan import VALID_ANCHORS_METADATA_KEY

        refs = [
            SampleRef(
                sample_id=f"m{i}",
                run_id="m",
                source_task_id=None,
                feature_store_uri=f"disagg://m/{i}",
                feature_keys={},
                feature_specs={},
                strategy="dspark",
                num_tokens=100 * (i + 1),
                metadata={VALID_ANCHORS_METADATA_KEY: 40 * (i + 1)},
            )
            for i in range(4)
        ]

        def contract(policy, max_len, refs=refs):
            return offline_window_plan(
                refs, policy, batch_size=1, accumulation_steps=2, max_len=max_len
            )[1]

        base = contract(SchedulingPolicy("length"), 400)
        self.assertEqual(base, contract(SchedulingPolicy("length"), 400))
        # max_len clamps lengths, so the costs and the digest change.
        self.assertNotEqual(
            base["costs_digest"],
            contract(SchedulingPolicy("length"), 250)["costs_digest"],
        )
        self.assertNotEqual(
            base["max_len"], contract(SchedulingPolicy("length"), 250)["max_len"]
        )
        # a different key, cap or metadata changes the digest too
        self.assertNotEqual(
            base["costs_digest"],
            contract(SchedulingPolicy("anchors"), 400)["costs_digest"],
        )
        self.assertNotEqual(
            contract(SchedulingPolicy("anchors", num_anchors=100), 400)["costs_digest"],
            contract(SchedulingPolicy("anchors", num_anchors=500), 400)["costs_digest"],
        )
        edited = list(refs)
        edited[0] = SampleRef(**{**refs[0].__dict__, "num_tokens": 350})
        self.assertNotEqual(
            base["costs_digest"],
            contract(SchedulingPolicy("length"), 400, edited)["costs_digest"],
        )

    def test_same_size_rewrite_with_newer_mtime_rescans(self):
        _write_features(self.root, [6, 6])
        refs = self._refs()
        costs, _, before = offline_sample_costs(
            refs, SchedulingPolicy("anchors"), max_len=8, cache_dir=self.cache
        )
        self.assertEqual(costs, [(2.0,), (2.0,)])  # mask 000111 -> 2 supervised pairs
        path = os.path.join(self.root, "0000.ckpt")
        record = torch.load(path, weights_only=False)
        size = os.stat(path).st_size
        record["loss_mask"] = torch.ones(6)  # same shape and size, 5 supervised pairs
        torch.save(record, path)
        later = os.stat(path).st_mtime_ns + 2_000_000_000
        os.utime(path, ns=(later, later))
        if os.stat(path).st_size == size:
            self.assertNotEqual(
                before,
                index_fingerprint([path, os.path.join(self.root, "0001.ckpt")], 8),
            )
        costs, _, after = offline_sample_costs(
            refs, SchedulingPolicy("anchors"), max_len=8, cache_dir=self.cache
        )
        self.assertNotEqual(before, after)
        self.assertEqual(costs, [(5.0,), (2.0,)])
        self.assertEqual(
            offline_window_plan(
                refs, None, batch_size=2, accumulation_steps=3, max_len=8
            ),
            (None, None),
        )


if __name__ == "__main__":
    unittest.main()
