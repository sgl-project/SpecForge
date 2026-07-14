# coding=utf-8
"""CPU seams for the unified entry's TP/DP/USP topology wiring."""

import unittest
from types import SimpleNamespace
from unittest import mock

from specforge.launch import (
    _offline_io,
    _plan_online_prompt_stream,
    _preposition_online_prompts,
    _shard_offline_refs,
    _shard_online_prompts,
)


class ParallelTopologyTest(unittest.TestCase):
    def test_online_target_dp_shards_are_disjoint_and_equally_padded(self):
        dp_group = object()
        prompts = [
            {"task_id": f"prompt-{index}", "payload": {"input_ids": [index]}}
            for index in range(5)
        ]

        def shard(dp_rank):
            with (
                mock.patch("torch.distributed.is_available", return_value=True),
                mock.patch("torch.distributed.is_initialized", return_value=True),
                mock.patch("torch.distributed.get_world_size", return_value=2),
                mock.patch("torch.distributed.get_rank", return_value=dp_rank),
                mock.patch("specforge.distributed.get_dp_group", return_value=dp_group),
            ):
                return _shard_online_prompts(prompts, shuffle=False)

        rank0 = shard(0)
        rank0_peer = shard(0)
        rank1 = shard(1)

        self.assertEqual(rank0, rank0_peer)
        self.assertEqual(len(rank0), len(rank1))
        self.assertEqual([item["payload"]["input_ids"][0] for item in rank0], [0, 2, 4])
        self.assertEqual([item["payload"]["input_ids"][0] for item in rank1], [1, 3, 0])
        self.assertTrue(all(item["metadata"]["target_dp_rank"] == 1 for item in rank1))

    def test_online_sampler_shuffles_by_epoch_and_drops_each_tp_tail(self):
        from torch.utils.data.distributed import DistributedSampler

        prompts = [
            {"task_id": f"prompt-{index}", "payload": {"input_ids": [index]}}
            for index in range(5)
        ]

        def source_indices(dp_rank, epoch):
            planned = _shard_online_prompts(
                prompts,
                seed=17,
                epoch=epoch,
                dp_rank=dp_rank,
                dp_size=2,
                tp_size=2,
                batch_size=1,
            )
            return [item["metadata"]["source_prompt_index"] for item in planned]

        for epoch in (0, 1):
            shards = []
            for dp_rank in (0, 1):
                sampler = DistributedSampler(
                    range(5),
                    num_replicas=2,
                    rank=dp_rank,
                    shuffle=True,
                    seed=17,
                    drop_last=False,
                )
                sampler.set_epoch(epoch)
                expected = list(sampler)[:2]
                actual = source_indices(dp_rank, epoch)
                self.assertEqual(actual, expected)
                # Repeating a target-DP rank models another peer in the same TP
                # group: it must see the identical full target capture shard.
                self.assertEqual(actual, source_indices(dp_rank, epoch))
                shards.append(set(actual))
            self.assertTrue(shards[0].isdisjoint(shards[1]))

        planned = _plan_online_prompt_stream(
            prompts,
            num_epochs=2,
            seed=17,
            dp_rank=0,
            dp_size=2,
            tp_size=2,
            batch_size=1,
        )
        # N=5, DP=2 pads each DP shard to 3. TP=2 and batch=1 retain two
        # prompts per epoch; the two one-prompt tails are never combined.
        self.assertEqual(len(planned), 4)
        self.assertEqual(
            [item["metadata"]["prompt_epoch"] for item in planned],
            [0, 0, 1, 1],
        )
        # One rank-local trained sample corresponds to a TP-wide pair of target
        # prompts, so resume skips exactly two entries from the planned queue.
        remaining = _preposition_online_prompts(
            planned, local_samples=1, tp_size=2
        )
        self.assertEqual(
            [item["metadata"]["prompt_epoch"] for item in remaining], [1, 1]
        )

    def test_offline_ref_sharding_pads_every_replica_to_equal_steps(self):
        dp_group = object()
        draft_dp_group = object()
        refs = list(range(5))

        def world_size(group):
            self.assertIn(group, (dp_group, draft_dp_group))
            return 2

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", side_effect=world_size),
            mock.patch("torch.distributed.get_rank", return_value=1),
            mock.patch("specforge.distributed.get_dp_group", return_value=dp_group),
            mock.patch(
                "specforge.distributed.get_draft_dp_group",
                return_value=draft_dp_group,
            ),
        ):
            self.assertEqual(
                _shard_offline_refs(
                    refs, use_usp_preprocess=False, shuffle=False
                ),
                [1, 3, 0],
            )
            self.assertEqual(
                _shard_offline_refs(refs, use_usp_preprocess=True, shuffle=False),
                [1, 3, 0],
            )

    def test_offline_sampler_rebuilds_seeded_dp_order_for_each_epoch(self):
        from torch.utils.data.distributed import DistributedSampler

        refs = list(range(5))
        usable_epochs = []
        for epoch in (0, 1):
            epoch_shards = []
            for dp_rank in (0, 1):
                expected_sampler = DistributedSampler(
                    refs,
                    num_replicas=2,
                    rank=dp_rank,
                    shuffle=True,
                    seed=23,
                    drop_last=False,
                )
                expected_sampler.set_epoch(epoch)
                expected = list(expected_sampler)
                actual = _shard_offline_refs(
                    refs,
                    use_usp_preprocess=False,
                    seed=23,
                    epoch=epoch,
                    dp_rank=dp_rank,
                    dp_size=2,
                )
                self.assertEqual(actual, expected)
                # Repeating a DP rank models the TP peers (or, for the USP
                # group selection, the SP peers) that must share one order.
                self.assertEqual(
                    actual,
                    _shard_offline_refs(
                        refs,
                        use_usp_preprocess=True,
                        seed=23,
                        epoch=epoch,
                        dp_rank=dp_rank,
                        dp_size=2,
                    ),
                )
                epoch_shards.append(actual[:2])
            self.assertTrue(set(epoch_shards[0]).isdisjoint(epoch_shards[1]))
            usable_epochs.append(epoch_shards[0])

        # Each rank receives three padded refs, but batch_size=2 drops the last
        # ref in each epoch. The two tails cannot form a cross-epoch batch.
        self.assertEqual(sum(map(len, usable_epochs)), 4)

    def test_offline_io_threads_usp_context_through_the_strategy_spec(self):
        calls = []
        spec = SimpleNamespace(
            name="eagle3",
            make_offline_collate=lambda: "collate",
            make_offline_transform=lambda max_len, **kw: calls.append((max_len, kw))
            or "transform",
        )
        self.assertEqual(
            _offline_io(
                spec,
                4096,
                ttt_length=7,
                use_usp_preprocess=True,
            ),
            ("collate", "transform"),
        )
        self.assertEqual(
            calls,
            [
                (
                    4096,
                    {"ttt_length": 7, "use_usp_preprocess": True},
                )
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
