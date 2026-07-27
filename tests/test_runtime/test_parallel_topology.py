# coding=utf-8
"""CPU seams for the unified entry's TP/DP/USP topology wiring."""

import unittest
from unittest import mock

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.launch import (
    _offline_io,
    _shard_offline_refs,
    build_disagg_offline_runtime,
    build_offline_runtime,
)


class ParallelTopologyTest(unittest.TestCase):
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
                _shard_offline_refs(refs, use_usp_preprocess=False, shuffle=False),
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
                # Repeating a draft-DP rank models the SP peers that must share
                # one order for USP sequence sharding.
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

    def test_offline_data_parallel_ranks_receive_disjoint_refs(self):
        # Validated offline configs keep tp_size=1, so the non-USP DP group is
        # the complete trainer world and every rank owns a distinct ref shard.
        refs = list(range(12))
        rank_shards = [
            _shard_offline_refs(
                refs,
                use_usp_preprocess=False,
                shuffle=False,
                dp_rank=rank,
                dp_size=4,
            )
            for rank in range(4)
        ]

        self.assertEqual(
            rank_shards,
            [
                [0, 4, 8],
                [1, 5, 9],
                [2, 6, 10],
                [3, 7, 11],
            ],
        )
        self.assertEqual(sorted(ref for shard in rank_shards for ref in shard), refs)

    def test_public_offline_builders_reject_trainer_tensor_parallelism(self):
        with self.assertRaisesRegex(
            ValueError, "do not implement trainer tensor parallelism"
        ):
            build_offline_runtime(
                algorithm=object(),
                hidden_states_path="/unused",
                draft_model=object(),
                target_head=None,
                optimizer_factory=object(),
                run_id="run",
                output_dir="/unused",
                tp_size=2,
            )

        with self.assertRaisesRegex(
            ValueError, "do not implement trainer tensor parallelism"
        ):
            build_disagg_offline_runtime(
                algorithm=object(),
                feature_store=object(),
                refs=[],
                draft_model=object(),
                target_head=None,
                optimizer_factory=object(),
                run_id="run",
                output_dir="/unused",
                tp_size=2,
            )

    def test_offline_io_resolves_the_builtin_provider_for_text(self):
        algorithm = builtin_algorithm_registry().resolve("eagle3")
        provider = algorithm.providers.offline_for("text")

        collator, normalizer = _offline_io(
            algorithm,
            "text",
            4096,
            ttt_length=7,
            use_usp_preprocess=False,
        )

        expected_normalizer = provider.build_normalizer(4096)
        self.assertIsInstance(collator, type(provider.build_collator()))
        self.assertIs(normalizer.func, expected_normalizer.func)
        self.assertEqual(normalizer.keywords, expected_normalizer.keywords)


if __name__ == "__main__":
    unittest.main(verbosity=2)
