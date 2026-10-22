"""Length grouping preserves distributed sampling and resumable epoch plans."""

import random
import types
import unittest
from collections import Counter
from dataclasses import replace
from unittest import mock

from specforge.data.length_bucketing import (
    bucket_by_length,
    length_bucket_fingerprint,
    sample_length,
)
from specforge.runtime.contracts import FeatureSpec, SampleRef


def _ref(index, length=0):
    return SampleRef(
        sample_id=str(index),
        run_id="length-test",
        source_task_id=None,
        feature_store_uri=f"file:///features/{index}.ckpt",
        feature_keys={"input_ids": "input_ids"},
        feature_specs={},
        strategy="dflash",
        num_tokens=length,
    )


class LengthBucketPlanTest(unittest.TestCase):
    def test_disabled_preserves_order_without_consulting_lengths(self):
        items = [4, 2, 4, 1]
        length_fn = mock.Mock(side_effect=AssertionError("read length"))
        self.assertEqual(
            bucket_by_length(items, length_fn=length_fn, batch_size=2), items
        )
        length_fn.assert_not_called()

    def test_reads_only_ref_metadata_and_respects_truncation(self):
        self.assertEqual(sample_length(_ref(0, 8192), max_len=2048), 2048)
        self.assertIsNone(sample_length(_ref(1)))
        for shape in ((321,), (1, 321)):
            ref = types.SimpleNamespace(
                num_tokens=0,
                feature_specs={
                    "input_ids": FeatureSpec("input_ids", shape, "torch.int64")
                },
            )
            self.assertEqual(sample_length(ref), 321)
        ref.feature_specs["input_ids"] = FeatureSpec(
            "input_ids", (2, 321), "torch.int64"
        )
        self.assertIsNone(sample_length(ref))

    def test_rank_striding_produces_similar_length_local_batches(self):
        items = list(range(1, 17))
        random.Random(17).shuffle(items)
        planned = bucket_by_length(
            items,
            length_fn=lambda x: x,
            batch_size=2,
            dp_size=2,
            length_bucket_size=4,
        )
        batches = []
        for rank in range(2):
            shard = planned[rank::2]
            batches.extend(tuple(sorted(shard[i : i + 2])) for i in range(0, 8, 2))
        self.assertCountEqual(batches, [(i, i + 1) for i in range(1, 17, 2)])

    def test_resume_fingerprint_detects_order_id_and_effective_length_changes(self):
        refs = [_ref("short", 16), _ref("long", 512)]
        fingerprint = length_bucket_fingerprint(refs, max_len=128)
        # File location and uncropped tail do not determine the sample order.
        equivalent = [
            replace(refs[0], feature_store_uri="file:///another/location.ckpt"),
            replace(refs[1], num_tokens=1024),
        ]
        self.assertEqual(
            fingerprint, length_bucket_fingerprint(equivalent, max_len=128)
        )
        self.assertEqual(
            fingerprint, length_bucket_fingerprint(list(refs), max_len=128)
        )
        for changed in (
            list(reversed(refs)),
            [replace(refs[0], sample_id="changed"), refs[1]],
            [replace(refs[0], num_tokens=17), refs[1]],
        ):
            self.assertNotEqual(
                fingerprint, length_bucket_fingerprint(changed, max_len=128)
            )
        self.assertNotEqual(fingerprint, length_bucket_fingerprint(refs, max_len=64))

    def test_windows_and_each_rank_drop_last_tail_preserve_sample_multiset(self):
        for dp_size in (1, 2, 3, 8):
            for batch_size in (1, 2, 5):
                for local_size in (1, 7, 13):
                    with self.subTest(dp=dp_size, batch=batch_size, size=local_size):
                        # Duplicate ids model DistributedSampler's padding.
                        items = [i % 11 for i in range(dp_size * local_size)]
                        random.Random(31).shuffle(items)
                        quantum = batch_size * dp_size
                        window = 3 * quantum
                        usable = len(items) // quantum * quantum
                        planned = bucket_by_length(
                            items,
                            length_fn=lambda x: x + 1,
                            batch_size=batch_size,
                            dp_size=dp_size,
                            length_bucket_size=3,
                            seed=31,
                        )
                        self.assertEqual(len(planned), len(items))
                        self.assertEqual(Counter(planned), Counter(items))
                        for start in range(0, usable, window):
                            stop = min(start + window, usable)
                            self.assertEqual(
                                Counter(planned[start:stop]), Counter(items[start:stop])
                            )
                        for rank in range(dp_size):
                            baseline_tail = items[rank::dp_size][
                                local_size // batch_size * batch_size :
                            ]
                            actual_tail = planned[rank::dp_size][
                                local_size // batch_size * batch_size :
                            ]
                            self.assertEqual(actual_tail, baseline_tail)

    def test_unknown_length_keeps_its_whole_window_in_original_order(self):
        items = [8, 1, 7, 2, 6, 3, 5, 4, 12, 9, 11, 10]
        planned = bucket_by_length(
            items,
            length_fn=lambda x: None if x == 6 else x,
            batch_size=2,
            length_bucket_size=4,
        )
        self.assertEqual(planned[:8], items[:8])
        self.assertCountEqual(planned[8:], items[8:])

    def test_plan_is_epoch_seed_deterministic_and_does_not_touch_global_rng(self):
        items = list(range(1, 65))
        random.Random(19).shuffle(items)
        state = random.getstate()
        kwargs = dict(
            length_fn=lambda x: x,
            batch_size=2,
            dp_size=2,
            length_bucket_size=8,
            seed=19,
        )
        epoch_zero = bucket_by_length(items, epoch=0, **kwargs)
        self.assertEqual(epoch_zero, bucket_by_length(items, epoch=0, **kwargs))
        self.assertNotEqual(epoch_zero, bucket_by_length(items, epoch=1, **kwargs))
        self.assertEqual(random.getstate(), state)
        # A resumed loader seeks into the rebuilt rank-local plan.
        self.assertEqual(
            epoch_zero[1::2][6:],
            bucket_by_length(items, epoch=0, **kwargs)[1::2][6:],
        )

    def test_mixed_length_padding_cost_decreases_without_removing_tokens(self):
        lengths = [8, 1024, 16, 512, 32, 256, 64, 128] * 8
        planned = bucket_by_length(
            lengths,
            length_fn=lambda x: x,
            batch_size=2,
            length_bucket_size=8,
        )
        padded = lambda xs: sum(2 * max(xs[i : i + 2]) for i in range(0, len(xs), 2))
        self.assertEqual(sum(planned), sum(lengths))
        self.assertLess(padded(planned), padded(lengths))

    def test_rejects_invalid_window_and_unpadded_distributed_plan(self):
        with self.assertRaisesRegex(ValueError, "length_bucket_size"):
            bucket_by_length(
                [], length_fn=lambda _: 1, batch_size=1, length_bucket_size=-1
            )
        with self.assertRaisesRegex(ValueError, "DP-padded"):
            bucket_by_length(
                [1],
                length_fn=lambda _: 1,
                batch_size=1,
                dp_size=2,
                length_bucket_size=2,
            )


class OfflineLengthBucketWiringTest(unittest.TestCase):
    def test_disabled_and_eval_match_pytorch_distributed_sampler_exactly(self):
        from torch.utils.data.distributed import DistributedSampler

        from specforge.launch import _distributed_sampler_indices

        for size in (0, 1, 5, 19):
            for dp_size in (1, 2, 4):
                for rank in range(dp_size):
                    for epoch in (0, 1):
                        for shuffle in (False, True):
                            sampler = DistributedSampler(
                                list(range(size)),
                                num_replicas=dp_size,
                                rank=rank,
                                seed=37,
                                shuffle=shuffle,
                                drop_last=False,
                            )
                            sampler.set_epoch(epoch)
                            kwargs = dict(
                                dp_rank=rank,
                                dp_size=dp_size,
                                seed=37,
                                epoch=epoch,
                                shuffle=shuffle,
                            )
                            self.assertEqual(
                                _distributed_sampler_indices(size, **kwargs),
                                list(sampler),
                            )
                            if not shuffle:
                                self.assertEqual(
                                    _distributed_sampler_indices(
                                        size,
                                        length_bucket_size=4,
                                        lengths=list(range(1, size + 1)),
                                        **kwargs,
                                    ),
                                    list(sampler),
                                )

    def test_enabled_shards_preserve_the_same_padding_and_dropped_ids(self):
        from specforge.launch import _shard_offline_refs

        refs = [_ref(i, 1 + (i * 17) % 113) for i in range(19)]
        baseline, planned = [], []
        for rank in range(3):
            kwargs = dict(
                use_usp_preprocess=False,
                seed=13,
                epoch=2,
                dp_rank=rank,
                dp_size=3,
                batch_size=2,
            )
            old = _shard_offline_refs(refs, **kwargs)
            new = _shard_offline_refs(refs, length_bucket_size=3, **kwargs)
            self.assertEqual(old[-1], new[-1])  # Seven refs per rank, B=2.
            self.assertEqual(
                new, _shard_offline_refs(refs, length_bucket_size=3, **kwargs)
            )
            baseline.extend(ref.sample_id for ref in old[:6])
            planned.extend(ref.sample_id for ref in new[:6])
        self.assertCountEqual(planned, baseline)

    def test_direct_builders_reject_unvalidated_algorithm_before_reading_features(self):
        from specforge.launch import build_disagg_offline_runtime, build_offline_runtime

        algorithm = types.SimpleNamespace(name="eagle3")
        kwargs = dict(
            algorithm=algorithm,
            draft_model=None,
            target_head=None,
            optimizer_factory=None,
            run_id="r",
            output_dir="/out",
            length_bucket_size=4,
        )
        with self.assertRaisesRegex(ValueError, "only strategy='dflash'"):
            build_offline_runtime(hidden_states_path="/not-read", **kwargs)
        with self.assertRaisesRegex(ValueError, "only strategy='dflash'"):
            build_disagg_offline_runtime(feature_store=None, refs=[], **kwargs)

    def test_builders_record_sampling_contract_and_rebuild_epoch_order(self):
        from specforge.launch import build_disagg_offline_runtime, build_offline_runtime

        refs = [_ref(i, 8 + i * 17) for i in range(32)]
        provider = types.SimpleNamespace(
            build_reader=lambda *_args, **_kwargs: types.SimpleNamespace(
                read=lambda: refs
            )
        )
        algorithm = types.SimpleNamespace(
            name="dflash",
            providers=types.SimpleNamespace(
                offline_for=lambda _: provider,
                step=types.SimpleNamespace(uses_external_target_head=False),
            ),
        )
        for builder, source in (
            (build_offline_runtime, dict(hidden_states_path="/features")),
            (build_disagg_offline_runtime, dict(feature_store=object(), refs=refs)),
        ):
            for window in (0, 4):
                with (
                    mock.patch("specforge.launch.DataFlowController"),
                    mock.patch("specforge.launch.LocalFeatureStore"),
                    mock.patch(
                        "specforge.launch._offline_io", return_value=(None, None)
                    ),
                    mock.patch("specforge.launch._assemble_trainer") as assemble,
                    mock.patch(
                        "specforge.data.offline_lengths.ensure_offline_lengths",
                        return_value=refs,
                    ) as index,
                ):
                    builder(
                        algorithm=algorithm,
                        draft_model=None,
                        target_head=None,
                        optimizer_factory=None,
                        run_id="r",
                        output_dir="/out",
                        batch_size=2,
                        max_len=128,
                        seed=7,
                        length_bucket_size=window,
                        **source,
                    )
                kwargs = assemble.call_args.kwargs
                contract = kwargs["checkpoint_extra"]
                self.assertEqual(
                    contract["offline_sampler_version"], 2 if window else 1
                )
                if window:
                    self.assertEqual(contract["length_bucket_size"], window)
                    self.assertEqual(contract["length_bucket_max_len"], 128)
                    self.assertEqual(
                        contract["length_bucket_ref_fingerprint"],
                        length_bucket_fingerprint(refs, max_len=128),
                    )
                    index.assert_called_once_with(refs, cache_dir="/out/length-cache")
                else:
                    self.assertEqual(
                        contract,
                        {
                            "offline_sampler_version": 1,
                            "sampler_seed": 7,
                            "source_dataset_size": len(refs),
                        },
                    )
                    index.assert_not_called()
                epoch_plan = kwargs["ref_source"]["refs_for_epoch"]
                self.assertEqual(kwargs["ref_source"]["refs"], epoch_plan(0))
                self.assertEqual(epoch_plan(1), epoch_plan(1))
                self.assertNotEqual(epoch_plan(0), epoch_plan(1))


if __name__ == "__main__":
    unittest.main()
