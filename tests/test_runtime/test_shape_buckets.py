"""``training.static_shape_buckets``: a few static lengths instead of one, one compiled graph per bucket."""

import os
import types
import unittest
from unittest import mock

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry

ALGORITHM = builtin_algorithm_registry().resolve("dflash")


def _sample(length, width=4):
    return {
        "input_ids": torch.arange(length).unsqueeze(0),
        "loss_mask": torch.ones(1, length, dtype=torch.long),
        "hidden_states": torch.randn(1, length, width),
    }


class TestBucketResolution(unittest.TestCase):
    def test_resolve_static_length(self):
        from specforge.algorithms.common.collation import resolve_static_length

        self.assertEqual(resolve_static_length(300, None), 300)
        self.assertEqual(resolve_static_length(300, 2048), 2048)
        with self.assertRaisesRegex(ValueError, "static batch length"):
            resolve_static_length(3000, 2048)
        buckets = (512, 1024, 2048)
        self.assertEqual(resolve_static_length(300, buckets), 512)
        self.assertEqual(resolve_static_length(512, buckets), 512)
        self.assertEqual(resolve_static_length(513, buckets), 1024)
        self.assertEqual(resolve_static_length(2048, buckets), 2048)
        with self.assertRaisesRegex(ValueError, "static batch length"):
            resolve_static_length(2049, buckets)
        # Unsorted input is tolerated.
        self.assertEqual(resolve_static_length(700, [2048, 512, 1024]), 1024)

    def test_collators_pad_to_the_smallest_fitting_bucket(self):
        from specforge.algorithms.common.hidden_states_data import build_collator
        from specforge.data.utils import DataCollatorWithPadding

        collate = build_collator(pad_to=(8, 16))
        self.assertEqual(tuple(collate([_sample(5), _sample(3)])["input_ids"].shape), (2, 8))
        self.assertEqual(tuple(collate([_sample(9), _sample(3)])["hidden_states"].shape), (2, 16, 4))

        def item(n):
            ones = torch.ones(1, n, dtype=torch.long)
            return {"input_ids": ones, "attention_mask": ones.clone(), "loss_mask": ones.clone()}

        eagle = DataCollatorWithPadding(pad_to=(8, 16))
        self.assertEqual(tuple(eagle([item(5), item(3)])["input_ids"].shape), (2, 8))
        self.assertEqual(tuple(eagle([item(9)])["input_ids"].shape), (1, 16))

    def test_launch_resolves_buckets(self):
        from specforge.launch import _offline_io, _static_pad_length, _streaming_collate

        self.assertEqual(_static_pad_length(True, 2048, [512, 1024]), (512, 1024, 2048))
        self.assertEqual(_static_pad_length(True, 2048, [512, 2048]), (512, 2048))
        self.assertEqual(_static_pad_length(True, 2048, None), 2048)
        self.assertIsNone(_static_pad_length(False, 2048, [512]))
        with self.assertRaisesRegex(ValueError, "exceed"):
            _static_pad_length(True, 2048, [4096])
        collate = _streaming_collate(ALGORITHM, "text", None, pad_to=(8, 16))
        self.assertEqual(tuple(collate([_sample(9), _sample(2)])["input_ids"].shape), (2, 16))
        collate, _ = _offline_io(
            ALGORITHM, "text", 16, ttt_length=7, use_usp_preprocess=False,
            static_shapes=True, static_shape_buckets=[8],
        )
        self.assertEqual(tuple(collate([_sample(5), _sample(3)])["input_ids"].shape), (2, 8))


class TestBucketConfig(unittest.TestCase):
    def test_training_config_validation(self):
        from specforge.config.schema import TrainingConfig

        with self.assertRaisesRegex(ValueError, "requires training.static_shapes"):
            TrainingConfig(static_shape_buckets=[512])
        with self.assertRaisesRegex(ValueError, "multiples of 128"):
            TrainingConfig(static_shapes=True, static_shape_buckets=[500])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            TrainingConfig(static_shapes=True, static_shape_buckets=[1024, 512])
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            TrainingConfig(static_shapes=True, static_shape_buckets=[])
        cfg = TrainingConfig(static_shapes=True, static_shape_buckets=[512, 1024])
        self.assertEqual(cfg.static_shape_buckets, [512, 1024])

    @staticmethod
    def _config(buckets, max_length=2048, compile_blocks=True):
        from specforge.config import Config

        training = {
            "strategy": "dflash", "backend": "fsdp2", "static_shapes": True,
            "static_shape_buckets": buckets, "compile_blocks": compile_blocks, "max_steps": 1,
        }
        return Config.model_validate(
            {
                "model": {"target_model_path": "t", "draft_model_config": "d"},
                "data": {"hidden_states_path": "features", "max_length": max_length},
                "training": training,
            }
        )

    def test_buckets_must_fit_max_length_and_get_it_appended(self):
        from specforge.training.assembly import _backend_options, shape_buckets

        with self.assertRaisesRegex(ValueError, "must not exceed data.max_length"):
            self._config([512, 4096])
        cfg = self._config([512, 1024])
        self.assertEqual(shape_buckets(cfg), (512, 1024, 2048))
        self.assertEqual(_backend_options(cfg).compile_shape_buckets, 3)
        self.assertTrue(_backend_options(cfg).compile_blocks)
        # Padding buckets without compile_blocks leave the backend options untouched
        # (FSDP1 rejects any set option, and buckets alone are a data-path setting).
        loose = self._config([512, 1024], compile_blocks=False)
        self.assertEqual(shape_buckets(loose), (512, 1024, 2048))
        self.assertEqual(_backend_options(loose).compile_shape_buckets, 0)

    def test_bucketed_compile_raises_the_recompile_limit(self):
        import torch._dynamo

        from specforge.training.fsdp2 import _configure_bucketed_compile

        cfg = torch._dynamo.config
        names = [n for n in ("recompile_limit", "cache_size_limit") if hasattr(cfg, n)]
        self.assertTrue(names)
        saved = {n: getattr(cfg, n) for n in names}
        try:
            for n in names:
                setattr(cfg, n, 8)
            _configure_bucketed_compile(4)
            for n in names:
                self.assertGreaterEqual(getattr(cfg, n), 12)
        finally:
            for n, v in saved.items():
                setattr(cfg, n, v)


class TestDisaggregatedLaunchForwardsBuckets(unittest.TestCase):
    def test_online_consumer_receives_the_buckets(self):
        from specforge.config import Config
        from specforge.training.disaggregated import _build_online

        cfg = Config.model_validate(
            {
                "model": {"target_model_path": "t", "draft_model_config": "d"},
                "data": {"prompts_path": "prompts.jsonl", "max_length": 2048},
                "training": {
                    "strategy": "dflash", "role": "consumer", "max_steps": 1, "backend": "fsdp2",
                    "static_shapes": True, "static_shape_buckets": [512, 1024], "compile_blocks": True,
                },
                "deployment": {
                    "mode": "disaggregated",
                    "disaggregated": {"control_dir": "/shared/buckets", "backend": "mooncake", "server_urls": ["http://capture:30000"]},
                },
            }
        )
        bundle = types.SimpleNamespace(model=object(), target_head=None, strategy_kwargs={})

        class _Trainer:
            def fit(self):
                return 1

        with (
            mock.patch.dict(os.environ, {"DISAGG_REF_CHANNEL": "/shared/refs"}),
            mock.patch("specforge.runtime.data_plane.streaming_ref_channel.StreamingRefChannel"),
            mock.patch("specforge.training.disaggregated._mooncake_store", return_value=mock.Mock()),
            mock.patch("specforge.launch.build_disagg_online_consumer", return_value=_Trainer()) as build,
        ):
            _build_online(
                cfg, algorithm=ALGORITHM, build_model_bundle=lambda _cfg: bundle,
                prepare_prompts=mock.Mock(), optimizer_factory=mock.Mock(), logger=None,
            )
        kw = build.call_args.kwargs
        self.assertEqual(kw["static_shape_buckets"], (512, 1024, 2048))
        self.assertEqual(kw["backend_options"].compile_shape_buckets, 3)


if __name__ == "__main__":
    unittest.main()
