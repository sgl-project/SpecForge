# coding=utf-8
"""CPU tests for the StrategySpec registry + strategy-parameterized launch.

These lock the composition contract introduced by the launch refactor:
  * the registry resolves eagle3 to a spec whose required_features match the
    strategy class, with its offline + online data paths wired;
  * launch builders take a ``strategy=`` parameter and do not expose
    strategy-specific entry points;
  * a strategy whose data path is not wired raises an actionable
    NotImplementedError rather than silently assembling eagle3-shaped features.

No GPU / model environment required.
"""

import os
import tempfile
import unittest

import torch

import specforge.launch as launch
from specforge.inference.adapters.policy import (
    DFLASH_FEATURE_SCHEMA,
    EAGLE3_FEATURE_SCHEMA,
)
from specforge.training.strategies.base import (
    DFlashTrainStrategy,
    DSparkTrainStrategy,
    Eagle3TrainStrategy,
)
from specforge.training.strategies.registry import (
    available_strategies,
    resolve_strategy,
)


class TestStrategyRegistry(unittest.TestCase):
    def test_builtins_registered(self):
        self.assertIn("eagle3", available_strategies())
        self.assertIn("dflash", available_strategies())
        self.assertIn("domino", available_strategies())
        self.assertIn("dspark", available_strategies())

    def test_dflash_offline_and_online_are_fully_wired(self):
        spec = resolve_strategy("dflash")
        self.assertEqual(
            spec.required_features, frozenset(DFlashTrainStrategy.required_features)
        )
        self.assertIsNotNone(spec.make_offline_reader)
        self.assertIsNotNone(spec.make_offline_transform)
        self.assertIsNotNone(spec.make_offline_collate)
        self.assertIsNotNone(spec.feature_schema)
        self.assertIs(spec.feature_schema, DFLASH_FEATURE_SCHEMA)
        self.assertIsNone(spec.feature_schema.target_feature)  # no target distribution
        self.assertTrue(spec.supports_online)
        # DFlash owns its own (frozen) head -> builders pass target_head=None
        self.assertFalse(spec.uses_target_head)

        class _M:
            draft_model = object()

        self.assertIsInstance(
            spec.make_strategy(_M(), target_head=None), DFlashTrainStrategy
        )

    def test_dflash_family_uses_one_feature_contract_offline_and_online(self):
        for name in ("dflash", "domino"):
            with self.subTest(strategy=name):
                spec = resolve_strategy(name)
                self.assertIs(spec.feature_schema, DFLASH_FEATURE_SCHEMA)
                self.assertIsNotNone(spec.make_offline_reader)
                self.assertIsNotNone(spec.make_offline_transform)
                self.assertIsNotNone(spec.make_offline_collate)

    def test_eagle3_uses_eagle3_feature_schema(self):
        self.assertIs(resolve_strategy("eagle3").feature_schema, EAGLE3_FEATURE_SCHEMA)

    def test_dspark_is_server_capture_online_only(self):
        spec = resolve_strategy("dspark")
        self.assertEqual(
            spec.required_features, frozenset(DSparkTrainStrategy.required_features)
        )
        self.assertTrue(spec.supports_online)
        self.assertIsNone(spec.make_offline_reader)
        self.assertIsNone(spec.feature_schema)
        self.assertFalse(spec.uses_target_head)
        with self.assertRaisesRegex(
            NotImplementedError, "requires a server feature source"
        ):
            launch.build_online_runtime(
                strategy="dspark",
                target_model=None,
                prompts=[],
                draft_model=None,
                optimizer_factory=None,
                run_id="dspark-server-guard",
                output_dir="unused",
                target_hidden_size=8,
            )

        class _M:
            draft_model = object()

        self.assertIsInstance(
            spec.make_strategy(_M(), target_head=None), DSparkTrainStrategy
        )

    def test_required_features_track_the_strategy_class(self):
        # The registry's required_features is the single source of truth wired
        # into CaptureConfig.from_strategy + loader validation; it must equal the
        # strategy class's own declaration.
        self.assertEqual(
            resolve_strategy("eagle3").required_features,
            frozenset(Eagle3TrainStrategy.required_features),
        )

    def test_dflash_and_domino_online_collate_pad_ragged_sequences(self):
        short = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "loss_mask": torch.tensor([[1, 1, 1]]),
            "hidden_states": torch.ones(1, 3, 4),
        }
        long = {
            "input_ids": torch.tensor([[4, 5, 6, 7, 8]]),
            "loss_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "hidden_states": torch.ones(1, 5, 4),
        }
        for name in ("dflash", "domino"):
            with self.subTest(strategy=name):
                batch = resolve_strategy(name).make_online_collate()([short, long])
                self.assertEqual(batch["input_ids"].shape, (2, 5))
                self.assertEqual(batch["hidden_states"].shape, (2, 5, 4))
                self.assertEqual(batch["loss_mask"][0].tolist(), [1, 1, 1, 0, 0])
                self.assertEqual(batch["input_ids"][0].tolist(), [1, 2, 3, 0, 0])
                self.assertTrue(torch.all(batch["hidden_states"][0, 3:] == 0))

    def test_dflash_and_domino_offline_transform_preserves_capture_contract(self):
        raw = {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "loss_mask": torch.tensor([1, 1, 1, 1]),
            "hidden_states": torch.arange(24).reshape(1, 4, 6),
        }
        for name in ("dflash", "domino"):
            with self.subTest(strategy=name):
                transformed = resolve_strategy(name).make_offline_transform(3)(raw)
                self.assertEqual(
                    set(transformed), {"input_ids", "loss_mask", "hidden_states"}
                )
                self.assertEqual(transformed["input_ids"].shape, (1, 3))
                self.assertEqual(transformed["loss_mask"].tolist(), [[1, 1, 1]])
                self.assertEqual(transformed["hidden_states"].shape, (1, 3, 6))
                self.assertTrue(
                    torch.equal(
                        transformed["hidden_states"], raw["hidden_states"][:, :3]
                    )
                )

    def test_dflash_family_offline_reader_stamps_strategy_specific_refs(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.save(
                {
                    "input_ids": torch.tensor([1, 2, 3]),
                    "loss_mask": torch.ones(3, dtype=torch.long),
                    "hidden_states": torch.ones(1, 3, 4),
                },
                os.path.join(directory, "0000.ckpt"),
            )
            for name in ("dflash", "domino"):
                with self.subTest(strategy=name):
                    reader = resolve_strategy(name).make_offline_reader(
                        directory,
                        run_id=f"{name}-run",
                        ttt_length=7,
                        max_len=16,
                    )
                    refs = reader.read()
                    self.assertEqual(len(refs), 1)
                    self.assertEqual(refs[0].strategy, name)
                    self.assertEqual(
                        set(refs[0].feature_keys),
                        {"input_ids", "loss_mask", "hidden_states"},
                    )
                    self.assertEqual(refs[0].metadata["format"], f"offline_{name}")
                    self.assertIsNone(refs[0].metadata["target_repr"])

    def test_disaggregated_ingest_uses_domino_offline_contract(self):
        from specforge.runtime.data_plane import LocalFeatureStore
        from specforge.runtime.data_plane.disagg_ingest import ingest_offline_features

        with tempfile.TemporaryDirectory() as directory:
            tensors = {
                "input_ids": torch.tensor([1, 2, 3]),
                "loss_mask": torch.ones(3, dtype=torch.long),
                "hidden_states": torch.ones(1, 3, 4),
            }
            torch.save(tensors, os.path.join(directory, "0000.ckpt"))
            store = LocalFeatureStore("offline-domino")
            refs = ingest_offline_features(
                store,
                directory,
                strategy="domino",
                run_id="domino-run",
            )
            self.assertEqual(len(refs), 1)
            self.assertEqual(refs[0].strategy, "domino")
            self.assertEqual(refs[0].metadata["format"], "offline_domino")
            loaded, handle = store.get(refs[0])
            self.addCleanup(store.release, handle)
            self.assertEqual(set(loaded), set(tensors))
            for key, value in tensors.items():
                self.assertTrue(torch.equal(loaded[key], value))

    def test_dspark_online_collate_pads_target_last_hidden(self):
        short = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "loss_mask": torch.tensor([[1, 1, 1]]),
            "hidden_states": torch.ones(1, 3, 4),
            "target_last_hidden_states": torch.ones(1, 3, 2),
        }
        long = {
            "input_ids": torch.tensor([[4, 5, 6, 7, 8]]),
            "loss_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "hidden_states": torch.ones(1, 5, 4),
            "target_last_hidden_states": torch.ones(1, 5, 2),
        }
        batch = resolve_strategy("dspark").make_online_collate()([short, long])
        self.assertEqual(batch["input_ids"].shape, (2, 5))
        self.assertEqual(batch["hidden_states"].shape, (2, 5, 4))
        self.assertEqual(batch["target_last_hidden_states"].shape, (2, 5, 2))
        self.assertTrue(torch.all(batch["target_last_hidden_states"][0, 3:] == 0))

    def test_eagle3_is_fully_wired(self):
        spec = resolve_strategy("eagle3")
        self.assertTrue(spec.supports_online)
        self.assertIsNotNone(spec.make_offline_reader)
        self.assertIsNotNone(spec.make_offline_transform)
        self.assertIsNotNone(spec.make_offline_collate)
        self.assertTrue(spec.uses_target_head)

    def test_resolve_unknown_strategy_raises(self):
        with self.assertRaises(KeyError):
            resolve_strategy("does-not-exist")

    def test_make_strategy_factory_constructs_the_right_class(self):
        class _M:
            draft_model = object()

        self.assertIsInstance(
            resolve_strategy("eagle3").make_strategy(_M(), target_head=None),
            Eagle3TrainStrategy,
        )


class TestLaunchGuards(unittest.TestCase):
    def test_offline_builder_rejects_unwired_strategy(self):
        # A strategy with no offline reader must fail fast and actionably, before
        # any model/controller is touched. Register a throwaway unwired spec so
        # this stays valid regardless of which built-ins wire their offline path.
        from specforge.training.strategies import registry as _reg
        from specforge.training.strategies.registry import (
            StrategySpec,
            register_strategy,
        )

        register_strategy(
            StrategySpec(
                name="_unwired_offline_test",
                required_features=frozenset({"x"}),
                make_strategy=lambda wrapped, *, target_head=None: None,
            )
        )
        self.addCleanup(_reg._REGISTRY.pop, "_unwired_offline_test", None)
        with self.assertRaises(NotImplementedError):
            launch.build_offline_runtime(
                strategy="_unwired_offline_test",
                hidden_states_path="unused",
                draft_model=None,
                target_head=None,
                optimizer_factory=None,
                run_id="r",
                output_dir="o",
            )

    def test_online_builder_rejects_unwired_strategy(self):
        # A strategy without an online capture path must fail fast and actionably.
        from specforge.training.strategies import registry as _reg
        from specforge.training.strategies.registry import (
            StrategySpec,
            register_strategy,
        )

        register_strategy(
            StrategySpec(
                name="_unwired_online_test",
                required_features=frozenset({"x"}),
                make_strategy=lambda wrapped, *, target_head=None: None,
                supports_online=False,
            )
        )
        self.addCleanup(_reg._REGISTRY.pop, "_unwired_online_test", None)
        with self.assertRaises(NotImplementedError):
            launch.build_online_runtime(
                strategy="_unwired_online_test",
                target_model=None,
                prompts=[],
                draft_model=None,
                optimizer_factory=None,
                run_id="r-online",
                output_dir="o",
                target_hidden_size=8,
            )


if __name__ == "__main__":
    unittest.main()
