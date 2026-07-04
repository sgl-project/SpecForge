# coding=utf-8
"""CPU tests for the StrategySpec registry + strategy-parameterized launch.

These lock the composition contract introduced by the launch refactor:
  * the registry resolves eagle3 to a spec whose required_features match the
    strategy class, with its offline + online data paths wired;
  * launch builders take a `strategy=` parameter (default "eagle3") and the old
    eagle3-named builders are exact aliases of the strategy-neutral ones;
  * a strategy whose data path is not wired raises an actionable
    NotImplementedError rather than silently assembling eagle3-shaped features.

No GPU / model environment required.
"""

import unittest

from specforge.runtime import launch
from specforge.training.strategies.base import DFlashTrainStrategy, Eagle3TrainStrategy
from specforge.training.strategies.registry import (
    available_strategies,
    resolve_strategy,
)


class TestStrategyRegistry(unittest.TestCase):
    def test_builtins_registered(self):
        self.assertIn("eagle3", available_strategies())
        self.assertIn("dflash", available_strategies())

    def test_dflash_fully_wired(self):
        spec = resolve_strategy("dflash")
        self.assertEqual(
            spec.required_features, frozenset(DFlashTrainStrategy.required_features)
        )
        # offline (reader/transform/collate) + online (adapter) both wired
        self.assertIsNotNone(spec.make_offline_reader)
        self.assertIsNotNone(spec.make_offline_transform)
        self.assertIsNotNone(spec.make_offline_collate)
        self.assertIsNotNone(spec.make_adapter)
        self.assertTrue(spec.supports_online)
        # DFlash owns its own (frozen) head -> builders pass target_head=None
        self.assertFalse(spec.uses_target_head)

        class _M:
            draft_model = object()

        self.assertIsInstance(
            spec.make_strategy(_M(), target_head=None), DFlashTrainStrategy
        )

    def test_required_features_track_the_strategy_class(self):
        # The registry's required_features is the single source of truth wired
        # into CaptureConfig.from_strategy + loader validation; it must equal the
        # strategy class's own declaration.
        self.assertEqual(
            resolve_strategy("eagle3").required_features,
            frozenset(Eagle3TrainStrategy.required_features),
        )

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


class TestLaunchBackCompatAndGuards(unittest.TestCase):
    def test_eagle3_named_builders_are_aliases_of_neutral_ones(self):
        self.assertIs(launch.build_offline_eagle3_runtime, launch.build_offline_runtime)
        self.assertIs(
            launch.build_offline_eagle3_controller, launch.build_offline_runtime
        )
        self.assertIs(
            launch.build_disagg_eagle3_runtime, launch.build_disagg_offline_runtime
        )
        self.assertIs(launch.build_online_eagle3_runtime, launch.build_online_runtime)
        self.assertIs(
            launch.build_disagg_online_eagle3_runtime,
            launch.build_disagg_online_runtime,
        )

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
                eagle3_model=None,
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
                eagle3_model=None,
                optimizer_factory=None,
                run_id="r-online",
                output_dir="o",
                target_hidden_size=8,
            )


if __name__ == "__main__":
    unittest.main()
