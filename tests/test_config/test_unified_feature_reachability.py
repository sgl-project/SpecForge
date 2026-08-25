# coding=utf-8
"""Config-to-assembly reachability for retained training capabilities."""

import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

from specforge.application import resolve_run
from specforge.config import Config
from specforge.training.assembly import (
    _configured_logger,
    _dataloader_num_workers,
    _logger,
    _profiling_options,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG_DIR = REPO_ROOT / "examples" / "configs"

OFFLINE_EAGLE3 = {
    "model": {
        "target_model_path": "target",
        "draft_model_config": "draft.json",
        "vocab_mapping_path": "mapping.pt",
    },
    "data": {"hidden_states_path": "features"},
}


class UnifiedFeatureReachabilityTest(unittest.TestCase):
    def test_all_example_configs_validate_through_the_typed_entry(self):
        paths = sorted(
            path
            for path in EXAMPLE_CONFIG_DIR.rglob("*.yaml")
            if not path.name.startswith(".")
        )
        self.assertTrue(paths)

        for path in paths:
            with self.subTest(config=path.name):
                resolve_run(Config.from_file(str(path)))

    def test_compact_teacher_reaches_the_eagle3_step_provider(self):
        cfg = Config.model_validate(
            {
                **OFFLINE_EAGLE3,
                "training": {
                    "trim_loss_positions": True,
                    "compact_teacher": True,
                    "compact_teacher_chunk_size": 2048,
                },
            }
        )
        resolved = resolve_run(cfg)

        self.assertEqual(
            resolved.algorithm.providers.step.options(cfg),
            {
                "trim_loss_positions": True,
                "compact_teacher": True,
                "compact_teacher_chunk_size": 2048,
            },
        )

    def test_trim_loss_positions_rejects_non_eagle3_strategy(self):
        cfg = Config.model_validate(
            {
                **OFFLINE_EAGLE3,
                "training": {
                    "strategy": "dflash",
                    "trim_loss_positions": True,
                },
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "algorithm 'dflash' does not support training.trim_loss_positions",
        ):
            resolve_run(cfg)

    def test_loader_and_profiler_options_reach_the_canonical_trainer(self):
        eagle = resolve_run(Config.model_validate(OFFLINE_EAGLE3))
        dflash = resolve_run(
            Config.model_validate(
                {
                    **OFFLINE_EAGLE3,
                    "training": {"strategy": "dflash"},
                }
            )
        )
        explicit = resolve_run(
            Config.model_validate(
                {
                    **OFFLINE_EAGLE3,
                    "data": {
                        **OFFLINE_EAGLE3["data"],
                        "dataloader_num_workers": 2,
                    },
                    "profiling": {
                        "enabled": True,
                        "start_step": 3,
                        "num_steps": 2,
                        "record_shapes": True,
                    },
                }
            )
        )

        self.assertEqual(_dataloader_num_workers(eagle.config, eagle.algorithm), 4)
        self.assertEqual(_dataloader_num_workers(dflash.config, dflash.algorithm), 8)
        self.assertEqual(
            _dataloader_num_workers(explicit.config, explicit.algorithm), 2
        )
        options = _profiling_options(explicit.config)
        self.assertTrue(options.enabled)
        self.assertEqual((options.start_step, options.num_steps), (3, 2))
        self.assertTrue(options.record_shapes)

    def test_compact_teacher_rejects_incompatible_entry_configs(self):
        online = {
            **OFFLINE_EAGLE3,
            "model": {
                **OFFLINE_EAGLE3["model"],
                "target_backend": "sglang",
            },
            "data": {"train_data_path": "train.jsonl"},
            "training": {"compact_teacher": True, "max_steps": 1},
            "deployment": {
                "mode": "disaggregated",
                "disaggregated": {
                    "control_dir": "/control",
                    "backend": "mooncake",
                    "server_urls": ["http://127.0.0.1:30000"],
                },
            },
        }
        with self.assertRaisesRegex(
            ValueError, "does not support compact teacher for mode='streaming'"
        ):
            resolve_run(Config.model_validate(online))

        with self.assertRaisesRegex(
            ValueError, "algorithm 'dflash' does not support compact teacher"
        ):
            resolve_run(
                Config.model_validate(
                    {
                        **OFFLINE_EAGLE3,
                        "training": {
                            "strategy": "dflash",
                            "compact_teacher": True,
                        },
                    }
                )
            )

        with self.assertRaisesRegex(
            ValidationError, "requires training.compact_teacher=true"
        ):
            Config.model_validate(
                {
                    **OFFLINE_EAGLE3,
                    "training": {"compact_teacher_chunk_size": 1024},
                }
            )

    def test_tracking_config_reaches_the_existing_tracker_adapter(self):
        cfg = Config.model_validate(
            {
                **OFFLINE_EAGLE3,
                "tracking": {
                    "report_to": "wandb",
                    "wandb_project": "project",
                    "wandb_name": "experiment",
                    "wandb_offline": True,
                    "wandb_dir": "/tmp/wandb",
                },
                "run_id": "run",
                "output_dir": "/tmp/output",
            }
        )
        tracker_logger = object()

        with mock.patch(
            "specforge.training.tracking.create_tracker_logger",
            return_value=tracker_logger,
        ) as create:
            result = _configured_logger(cfg)

        self.assertIs(result, tracker_logger)
        args, output_dir = create.call_args.args
        self.assertEqual(args.report_to, "wandb")
        self.assertEqual(args.wandb_project, "project")
        self.assertEqual(args.wandb_name, "experiment")
        self.assertTrue(args.wandb_offline)
        self.assertEqual(args.wandb_dir, "/tmp/wandb")
        self.assertEqual(args.specforge_config["run_id"], "run")
        self.assertEqual(
            args.specforge_config["training"]["strategy"],
            cfg.training.strategy,
        )
        self.assertEqual(output_dir, "/tmp/output")
        self.assertIs(create.call_args.kwargs["console_logger"], _logger)

    def test_tracking_backend_is_strictly_typed(self):
        with self.assertRaises(ValidationError):
            Config.model_validate(
                {**OFFLINE_EAGLE3, "tracking": {"report_to": "unknown"}}
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
