# coding=utf-8
"""Config-to-assembly reachability for retained training capabilities."""

import unittest
from pathlib import Path
from unittest import mock

import yaml
from pydantic import ValidationError

from specforge.application import resolve_run
from specforge.config import Config, flatten_drafter
from specforge.training.assembly import (
    _configured_logger,
    _dataloader_num_workers,
    _logger,
    _profiling_options,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG_DIR = REPO_ROOT / "examples" / "configs"

OFFLINE_EAGLE3 = {
    "draft": "specforge/config/draft/presets/llama3-8B-eagle3.yaml",
    "model": {
        "target_model_path": "target",
        "vocab_mapping_path": "mapping.pt",
    },
    "data": {"hidden_states_path": "features"},
}


class UnifiedFeatureReachabilityTest(unittest.TestCase):
    def test_all_example_configs_validate_without_remote_target_downloads(self):
        paths = sorted(
            path
            for path in EXAMPLE_CONFIG_DIR.rglob("*.yaml")
            if not path.name.startswith(".")
        )
        self.assertTrue(paths)

        configs = []
        for path in paths:
            recipe = Config.from_file(str(path))
            # A drafter is checked as the single-draft run it flattens to.
            names = [drafter.name for drafter in recipe.drafters or ()]
            configs += [
                (f"{path.name}#{name}", flatten_drafter(recipe, name)) for name in names
            ] or [(path.name, recipe)]
        for label, config in configs:
            with self.subTest(config=label):
                # Native MTP presets use matching local metadata, without a Hub call.
                target = dict(
                    model_type="qwen3_5",
                    hidden_size=2560,
                    vocab_size=248320,
                    intermediate_size=9216,
                    hidden_act="silu",
                    rms_norm_eps=1e-6,
                    num_attention_heads=16,
                    num_key_value_heads=4,
                    head_dim=256,
                    max_position_embeddings=262144,
                    attn_output_gate=True,
                    rope_parameters=dict(
                        rope_type="mrope",
                        rope_theta=10000000.0,
                        partial_rotary_factor=0.25,
                        mrope_section=[11, 11, 10],
                        mrope_interleaved=True,
                    ),
                )
                if config.draft:
                    payload = (
                        yaml.safe_load((REPO_ROOT / config.draft).read_text())
                        if isinstance(config.draft, str)
                        else config.draft
                    )
                    if payload["architectures"] != ["MTPDraftModel"]:
                        features = payload["spec_config"]["target_features"]
                        target = {
                            "hidden_size": features["hidden_size"],
                            "vocab_size": payload["vocab_size"],
                            "num_hidden_layers": max(
                                features.get("target_layer_ids", [0])
                            )
                            + 2,
                        }
                        transform = payload.get("target_logit_transform", {})
                        if transform.get("type") == "mup":
                            target["logits_mup_width_multiplier"] = transform[
                                "multiplier"
                            ]
                resolve_run(config, target_config=target)

    def test_compact_teacher_reaches_the_eagle3_step_provider(self):
        cfg = Config.model_validate(
            {
                **OFFLINE_EAGLE3,
                "training": {
                    "loss": {"execution": {"trim": True, "teacher_chunk_size": 2048}},
                },
            }
        )
        resolved = resolve_run(cfg)

        self.assertEqual(
            resolved.config.training.loss["execution"],
            {"trim": True, "teacher_chunk_size": 2048},
        )

    def test_trim_loss_positions_reaches_block_objectives(self):
        cfg = Config.model_validate(
            {
                **OFFLINE_EAGLE3,
                "model": {"target_model_path": "target"},
                "draft": "specforge/config/draft/presets/qwen3-8b-dflash.yaml",
                "training": {
                    "loss": {"execution": {"trim": True}},
                },
            }
        )

        resolved = resolve_run(cfg)
        self.assertTrue(resolved.config.training.loss["execution"]["trim"])

    def test_loader_and_profiler_options_reach_the_canonical_trainer(self):
        eagle = resolve_run(Config.model_validate(OFFLINE_EAGLE3))
        dflash = resolve_run(
            Config.model_validate(
                {
                    **OFFLINE_EAGLE3,
                    "draft": "specforge/config/draft/presets/qwen3-8b-dflash.yaml",
                    "model": {"target_model_path": "target"},
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
        # the block objectives declare the loader default they were tuned with
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
            "training": {
                "loss": {"execution": {"teacher_chunk_size": 32768}},
                "max_steps": 1,
            },
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
            ValueError, "compact teacher currently requires offline"
        ):
            resolve_run(Config.model_validate(online))

        with self.assertRaisesRegex(
            ValueError, "compact teacher requires a vocabulary-aware protocol"
        ):
            resolve_run(
                Config.model_validate(
                    {
                        **OFFLINE_EAGLE3,
                        "draft": "specforge/config/draft/presets/qwen3-8b-dflash.yaml",
                        "model": {"target_model_path": "target"},
                        "training": {
                            "loss": {"execution": {"teacher_chunk_size": 32768}},
                        },
                    }
                )
            )

        with self.assertRaisesRegex(ValidationError, "compact_teacher_chunk_size"):
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
            args.specforge_config["training"]["objective"],
            cfg.training.objective,
        )
        self.assertEqual(output_dir, "/tmp/output")
        self.assertIs(create.call_args.kwargs["console_logger"], _logger)

    def test_only_global_rank_zero_creates_an_external_tracker(self):
        cfg = Config.model_validate(
            {**OFFLINE_EAGLE3, "tracking": {"report_to": "wandb"}}
        )
        with (
            mock.patch("specforge.training.tracking.create_tracker_logger") as create,
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_rank", return_value=3),
        ):
            self.assertIs(_configured_logger(cfg), _logger)
        create.assert_not_called()

    def test_primary_rank_keeps_the_console_logger_without_a_tracker(self):
        cfg = Config.model_validate(OFFLINE_EAGLE3)
        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_rank", return_value=0),
        ):
            self.assertIs(_configured_logger(cfg), _logger)

    def test_tracking_backend_is_strictly_typed(self):
        with self.assertRaises(ValidationError):
            Config.model_validate(
                {**OFFLINE_EAGLE3, "tracking": {"report_to": "unknown"}}
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
