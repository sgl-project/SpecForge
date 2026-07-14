# coding=utf-8
"""Config schema mechanics: parsing, validation, dotted overrides."""

import copy
import json
import os
import tempfile
import unittest

from pydantic import ValidationError

from specforge.config import Config, apply_overrides, load_config

MINIMAL = {
    "model": {
        "target_model_path": "some/target",
        "draft_model_config": "draft.json",
        "vocab_mapping_path": "/mapping.pt",
    },
    "data": {"hidden_states_path": "/features"},
}


def _write(payload: dict, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        if suffix == ".yaml":
            import yaml

            yaml.safe_dump(payload, f)
        else:
            json.dump(payload, f)
    return path


class ConfigSchemaTest(unittest.TestCase):
    def test_qwen_vl_uses_online_eagle3_without_a_second_entry(self):
        payload = {
            **MINIMAL,
            "model": {
                **MINIMAL["model"],
                "input_modality": "qwen2_5_vl",
                "target_backend": "hf",
            },
            "data": {
                "train_data_path": "/vlm.jsonl",
                "chat_template": "qwen2-vl",
                "min_pixels": 50176,
                "max_pixels": 802816,
            },
        }
        cfg = Config.model_validate(payload)
        self.assertEqual(cfg.model.input_modality, "qwen2_5_vl")
        self.assertEqual(cfg.mode, "online")

        invalid = copy.deepcopy(payload)
        invalid["data"] = {"hidden_states_path": "/features"}
        with self.assertRaisesRegex(ValidationError, "online target capture"):
            Config.model_validate(invalid)

        invalid = copy.deepcopy(payload)
        invalid["training"] = {"strategy": "dflash"}
        with self.assertRaisesRegex(ValidationError, "does not support input_modality"):
            Config.model_validate(invalid)

    def test_target_output_sharding_is_a_colocated_text_sglang_option(self):
        payload = {
            **MINIMAL,
            "model": {
                **MINIMAL["model"],
                "target_backend": "sglang",
                "shard_target_output": True,
            },
            "data": {"train_data_path": "/train.jsonl"},
        }
        cfg = Config.model_validate(payload)
        self.assertTrue(cfg.model.shard_target_output)

        invalid = copy.deepcopy(payload)
        invalid["model"]["target_backend"] = "hf"
        with self.assertRaisesRegex(ValidationError, "target_backend=sglang"):
            Config.model_validate(invalid)

        invalid = copy.deepcopy(payload)
        invalid["model"]["input_modality"] = "qwen2_5_vl"
        invalid["data"] = {
            "train_data_path": "/vlm.jsonl",
            "chat_template": "qwen2-vl",
        }
        with self.assertRaisesRegex(ValidationError, "not supported for VLM"):
            Config.model_validate(invalid)

    def test_sglang_expert_parallelism_must_fit_target_tensor_parallelism(self):
        payload = {
            **MINIMAL,
            "model": {
                **MINIMAL["model"],
                "target_backend": "sglang",
                "sglang_ep_size": 2,
            },
            "training": {"tp_size": 4},
        }
        config = Config.model_validate(payload)
        self.assertEqual(config.model.sglang_ep_size, 2)

        invalid = copy.deepcopy(payload)
        invalid["model"]["sglang_ep_size"] = 3
        with self.assertRaisesRegex(ValidationError, "evenly divide"):
            Config.model_validate(invalid)

        invalid = copy.deepcopy(payload)
        invalid["model"]["sglang_ep_size"] = 8
        with self.assertRaisesRegex(ValidationError, "no larger"):
            Config.model_validate(invalid)

    def test_online_eagle3_preserves_multi_sample_batches(self):
        payload = {
            **MINIMAL,
            "data": {"train_data_path": "/train.jsonl"},
            "training": {"strategy": "eagle3", "batch_size": 4},
        }
        cfg = Config.model_validate(payload)
        self.assertEqual(cfg.training.batch_size, 4)

    def test_from_file_yaml_and_json(self):
        for suffix in (".yaml", ".json"):
            path = _write(MINIMAL, suffix)
            self.addCleanup(os.unlink, path)
            cfg = Config.from_file(path)
            self.assertEqual(cfg.model.target_model_path, "some/target")
            self.assertEqual(cfg.mode, "offline")
            self.assertEqual(cfg.training.strategy, "eagle3")  # defaults applied

    def test_exactly_one_data_source(self):
        with self.assertRaises(ValidationError):
            Config.model_validate({**MINIMAL, "data": {}})
        with self.assertRaises(ValidationError):
            Config.model_validate(
                {
                    **MINIMAL,
                    "data": {"hidden_states_path": "/a", "prompts_path": "/b"},
                }
            )
        online = Config.model_validate(
            {**MINIMAL, "data": {"prompts_path": "/prompts.jsonl"}}
        )
        self.assertEqual(online.mode, "online")
        raw = Config.model_validate(
            {**MINIMAL, "data": {"train_data_path": "/conversations.jsonl"}}
        )
        self.assertEqual(raw.mode, "online")

    def test_eval_source_and_interval_form_one_mode_matched_pair(self):
        offline = Config.model_validate(
            {
                **MINIMAL,
                "data": {
                    "hidden_states_path": "/train-features",
                    "eval_hidden_states_path": "/eval-features",
                },
                "training": {"eval_interval": 10},
            }
        )
        self.assertEqual(offline.data.eval_hidden_states_path, "/eval-features")
        self.assertEqual(offline.training.eval_interval, 10)

        online = Config.model_validate(
            {
                **MINIMAL,
                "data": {
                    "train_data_path": "/train.jsonl",
                    "eval_data_path": "/eval.jsonl",
                },
                "training": {"eval_interval": 5},
            }
        )
        self.assertEqual(online.data.eval_data_path, "/eval.jsonl")

        for data, training in (
            (
                {
                    "hidden_states_path": "/train",
                    "eval_hidden_states_path": "/eval",
                },
                {},
            ),
            ({"hidden_states_path": "/train"}, {"eval_interval": 2}),
        ):
            with self.subTest(data=data, training=training):
                with self.assertRaisesRegex(ValidationError, "configured together"):
                    Config.model_validate(
                        {**MINIMAL, "data": data, "training": training}
                    )

        with self.assertRaisesRegex(ValidationError, "at most one"):
            Config.model_validate(
                {
                    **MINIMAL,
                    "data": {
                        "hidden_states_path": "/train",
                        "eval_data_path": "/eval.jsonl",
                        "eval_hidden_states_path": "/eval-features",
                    },
                    "training": {"eval_interval": 2},
                }
            )

    def test_eval_mode_must_match_training_and_online_disagg_is_rejected(self):
        with self.assertRaisesRegex(ValidationError, "online training data source"):
            Config.model_validate(
                {
                    **MINIMAL,
                    "data": {
                        "hidden_states_path": "/train-features",
                        "eval_data_path": "/eval.jsonl",
                    },
                    "training": {"eval_interval": 2},
                }
            )
        with self.assertRaisesRegex(ValidationError, "offline training data source"):
            Config.model_validate(
                {
                    **MINIMAL,
                    "data": {
                        "train_data_path": "/train.jsonl",
                        "eval_hidden_states_path": "/eval-features",
                    },
                    "training": {"eval_interval": 2},
                }
            )
        with self.assertRaisesRegex(ValidationError, "online disaggregated evaluation"):
            Config.model_validate(
                {
                    **MINIMAL,
                    "data": {
                        "train_data_path": "/train.jsonl",
                        "eval_data_path": "/eval.jsonl",
                    },
                    "training": {
                        "strategy": "dflash",
                        "deployment_mode": "disaggregated",
                        "role": "consumer",
                        "total_steps": 10,
                        "eval_interval": 2,
                    },
                }
            )

    def test_local_offline_eagle3_can_derive_vocab_mapping(self):
        missing = copy.deepcopy(MINIMAL)
        missing["model"].pop("vocab_mapping_path")
        config = Config.model_validate(missing)
        self.assertEqual(config.mode, "offline")
        self.assertEqual(config.model.vocab_mapping_path, "")
        self.assertEqual(Config.model_validate(MINIMAL).mode, "offline")

    def test_disaggregated_eagle3_requires_shared_vocab_mapping(self):
        missing = copy.deepcopy(MINIMAL)
        missing["model"].pop("vocab_mapping_path")
        missing["training"] = {
            "deployment_mode": "disaggregated",
            "role": "consumer",
        }
        with self.assertRaisesRegex(ValidationError, "vocab_mapping_path"):
            Config.model_validate(missing)

    def test_resume_allows_trainers_and_rejects_producer(self):
        local = Config.model_validate(
            {
                **MINIMAL,
                "training": {"resume_from": "/checkpoints/run-latest"},
            }
        )
        self.assertEqual(local.training.resume_from, "/checkpoints/run-latest")

        local_online = Config.model_validate(
            {
                **MINIMAL,
                "data": {"train_data_path": "/data.jsonl"},
                "training": {"resume_from": "/checkpoints/run-latest"},
            }
        )
        self.assertEqual(local_online.mode, "online")
        self.assertEqual(local_online.training.resume_from, "/checkpoints/run-latest")

        disagg_consumer = Config.model_validate(
            {
                **MINIMAL,
                "data": {"train_data_path": "/data.jsonl"},
                "training": {
                    "strategy": "dflash",
                    "deployment_mode": "disaggregated",
                    "role": "consumer",
                    "total_steps": 10,
                    "resume_from": "/checkpoints/run-latest",
                },
            }
        )
        self.assertEqual(
            disagg_consumer.training.resume_from, "/checkpoints/run-latest"
        )
        with self.assertRaisesRegex(ValidationError, "trainer role"):
            Config.model_validate(
                {
                    **MINIMAL,
                    "training": {
                        "deployment_mode": "disaggregated",
                        "role": "producer",
                        "resume_from": "/checkpoints/run-latest",
                    },
                }
            )

    def test_unknown_backend_rejected(self):
        bad = {**MINIMAL, "model": {**MINIMAL["model"], "target_backend": "vllm"}}
        with self.assertRaises(ValidationError):
            Config.model_validate(bad)

    def test_unknown_fields_and_unsupported_modes_fail_early(self):
        with self.assertRaises(ValidationError):
            Config.model_validate(
                {
                    **MINIMAL,
                    "data": {"train_data_path": "/data.jsonl", "is_vlm": True},
                }
            )
        with self.assertRaises(ValidationError):
            Config.model_validate(
                {
                    **MINIMAL,
                    "training": {"attention_backend": "unknown_backend"},
                }
            )
        with self.assertRaises(ValidationError):
            Config.model_validate(
                {
                    **MINIMAL,
                    "training": {"not_a_training_field": 1},
                }
            )
        offline_dflash = Config.model_validate(
            {**MINIMAL, "training": {"strategy": "dflash"}}
        )
        self.assertEqual(offline_dflash.mode, "offline")
        with self.assertRaises(ValidationError):
            Config.model_validate(
                {
                    **MINIMAL,
                    "training": {"deployment_mode": "dataflow_colocated"},
                }
            )
        dspark = Config.model_validate(
            {
                **MINIMAL,
                "data": {"train_data_path": "/data.jsonl"},
                "training": {
                    "strategy": "dspark",
                    "deployment_mode": "disaggregated",
                    "role": "producer",
                    "total_steps": 10,
                },
            }
        )
        self.assertEqual(dspark.training.strategy, "dspark")
        with self.assertRaisesRegex(ValidationError, "target_backend=sglang"):
            Config.model_validate(
                {
                    **MINIMAL,
                    "model": {**MINIMAL["model"], "target_backend": "hf"},
                    "data": {"train_data_path": "/data.jsonl"},
                    "training": {
                        "strategy": "dflash",
                        "deployment_mode": "disaggregated",
                        "role": "producer",
                        "total_steps": 10,
                    },
                }
            )

    def test_registered_strategy_needs_no_schema_edit(self):
        from specforge.training.strategies import registry as strategy_registry
        from specforge.training.strategies.assembly import (
            DraftConfigSpec,
            StrategyAssemblySpec,
            StrategyModelParts,
        )
        from specforge.training.strategies.registry import (
            StrategySpec,
            register_strategy,
        )

        name = "_config_registry_test"
        register_strategy(
            StrategySpec(
                name=name,
                required_features=frozenset({"input_ids"}),
                make_strategy=lambda wrapped, *, target_head=None: None,
                assembly=StrategyAssemblySpec(
                    draft_config=DraftConfigSpec(architecture="UnusedDraft"),
                    make_draft_model=lambda cfg, draft_config: None,
                    make_model=(
                        lambda cfg, draft, draft_config, target_config, tokenizer: (
                            StrategyModelParts(model=None)
                        )
                    ),
                ),
                supported_modes=frozenset({"offline"}),
                supported_attention_backends=frozenset({"flex_attention"}),
            )
        )
        self.addCleanup(strategy_registry._REGISTRY.pop, name, None)

        payload = copy.deepcopy(MINIMAL)
        payload["training"] = {"strategy": name}
        cfg = Config.model_validate(payload)
        self.assertEqual(cfg.training.strategy, name)

    def test_strategy_specific_capture_and_attention_are_validated(self):
        online = {**MINIMAL, "data": {"train_data_path": "/data.jsonl"}}
        with self.assertRaisesRegex(ValidationError, "exactly three"):
            Config.model_validate(
                {
                    **online,
                    "model": {
                        **online["model"],
                        "aux_hidden_state_layer_ids": [1, 2],
                    },
                }
            )

        with self.assertRaisesRegex(ValidationError, "would be ignored"):
            Config.model_validate(
                {
                    **online,
                    "model": {
                        **online["model"],
                        "aux_hidden_state_layer_ids": [1, 2, 3],
                    },
                    "training": {"strategy": "dflash"},
                }
            )
        with self.assertRaisesRegex(
            ValidationError, "does not support attention_backend"
        ):
            Config.model_validate(
                {
                    **online,
                    "training": {
                        "strategy": "peagle",
                        "attention_backend": "sdpa",
                    },
                }
            )
        with self.assertRaisesRegex(
            ValidationError, "does not support attention_backend"
        ):
            Config.model_validate(
                {**online, "training": {"attention_backend": "eager"}}
            )
        with self.assertRaisesRegex(
            ValidationError, "does not support attention_backend"
        ):
            Config.model_validate(
                {
                    **online,
                    "training": {
                        "strategy": "dflash",
                        "attention_backend": "fa",
                    },
                }
            )

    def test_online_producer_cannot_claim_the_consumer_ledger(self):
        payload = {
            **MINIMAL,
            "model": {**MINIMAL["model"], "target_backend": "sglang"},
            "data": {"train_data_path": "/data.jsonl"},
            "training": {
                "deployment_mode": "disaggregated",
                "role": "producer",
                "max_steps": 1,
                "metadata_db_path": "/shared/consumer.sqlite",
            },
        }
        with self.assertRaisesRegex(ValidationError, "consumer only"):
            Config.model_validate(payload)

    def test_offline_run_cannot_configure_an_online_consumer_ledger(self):
        payload = {
            **MINIMAL,
            "training": {"metadata_db_path": "/shared/consumer.sqlite"},
        }
        with self.assertRaisesRegex(ValidationError, "consumer only"):
            Config.model_validate(payload)

    def test_invalid_core_bounds_fail_during_config_validation(self):
        cases = (
            ("data.max_length", 0),
            ("data.build_dataset_num_proc", 0),
            ("data.dataloader_num_workers", -1),
            ("training.num_epochs", 0),
            ("training.max_steps", 0),
            ("training.total_steps", 0),
            ("training.batch_size", 0),
            ("training.accumulation_steps", 0),
            ("training.save_interval", -1),
            ("training.eval_interval", -1),
            ("training.log_interval", 0),
            ("training.max_checkpoints", -1),
            ("training.tp_size", 0),
            ("training.sp_ulysses_size", 0),
            ("training.sp_ring_size", 0),
            ("training.dist_timeout", 0),
        )
        for path, value in cases:
            with self.subTest(path=path):
                raw = copy.deepcopy(MINIMAL)
                section, field = path.split(".")
                raw.setdefault(section, {})[field] = value
                with self.assertRaises(ValidationError):
                    Config.model_validate(raw)

    def test_typed_profiler_and_streaming_runtime_bounds(self):
        cfg = Config.model_validate(
            {
                **MINIMAL,
                "data": {
                    **MINIMAL["data"],
                    "dataloader_num_workers": 3,
                },
                "profiling": {
                    "enabled": True,
                    "start_step": 2,
                    "num_steps": 5,
                    "record_shapes": True,
                },
                "runtime": {
                    "producer_lease": 4,
                    "in_flight_high_watermark": 32,
                    "in_flight_low_watermark": 16,
                    "resident_high_watermark_bytes": 4096,
                    "resident_low_watermark_bytes": 2048,
                    "feature_store_max_resident_bytes": 8192,
                },
            }
        )
        self.assertEqual(cfg.data.dataloader_num_workers, 3)
        self.assertEqual(cfg.profiling.num_steps, 5)
        self.assertEqual(cfg.runtime.in_flight_low_watermark, 16)

        invalid_runtime = (
            {
                "in_flight_high_watermark": 4,
                "in_flight_low_watermark": 5,
            },
            {"resident_low_watermark_bytes": 1},
            {
                "resident_high_watermark_bytes": 10,
                "resident_low_watermark_bytes": 11,
            },
            {
                "resident_high_watermark_bytes": 10,
                "feature_store_max_resident_bytes": 9,
            },
        )
        for runtime in invalid_runtime:
            with self.subTest(runtime=runtime), self.assertRaises(ValidationError):
                Config.model_validate({**MINIMAL, "runtime": runtime})

    def test_capture_only_producer_rejects_training_profiler(self):
        with self.assertRaisesRegex(ValidationError, "trainer roles"):
            Config.model_validate(
                {
                    **MINIMAL,
                    "data": {"train_data_path": "/data.jsonl"},
                    "training": {
                        "deployment_mode": "disaggregated",
                        "role": "producer",
                        "max_steps": 1,
                    },
                    "profiling": {"enabled": True},
                }
            )

    def test_local_tp_and_offline_usp_topologies_are_validated(self):
        tp = Config.model_validate(
            {
                **MINIMAL,
                "data": {"prompts_path": "/prompts.jsonl"},
                "training": {"tp_size": 2},
            }
        )
        tp.validate_world_size(4)
        with self.assertRaisesRegex(ValueError, "divisible"):
            tp.validate_world_size(3)

        usp = Config.model_validate(
            {
                **MINIMAL,
                "training": {
                    "attention_backend": "usp",
                    "sp_ulysses_size": 2,
                    "sp_ring_size": 1,
                },
            }
        )
        usp.validate_world_size(2)
        with self.assertRaisesRegex(ValidationError, "offline features"):
            Config.model_validate(
                {
                    **MINIMAL,
                    "data": {"prompts_path": "/prompts.jsonl"},
                    "training": {
                        "attention_backend": "usp",
                        "sp_ulysses_size": 2,
                    },
                }
            )
        with self.assertRaisesRegex(ValidationError, "attention_backend=usp"):
            Config.model_validate(
                {
                    **MINIMAL,
                    "training": {"sp_ulysses_size": 2},
                }
            )

    def test_overrides_coerce_and_revalidate(self):
        cfg = Config.model_validate(MINIMAL)
        out = apply_overrides(
            cfg, ["training.learning_rate=1e-3", "training.max_steps=7", "run_id=r2"]
        )
        self.assertEqual(out.training.learning_rate, 1e-3)
        self.assertEqual(out.training.max_steps, 7)
        self.assertEqual(out.run_id, "r2")
        # original untouched
        self.assertIsNone(cfg.training.max_steps)

    def test_override_bad_path_or_form_raises(self):
        cfg = Config.model_validate(MINIMAL)
        with self.assertRaises(ValueError):
            apply_overrides(cfg, ["training.no_such_field=1"])
        with self.assertRaises(ValueError):
            apply_overrides(cfg, ["not-an-assignment"])

    def test_load_config_applies_overrides(self):
        path = _write(MINIMAL, ".json")
        self.addCleanup(os.unlink, path)
        cfg = load_config(path, ["training.batch_size=4"])
        self.assertEqual(cfg.training.batch_size, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
