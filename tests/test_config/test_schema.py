# coding=utf-8
"""Config schema mechanics: parsing, validation, dotted overrides."""

import copy
import json
import os
import tempfile
import unittest
from dataclasses import replace

from pydantic import ValidationError

from specforge.algorithms import AlgorithmRegistration, AlgorithmRegistry
from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.application import resolve_run
from specforge.config import Config, apply_overrides, load_config

MINIMAL = {
    "model": {
        "target_model_path": "some/target",
        "draft_model_config": "draft.json",
        "vocab_mapping_path": "/mapping.pt",
    },
    "data": {"hidden_states_path": "/features"},
}

ONLINE_DEPLOYMENT = {
    "mode": "disaggregated",
    "disaggregated": {
        "control_dir": "/control",
        "backend": "mooncake",
        "server_urls": ["http://127.0.0.1:30000"],
    },
}


def _online_payload(strategy: str = "eagle3") -> dict:
    payload = copy.deepcopy(MINIMAL)
    payload["model"]["target_backend"] = "sglang"
    payload["data"] = {"train_data_path": "/train.jsonl"}
    payload["training"] = {"strategy": strategy, "max_steps": 1}
    payload["deployment"] = copy.deepcopy(ONLINE_DEPLOYMENT)
    return payload


def _managed_local_payload(*, ep_size: int) -> dict:
    payload = _online_payload()
    payload["model"]["sglang_ep_size"] = ep_size
    payload["deployment"]["disaggregated"] = {
        "control_dir": "/control",
        "backend": "mooncake",
        "managed_local": {
            "trainer_cuda_visible_devices": ["4"],
            "capture_servers": [
                {
                    "port": 30000,
                    "cuda_visible_devices": ["0", "1", "2", "3"],
                    "tp_size": 4,
                }
            ],
        },
    }
    return payload


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
    def test_finite_online_run_can_derive_its_schedule_from_the_producer(self):
        payload = _online_payload("domino")
        payload["training"].pop("max_steps")

        config = Config.model_validate(payload)

        self.assertIsNone(config.training.max_steps)
        self.assertIsNone(config.training.total_steps)

    def test_fsdp_sharding_is_typed(self):
        payload = copy.deepcopy(MINIMAL)
        payload["training"] = {"fsdp_sharding": "NO_SHARD"}
        self.assertEqual(
            Config.model_validate(payload).training.fsdp_sharding,
            "NO_SHARD",
        )

        payload["training"]["fsdp_sharding"] = "INVALID"
        with self.assertRaises(ValidationError):
            Config.model_validate(payload)

    def test_removed_vlm_pixel_knobs_are_rejected(self):
        for field in ("min_pixels", "max_pixels"):
            payload = copy.deepcopy(MINIMAL)
            payload["data"][field] = 1
            with self.subTest(field=field), self.assertRaises(ValidationError):
                Config.model_validate(payload)

    def test_input_modality_is_extensible_and_resolved_by_providers(self):
        payload = _online_payload()
        payload["model"]["input_modality"] = "vendor_vlm"
        payload["data"].update(
            {
                "chat_template": "vendor-template",
            }
        )
        cfg = Config.model_validate(payload)
        self.assertEqual(cfg.model.input_modality, "vendor_vlm")
        self.assertEqual(cfg.mode, "online")
        with self.assertRaisesRegex(
            ValueError,
            "no streaming feature contract and provider for modality 'vendor_vlm'",
        ):
            resolve_run(cfg)

        for modality in ("", "two words", " trailing"):
            invalid = copy.deepcopy(payload)
            invalid["model"]["input_modality"] = modality
            with self.subTest(modality=modality), self.assertRaises(ValidationError):
                Config.model_validate(invalid)

    def test_target_output_sharding_is_unavailable_with_server_capture(self):
        payload = _online_payload()
        payload["model"]["shard_target_output"] = True
        cfg = Config.model_validate(payload)
        self.assertTrue(cfg.model.shard_target_output)
        with self.assertRaisesRegex(
            ValueError, "unavailable with external server capture"
        ):
            resolve_run(cfg)

    def test_sglang_expert_parallelism_must_fit_target_tensor_parallelism(self):
        payload = _managed_local_payload(ep_size=2)
        config = Config.model_validate(payload)
        self.assertEqual(config.model.sglang_ep_size, 2)

        invalid = _managed_local_payload(ep_size=3)
        with self.assertRaisesRegex(ValidationError, "evenly divide"):
            Config.model_validate(invalid)

        invalid = _managed_local_payload(ep_size=8)
        with self.assertRaisesRegex(ValidationError, "no larger"):
            Config.model_validate(invalid)

    def test_online_eagle3_preserves_multi_sample_batches(self):
        payload = _online_payload()
        payload["training"]["batch_size"] = 4
        cfg = Config.model_validate(payload)
        resolve_run(cfg)
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
        online_payload = _online_payload()
        online_payload["data"] = {"prompts_path": "/prompts.jsonl"}
        online = Config.model_validate(online_payload)
        self.assertEqual(online.mode, "online")
        raw_payload = _online_payload()
        raw_payload["data"] = {"train_data_path": "/conversations.jsonl"}
        raw = Config.model_validate(raw_payload)
        self.assertEqual(raw.mode, "online")

    def test_offline_eval_source_and_interval_form_one_pair(self):
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

    def test_online_evaluation_is_explicitly_unsupported(self):
        offline_payload = {
            **MINIMAL,
            "data": {
                "hidden_states_path": "/train-features",
                "eval_data_path": "/eval.jsonl",
            },
            "training": {"eval_interval": 2},
        }
        online_payload = _online_payload()
        online_payload["data"]["eval_data_path"] = "/eval.jsonl"
        online_payload["training"]["eval_interval"] = 2

        for mode, payload in (
            ("offline", offline_payload),
            ("online", online_payload),
        ):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(
                    ValidationError, "data.eval_data_path is unsupported"
                ):
                    Config.model_validate(payload)

    def test_offline_eval_source_requires_offline_training(self):
        mismatched_online = _online_payload()
        mismatched_online["data"]["eval_hidden_states_path"] = "/eval-features"
        mismatched_online["training"]["eval_interval"] = 2
        with self.assertRaisesRegex(ValidationError, "offline training data source"):
            Config.model_validate(mismatched_online)

    def test_local_offline_eagle3_can_derive_vocab_mapping(self):
        missing = copy.deepcopy(MINIMAL)
        missing["model"].pop("vocab_mapping_path")
        config = Config.model_validate(missing)
        resolve_run(config)
        self.assertEqual(config.mode, "offline")
        self.assertEqual(config.model.vocab_mapping_path, "")
        self.assertEqual(Config.model_validate(MINIMAL).mode, "offline")

    def test_disaggregated_eagle3_requires_shared_vocab_mapping(self):
        missing = copy.deepcopy(MINIMAL)
        missing["model"].pop("vocab_mapping_path")
        missing["deployment"] = {
            "mode": "disaggregated",
            "disaggregated": {
                "control_dir": "/control",
                "backend": "shared_dir",
                "store_root": "/features",
            },
        }
        with self.assertRaisesRegex(ValueError, "vocab_mapping_path"):
            resolve_run(Config.model_validate(missing))

    def test_resume_allows_trainers_and_rejects_producer(self):
        local = Config.model_validate(
            {
                **MINIMAL,
                "training": {"resume_from": "/checkpoints/run-latest"},
            }
        )
        self.assertEqual(local.training.resume_from, "/checkpoints/run-latest")

        consumer_payload = _online_payload("dflash")
        consumer_payload["training"].update(
            {
                "role": "consumer",
                "total_steps": 10,
                "resume_from": "/checkpoints/run-latest",
            }
        )
        disagg_consumer = Config.model_validate(consumer_payload)
        resolve_run(disagg_consumer)
        self.assertEqual(
            disagg_consumer.training.resume_from, "/checkpoints/run-latest"
        )

        producer_payload = _online_payload("dflash")
        producer_payload["training"].update(
            {"role": "producer", "resume_from": "/checkpoints/run-latest"}
        )
        with self.assertRaisesRegex(ValidationError, "trainer role"):
            Config.model_validate(producer_payload)

    def test_unknown_backend_rejected(self):
        bad = {**MINIMAL, "model": {**MINIMAL["model"], "target_backend": "vllm"}}
        with self.assertRaises(ValidationError):
            Config.model_validate(bad)

    def test_removed_dspark_objective_selector_is_rejected(self):
        payload = _online_payload("dspark")
        payload["training"]["dspark_objective_mode"] = "legacy"
        with self.assertRaisesRegex(ValidationError, "dspark_objective_mode"):
            Config.model_validate(payload)

    def test_objective_chunk_blocks_is_shared_and_typed(self):
        for strategy in ("dflash", "domino", "dspark"):
            payload = _online_payload(strategy)
            payload["training"]["objective_chunk_blocks"] = 0
            with self.subTest(strategy=strategy):
                config = Config.model_validate(payload)
                self.assertEqual(config.training.objective_chunk_blocks, 0)

        payload = _online_payload("dflash")
        payload["training"]["objective_chunk_blocks"] = -1
        with self.assertRaisesRegex(ValidationError, "objective_chunk_blocks"):
            Config.model_validate(payload)

        payload = _online_payload("dspark")
        payload["training"]["dspark_objective_chunk_blocks"] = 1
        with self.assertRaisesRegex(ValidationError, "dspark_objective_chunk_blocks"):
            Config.model_validate(payload)

    def test_unknown_fields_and_unsupported_modes_fail_early(self):
        with self.assertRaises(ValidationError):
            Config.model_validate(
                {
                    **MINIMAL,
                    "data": {
                        "train_data_path": "/data.jsonl",
                        "unexpected_data_field": True,
                    },
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
        resolve_run(offline_dflash)
        self.assertEqual(offline_dflash.mode, "offline")
        with self.assertRaises(ValidationError):
            Config.model_validate(
                {
                    **MINIMAL,
                    "training": {"deployment_mode": "dataflow_colocated"},
                }
            )
        dspark_payload = _online_payload("dspark")
        dspark_payload["training"].update({"role": "producer", "total_steps": 10})
        dspark = Config.model_validate(dspark_payload)
        resolve_run(dspark)
        self.assertEqual(dspark.training.strategy, "dspark")
        # The in-process HF/custom backends were removed with the server-only
        # cutover; retired names must fail at load for BOTH topologies rather
        # than being silently ignored on offline runs.
        invalid_backend = _online_payload("dflash")
        invalid_backend["model"]["target_backend"] = "hf"
        with self.assertRaisesRegex(ValidationError, "target_backend"):
            Config.model_validate(invalid_backend)
        offline_backend = copy.deepcopy(MINIMAL)
        offline_backend["model"]["target_backend"] = "custom"
        with self.assertRaisesRegex(ValidationError, "target_backend"):
            Config.model_validate(offline_backend)

        unknown = copy.deepcopy(MINIMAL)
        unknown["training"] = {"strategy": "unknown_algorithm"}
        cfg = Config.model_validate(unknown)
        with self.assertRaisesRegex(ValueError, "unknown algorithm"):
            resolve_run(cfg)

    def test_registered_strategy_needs_no_schema_edit(self):
        name = "config_registry_test"
        builtin = builtin_algorithm_registry()
        base = builtin.resolve("eagle3")
        registration = AlgorithmRegistration(
            spec=replace(base.spec, name=name),
            providers=replace(base.providers, algorithm_name=name),
        )
        registry = AlgorithmRegistry((registration,))

        payload = copy.deepcopy(MINIMAL)
        payload["training"] = {"strategy": name}
        cfg = Config.model_validate(payload)
        self.assertEqual(cfg.training.strategy, name)
        self.assertIs(resolve_run(cfg, registry).algorithm, registration)
        self.assertNotIn(name, builtin.names)

    def test_strategy_specific_capture_and_attention_are_validated(self):
        online = _online_payload()
        with self.assertRaisesRegex(ValueError, "exactly three"):
            resolve_run(
                Config.model_validate(
                    {
                        **online,
                        "model": {
                            **online["model"],
                            "aux_hidden_state_layer_ids": [1, 2],
                        },
                    }
                )
            )

        with self.assertRaisesRegex(ValueError, "would be ignored"):
            resolve_run(
                Config.model_validate(
                    {
                        **online,
                        "model": {
                            **online["model"],
                            "aux_hidden_state_layer_ids": [1, 2, 3],
                        },
                        "training": {
                            **online["training"],
                            "strategy": "dflash",
                        },
                    }
                )
            )
        with self.assertRaisesRegex(ValueError, "does not support attention_backend"):
            resolve_run(
                Config.model_validate(
                    {
                        **online,
                        "training": {
                            **online["training"],
                            "strategy": "peagle",
                            "attention_backend": "sdpa",
                        },
                    }
                )
            )
        with self.assertRaisesRegex(ValueError, "does not support attention_backend"):
            resolve_run(
                Config.model_validate(
                    {
                        **online,
                        "training": {
                            **online["training"],
                            "attention_backend": "eager",
                        },
                    }
                )
            )
        with self.assertRaisesRegex(ValueError, "does not support attention_backend"):
            resolve_run(
                Config.model_validate(
                    {
                        **online,
                        "training": {
                            **online["training"],
                            "strategy": "dflash",
                            "attention_backend": "fa",
                        },
                    }
                )
            )

    def test_legacy_deployment_fields_are_not_domain_model_fields(self):
        for field, value in (
            ("deployment_mode", "disaggregated"),
            ("server_urls", ["http://127.0.0.1:30000"]),
            ("metadata_db_path", "/shared/consumer.sqlite"),
        ):
            payload = copy.deepcopy(MINIMAL)
            payload["training"] = {field: value}
            with self.subTest(field=field), self.assertRaises(ValidationError):
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
        payload = _online_payload("dflash")
        payload["training"].update({"role": "producer", "max_steps": 1})
        payload["profiling"] = {"enabled": True}
        with self.assertRaisesRegex(ValidationError, "trainer roles"):
            Config.model_validate(payload)

    def test_offline_dp_and_usp_topologies_are_validated(self):
        with self.assertRaisesRegex(
            ValidationError, "do not implement trainer tensor parallelism"
        ):
            Config.model_validate(
                {
                    **MINIMAL,
                    "training": {"tp_size": 2},
                }
            )

        dp = Config.model_validate(
            {
                **MINIMAL,
                "training": {"tp_size": 1},
            }
        )
        dp.validate_world_size(4)

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
        with self.assertRaisesRegex(ValueError, "divisible"):
            usp.validate_world_size(3)
        online_usp = _online_payload()
        online_usp["training"].update(
            {"attention_backend": "usp", "sp_ulysses_size": 2}
        )
        with self.assertRaisesRegex(ValidationError, "offline features"):
            Config.model_validate(online_usp)
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
