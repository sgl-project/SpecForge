# coding=utf-8
"""Canonical ``specforge train`` assembly for windowed fanout roles."""

from __future__ import annotations

import os
import tempfile
import types
import unittest
from unittest import mock

from specforge.config import Config
from specforge.training.disaggregated import (
    _build_windowed_online,
    _stabilize_windowed_prompts,
)


def _config(root: str, *, role: str, resume_from: str | None = None) -> Config:
    return Config.model_validate(
        {
            "run_id": "fanout-run",
            "output_dir": os.path.join(root, "output"),
            "model": {
                "target_model_path": "target",
                "draft_model_config": "draft",
                "target_backend": "sglang",
            },
            "data": {
                "train_data_path": "/prompts.jsonl",
                "max_prompts": 4,
            },
            "training": {
                "strategy": "dflash",
                "role": role,
                "batch_size": 2,
                "accumulation_steps": 1,
                "num_epochs": 1,
                "max_steps": 2,
            },
            "deployment": {
                "mode": "disaggregated",
                "trainer": {"nnodes": 1, "nproc_per_node": 1},
                "disaggregated": {
                    "control_dir": root,
                    "backend": "mooncake",
                    "server_urls": ["http://capture:30000"],
                    "windowed_fanout": {
                        "max_live_bytes": 1 << 30,
                        "max_outstanding_per_consumer": 4,
                        "consumers": [
                            {
                                "consumer_id": "block-4",
                                "seed": 42,
                                "loss_type": "dflash",
                                "loss_decay_gamma": 7.0,
                                "dpace_alpha": 0.5,
                                "draft_block_size": 4,
                                "num_anchors": 128,
                                "learning_rate": 0.0006,
                                "warmup_ratio": 0.04,
                                "cuda_visible_device": "1",
                                "resume_from": resume_from,
                            }
                        ],
                    },
                },
            },
        }
    )


def _algorithm():
    streaming = types.SimpleNamespace(
        target_representation=None,
        layout=types.SimpleNamespace(
            aux_feature="aux",
            last_hidden_feature="last_hidden",
            passthrough=(),
            attention_mask_feature=None,
        ),
        create_input_adapter=mock.Mock(return_value=None),
    )
    providers = types.SimpleNamespace(
        server_streaming_for=mock.Mock(return_value=streaming),
        model=types.SimpleNamespace(draft_config=object()),
    )
    return types.SimpleNamespace(name="dflash", providers=providers), streaming


def _environment(root: str, **extra: str) -> dict[str, str]:
    return {
        "DISAGG_WINDOW_REGISTRY": os.path.join(root, "window.db"),
        "DISAGG_DB": os.path.join(root, "consumer.db"),
        **extra,
    }


class WindowedFanoutAssemblyTest(unittest.TestCase):
    def test_producer_uses_canonical_prompts_and_cleans_owner_on_failure(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = _config(root, role="producer")
            algorithm, _ = _algorithm()
            runtime = mock.Mock()
            failure = RuntimeError("capture failed")
            runtime.drive.side_effect = failure
            store = mock.Mock()
            prompts = [
                {"payload": {"input_ids": [index], "loss_mask": [1]}}
                for index in range(4)
            ]
            prepare_prompts = mock.Mock(return_value=prompts)

            with (
                mock.patch.dict(os.environ, _environment(root), clear=False),
                mock.patch(
                    "specforge.training.disaggregated._producer_capture_metadata",
                    return_value=([1, 2], 16, 32, 24),
                ),
                mock.patch(
                    "specforge.launch.build_disagg_windowed_capture_contract",
                    return_value=(object(), "digest"),
                ),
                mock.patch(
                    "specforge.training.assembly._load_input_tools",
                    return_value=object(),
                ),
                mock.patch(
                    "specforge.training.model_loading.resolve_draft_config",
                    return_value=object(),
                ),
                mock.patch(
                    "specforge.inference.adapters.server_capture."
                    "SGLangServerCaptureAdapter"
                ) as adapter,
                mock.patch(
                    "specforge.launch.build_disagg_online_windowed_producer",
                    autospec=True,
                    return_value=runtime,
                ) as build_producer,
                mock.patch(
                    "specforge.training.disaggregated._mooncake_store",
                    return_value=store,
                ) as build_store,
                mock.patch(
                    "specforge.runtime.data_plane.feature_store."
                    "drain_feature_store_removals"
                ) as drain,
            ):
                run = _build_windowed_online(
                    cfg,
                    algorithm=algorithm,
                    build_model_bundle=mock.Mock(),
                    prepare_prompts=prepare_prompts,
                    optimizer_factory=mock.Mock(),
                    logger=mock.Mock(),
                )
                with self.assertRaises(RuntimeError) as raised:
                    run.run()

            self.assertIs(raised.exception, failure)
            prepare_prompts.assert_called_once()
            adapter.assert_called_once()
            build_store.assert_called_once_with(cfg, lifetime_owner=True)
            prepared = build_producer.call_args.kwargs["prompts"]
            self.assertEqual(
                [prompt["payload"] for prompt in prepared],
                [prompt["payload"] for prompt in prompts],
            )
            self.assertEqual(
                [prompt["task_id"] for prompt in prepared],
                [prompt["task_id"] for prompt in _stabilize_windowed_prompts(prompts)],
            )
            self.assertTrue(all("task_id" not in prompt for prompt in prompts))
            self.assertEqual(
                build_producer.call_args.kwargs["consumer_ids"], ("block-4",)
            )
            self.assertEqual(build_producer.call_args.kwargs["modality"], "text")
            runtime.close.assert_called_once_with()
            store.abort_all.assert_called_once_with(
                reason="windowed-attempt-failed", force=True
            )
            drain.assert_called_once_with(store)

    def test_producer_rejects_incomplete_fixed_prompt_inventory(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = _config(root, role="producer")
            algorithm, _ = _algorithm()
            with (
                mock.patch.dict(os.environ, _environment(root), clear=False),
                mock.patch(
                    "specforge.training.disaggregated._producer_capture_metadata",
                    return_value=([], 16, 32, 24),
                ),
                mock.patch(
                    "specforge.launch.build_disagg_windowed_capture_contract",
                    return_value=(object(), "digest"),
                ),
                mock.patch(
                    "specforge.training.assembly._load_input_tools",
                    return_value=object(),
                ),
                mock.patch(
                    "specforge.training.model_loading.resolve_draft_config",
                    return_value=object(),
                ),
                mock.patch(
                    "specforge.launch.build_disagg_online_windowed_producer"
                ) as build_producer,
            ):
                with self.assertRaisesRegex(ValueError, "prepared prompt count"):
                    _build_windowed_online(
                        cfg,
                        algorithm=algorithm,
                        build_model_bundle=mock.Mock(),
                        prepare_prompts=mock.Mock(return_value=[{"task_id": "0"}]),
                        optimizer_factory=mock.Mock(),
                        logger=mock.Mock(),
                    )
            build_producer.assert_not_called()

    def test_prompt_ids_are_deterministic_and_conflicts_are_rejected(self):
        prompts = [
            {"payload": {"input_ids": [1, 2], "loss_mask": [0, 1]}},
            {"payload": {"input_ids": [1, 2], "loss_mask": [0, 1]}},
        ]
        first = _stabilize_windowed_prompts(prompts)
        second = _stabilize_windowed_prompts(prompts)

        self.assertEqual(
            [prompt["task_id"] for prompt in first],
            [prompt["task_id"] for prompt in second],
        )
        self.assertNotEqual(first[0]["task_id"], first[1]["task_id"])
        with self.assertRaisesRegex(ValueError, "duplicated"):
            _stabilize_windowed_prompts([{"task_id": "same"}, {"task_id": "same"}])
        with self.assertRaisesRegex(ValueError, "must provide one explicitly"):
            _stabilize_windowed_prompts([{"payload": {"opaque": object()}}])

    def test_consumer_registers_before_model_build_and_forwards_resume(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = os.path.join(root, "checkpoint-1")
            cfg = _config(root, role="consumer", resume_from=checkpoint)
            algorithm, _ = _algorithm()
            events: list[str] = []
            registry = mock.Mock()
            registry.wait_initialized.return_value = {
                "run_id": cfg.run_id,
                "contract_digest": "digest",
                "total_samples": cfg.data.max_prompts,
            }
            control = mock.Mock()
            runtime = mock.Mock()
            runtime.run.return_value = 2
            bundle = types.SimpleNamespace(model=object(), strategy_kwargs={"x": 1})

            def register(*_args, **_kwargs):
                events.append("register")
                return control

            def build_model(_cfg):
                events.append("model")
                return bundle

            store = mock.Mock(lifetime_owner=False)
            optimizer = mock.Mock(return_value=object())
            with (
                mock.patch.dict(
                    os.environ,
                    _environment(root, SPECFORGE_FANOUT_CONSUMER_ID="block-4"),
                    clear=False,
                ),
                mock.patch(
                    "specforge.training.disaggregated._producer_capture_metadata",
                    return_value=([], 16, 32, 24),
                ),
                mock.patch(
                    "specforge.launch.build_disagg_windowed_capture_contract",
                    return_value=(object(), "digest"),
                ),
                mock.patch(
                    "specforge.runtime.data_plane.windowed_capture."
                    "SQLiteWindowedCaptureRegistry",
                    return_value=registry,
                ),
                mock.patch(
                    "specforge.runtime.data_plane.windowed_capture_runtime."
                    "start_windowed_consumer_control",
                    side_effect=register,
                ),
                mock.patch(
                    "specforge.training.disaggregated._mooncake_store",
                    return_value=store,
                ) as build_store,
                mock.patch(
                    "specforge.launch.build_disagg_online_windowed_consumer",
                    return_value=runtime,
                ) as build_consumer,
            ):
                run = _build_windowed_online(
                    cfg,
                    algorithm=algorithm,
                    build_model_bundle=build_model,
                    prepare_prompts=mock.Mock(),
                    optimizer_factory=optimizer,
                    logger=mock.Mock(),
                )
                self.assertEqual(run.run(), 2)

            self.assertEqual(events, ["register", "model"])
            build_store.assert_called_once_with(cfg, lifetime_owner=False)
            kwargs = build_consumer.call_args.kwargs
            self.assertEqual(kwargs["consumer_id"], "block-4")
            self.assertEqual(
                kwargs["metadata_db_path"], _environment(root)["DISAGG_DB"]
            )
            self.assertTrue(kwargs["resume"])
            self.assertEqual(kwargs["resume_from"], checkpoint)
            self.assertEqual(kwargs["strategy_kwargs"], {"x": 1})
            self.assertIs(kwargs["consumer_control"], control)
            runtime.close.assert_called_once_with()

    def test_model_build_failure_is_reported_and_registry_is_closed(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = _config(root, role="consumer")
            algorithm, _ = _algorithm()
            registry = mock.Mock()
            registry.wait_initialized.return_value = {
                "run_id": cfg.run_id,
                "contract_digest": "digest",
                "total_samples": cfg.data.max_prompts,
            }
            control = mock.Mock()
            failure = RuntimeError("model build failed")
            with (
                mock.patch.dict(
                    os.environ,
                    _environment(root, SPECFORGE_FANOUT_CONSUMER_ID="block-4"),
                    clear=False,
                ),
                mock.patch(
                    "specforge.training.disaggregated._producer_capture_metadata",
                    return_value=([], 16, 32, 24),
                ),
                mock.patch(
                    "specforge.launch.build_disagg_windowed_capture_contract",
                    return_value=(object(), "digest"),
                ),
                mock.patch(
                    "specforge.runtime.data_plane.windowed_capture."
                    "SQLiteWindowedCaptureRegistry",
                    return_value=registry,
                ),
                mock.patch(
                    "specforge.runtime.data_plane.windowed_capture_runtime."
                    "start_windowed_consumer_control",
                    return_value=control,
                ),
                mock.patch(
                    "specforge.launch.build_disagg_online_windowed_consumer"
                ) as build_consumer,
            ):
                with self.assertRaises(RuntimeError) as raised:
                    _build_windowed_online(
                        cfg,
                        algorithm=algorithm,
                        build_model_bundle=mock.Mock(side_effect=failure),
                        prepare_prompts=mock.Mock(),
                        optimizer_factory=mock.Mock(),
                        logger=mock.Mock(),
                    )

            self.assertIs(raised.exception, failure)
            control.fail.assert_called_once_with(failure)
            control.close.assert_called_once_with()
            registry.close.assert_called_once_with()
            build_consumer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
