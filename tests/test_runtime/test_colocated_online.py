# coding=utf-8
"""CPU gates for colocated online planning, capture mapping, and residency."""

from __future__ import annotations

import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.providers import (
    ServerCaptureLayout,
    ServerStreamingProvider,
)
from specforge.config import Config
from specforge.inference.adapters import LocalSGLangCaptureAdapter
from specforge.inference.batch_partition import TargetBatchPartition
from specforge.inference.capture import CaptureConfig
from specforge.launch import _plan_online_prompt_stream
from specforge.runtime.contracts import PromptTask, SampleRef
from specforge.runtime.control_plane import DataFlowController
from specforge.runtime.data_plane import LocalRolloutStream
from specforge.training.assembly import (
    TrainingRun,
    _prepare_colocated_prompts,
    build_training_run,
)


def _prompts(count):
    return [
        {
            "task_id": f"source-{index}",
            "payload": {"input_ids": [index, index + 1], "loss_mask": [0, 1]},
        }
        for index in range(count)
    ]


class _FakeCapture:
    capture_layers = [1, 3, 7]

    def __init__(self):
        parameter = torch.nn.Parameter(torch.empty(0))
        model = types.SimpleNamespace(parameters=lambda: iter((parameter,)))
        runner = types.SimpleNamespace(model=model)
        self._backend = types.SimpleNamespace(model_runner=runner)

    def capture_rows(self, rows):
        return (
            tuple(torch.ones(len(row), 12) for row in rows),
            tuple(torch.ones(len(row), 4) for row in rows),
        )


class _FakeStore:
    def __init__(self):
        self.resident = {}

    def put(self, *, sample_id, source_task_id):
        self.resident[sample_id] = object()
        return SampleRef(
            sample_id=sample_id,
            run_id="run",
            source_task_id=source_task_id,
            feature_store_uri=f"fake://run/{sample_id}",
            feature_keys={"feature": f"{sample_id}/feature"},
            feature_specs={},
            strategy="eagle3",
        )

    def release(self, sample_id):
        self.resident.pop(sample_id, None)

    def abort(self, sample_id, *, reason):
        del reason
        self.resident.pop(sample_id, None)

    def abort_all(self, *, reason):
        del reason
        count = len(self.resident)
        self.resident.clear()
        return count

    def health(self):
        return {"resident_samples": len(self.resident)}


class _FakeWorker:
    def __init__(self, controller, store):
        self.controller = controller
        self.store = store
        self.worker_id = controller.register_rollout_worker(
            {"worker_id": "fake", "role": "rollout"}
        )
        self.stopped = None

    def start(self):
        pass

    def stop(self, reason="stopped"):
        self.stopped = reason

    def run_once(self, max_tasks):
        tasks = self.controller.lease_prompt_tasks(self.worker_id, max_tasks)
        refs = [
            self.store.put(sample_id=f"run:{task.task_id}", source_task_id=task.task_id)
            for task in tasks
        ]
        self.controller.commit_samples(self.worker_id, refs)
        return refs


class ColocatedOnlineTest(unittest.TestCase):
    def test_distributed_prompt_preparation_reads_rank_zero_cache(self):
        config = Config.model_validate(
            {
                "model": {
                    "target_model_path": "target",
                    "draft_model_config": "draft.json",
                },
                "data": {
                    "prompts_path": "prompts.jsonl",
                    "cache_dir": "/cache",
                    "cache_key": "prepared",
                },
                "training": {"strategy": "dflash", "max_steps": 1},
                "deployment": {"mode": "local_colocated"},
            }
        )
        prompts = _prompts(2)
        algorithm = builtin_algorithm_registry().resolve("dflash")

        with (
            mock.patch(
                "specforge.training.assembly._prepare_prompts",
                return_value=prompts,
            ) as prepare,
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=8),
            mock.patch("torch.distributed.get_rank", return_value=3),
            mock.patch("torch.distributed.all_gather_object") as gather,
            mock.patch("os.path.isfile", return_value=True),
            mock.patch.dict("os.environ", {"LOCAL_RANK": "3"}),
        ):
            result = _prepare_colocated_prompts(
                config,
                object(),
                algorithm=algorithm,
                draft_config=object(),
            )

        self.assertIs(result, prompts)
        prepare.assert_called_once()
        self.assertEqual(gather.call_count, 2)

    def test_node_cache_owner_prepares_when_rank_zero_cache_is_not_visible(self):
        config = Config.model_validate(
            {
                "model": {
                    "target_model_path": "target",
                    "draft_model_config": "draft.json",
                },
                "data": {
                    "prompts_path": "prompts.jsonl",
                    "cache_dir": "/cache",
                    "cache_key": "prepared",
                },
                "training": {"strategy": "dflash", "max_steps": 1},
                "deployment": {"mode": "local_colocated"},
            }
        )
        prompts = _prompts(2)
        algorithm = builtin_algorithm_registry().resolve("dflash")

        with (
            mock.patch(
                "specforge.training.assembly._prepare_prompts",
                return_value=prompts,
            ) as prepare,
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=16),
            mock.patch("torch.distributed.get_rank", return_value=8),
            mock.patch("torch.distributed.all_gather_object") as gather,
            mock.patch("os.path.isfile", return_value=False),
            mock.patch("os.makedirs"),
            mock.patch("builtins.open", mock.mock_open()),
            mock.patch("os.replace"),
            mock.patch.dict("os.environ", {"LOCAL_RANK": "0"}),
        ):
            result = _prepare_colocated_prompts(
                config,
                object(),
                algorithm=algorithm,
                draft_config=object(),
            )

        self.assertIs(result, prompts)
        prepare.assert_called_once()
        self.assertEqual(gather.call_count, 2)

    def test_composition_builds_local_sglang_before_the_trainer_runtime(self):
        algorithm = builtin_algorithm_registry().resolve("dflash")
        config = Config.model_validate(
            {
                "model": {
                    "target_model_path": "target",
                    "draft_model_config": "draft.json",
                    "sglang_context_length": 128,
                    "sglang_mem_fraction_static": 0.4,
                },
                "data": {"prompts_path": "prompts.jsonl", "max_length": 64},
                "training": {
                    "strategy": "dflash",
                    "max_steps": 1,
                    "batch_size": 1,
                },
                "deployment": {"mode": "local_colocated"},
            }
        )
        bundle = SimpleNamespace(
            model=object(),
            draft_model=object(),
            draft_config=object(),
            input_tools=object(),
            target_head=None,
            target_hidden_size=4,
            target_vocab_size=16,
            draft_vocab_size=16,
            capture_layers=[1],
            strategy_kwargs={},
        )
        target_capture = mock.Mock()
        trainer = object()
        prompts = _prompts(4)

        with (
            mock.patch(
                "specforge.training.assembly.build_model_bundle",
                return_value=bundle,
            ),
            mock.patch(
                "specforge.training.assembly._prepare_prompts",
                return_value=prompts,
            ),
            mock.patch(
                "specforge.training.assembly._configured_logger",
                return_value=mock.Mock(),
            ),
            mock.patch(
                "specforge.offline_capture.load_offline_capture",
                return_value=target_capture,
            ) as load_capture,
            mock.patch(
                "specforge.launch.build_colocated_online_runtime",
                return_value=trainer,
            ) as build_runtime,
        ):
            run = build_training_run(config, algorithm=algorithm)

        self.assertIsInstance(run, TrainingRun)
        self.assertIs(run.trainer, trainer)
        self.assertEqual(load_capture.call_args.kwargs["context_length"], 128)
        self.assertEqual(load_capture.call_args.kwargs["max_running_requests"], 1)
        target_capture.set_capture_layers.assert_called_once_with(
            [1], capture_method="dflash"
        )
        self.assertEqual(build_runtime.call_args.kwargs["dataset_size"], 4)
        self.assertTrue(build_runtime.call_args.kwargs["zero_copy_features"])

    def test_prompt_plan_is_island_disjoint_and_tp_batch_aligned(self):
        rank0 = _plan_online_prompt_stream(
            _prompts(19),
            num_epochs=2,
            seed=17,
            tp_size=2,
            batch_size=2,
            dp_rank=0,
            dp_size=2,
        )
        rank1 = _plan_online_prompt_stream(
            _prompts(19),
            num_epochs=2,
            seed=17,
            tp_size=2,
            batch_size=2,
            dp_rank=1,
            dp_size=2,
        )
        self.assertEqual(len(rank0) % 4, 0)
        self.assertEqual(len(rank1) % 4, 0)
        for epoch in range(2):
            left = {
                item["metadata"]["source_prompt_index"]
                for item in rank0
                if item["metadata"]["prompt_epoch"] == epoch
            }
            right = {
                item["metadata"]["source_prompt_index"]
                for item in rank1
                if item["metadata"]["prompt_epoch"] == epoch
            }
            self.assertTrue(left.isdisjoint(right))

    def test_local_adapter_emits_server_streaming_layout(self):
        provider = ServerStreamingProvider(
            modality="text",
            capture_method="eagle3",
            target_representation="hidden_state",
            layout=ServerCaptureLayout(
                aux_feature="hidden_state",
                last_hidden_feature="target",
                passthrough=(
                    ("input_ids", "input_ids", ()),
                    ("loss_mask", "loss_mask", ()),
                ),
                attention_mask_feature="attention_mask",
            ),
            build_collator=lambda: None,
        )
        adapter = LocalSGLangCaptureAdapter(
            _FakeCapture(), provider=provider, synchronize_after_capture=False
        )
        task = PromptTask(
            task_id="task",
            run_id="run",
            source_id="source",
            payload={"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]},
            max_length=8,
        )
        capture = CaptureConfig.from_strategy(
            required_features={
                "input_ids",
                "loss_mask",
                "attention_mask",
                "hidden_state",
                "target",
            },
            aux_hidden_state_layer_ids=(1, 3, 7),
            target_repr="hidden_state",
            target_hidden_size=4,
        )

        [features] = adapter.generate_features([task], capture=capture)

        self.assertEqual(tuple(features["hidden_state"].shape), (1, 3, 12))
        self.assertEqual(tuple(features["target"].shape), (1, 3, 4))
        self.assertEqual(tuple(features["input_ids"].shape), (1, 3))
        self.assertTrue(features["attention_mask"].all())
        self.assertEqual(features["__aux_layer_ids__"], (1, 3, 7))

    def test_local_adapter_detaches_only_this_tp_ranks_rows(self):
        class _PackedCapture(_FakeCapture):
            def capture_rows(self, rows):
                lengths = [len(row) for row in rows]
                total = sum(lengths)
                aux = torch.ones(total, 12)
                last = torch.ones(total, 4)
                return torch.split(aux, lengths), torch.split(last, lengths)

        provider = ServerStreamingProvider(
            modality="text",
            capture_method="dspark",
            target_representation="hidden_state",
            layout=ServerCaptureLayout(
                aux_feature="hidden_states",
                last_hidden_feature="target_last_hidden_states",
                passthrough=(("input_ids", "input_ids", ()),),
            ),
            build_collator=lambda: None,
        )
        adapter = LocalSGLangCaptureAdapter(
            _PackedCapture(),
            provider=provider,
            synchronize_after_capture=False,
            batch_partition=TargetBatchPartition(rank=1, size=2),
        )
        tasks = [
            PromptTask(
                task_id=f"task-{index}",
                run_id="run",
                source_id="source",
                payload={"input_ids": [index, index + 1], "loss_mask": [0, 1]},
                max_length=8,
            )
            for index in range(4)
        ]

        features = adapter.generate_features(
            tasks,
            capture=CaptureConfig.from_strategy(
                required_features={
                    "input_ids",
                    "hidden_states",
                    "target_last_hidden_states",
                },
                aux_hidden_state_layer_ids=(1, 3, 7),
                target_repr="hidden_state",
                target_hidden_size=4,
            ),
        )

        self.assertTrue(adapter.returns_local_batch)
        self.assertEqual([row["input_ids"][0, 0].item() for row in features], [2, 3])
        # clone() severs each local row from the full packed capture allocation.
        for row in features:
            hidden = row["hidden_states"]
            self.assertEqual(
                hidden.untyped_storage().nbytes(),
                hidden.numel() * hidden.element_size(),
            )

    def test_pull_through_stream_keeps_one_batch_resident(self):
        controller = DataFlowController("run")
        controller.ingest_prompts(_prompts(8))
        store = _FakeStore()
        worker = _FakeWorker(controller, store)
        stream = LocalRolloutStream(
            controller=controller,
            workers=[worker],
            feature_store=store,
            max_resident_samples=2,
        )
        consumed = 0
        with stream:
            while refs := stream.get(2):
                self.assertLessEqual(store.health()["resident_samples"], 2)
                for ref in refs:
                    store.release(ref.sample_id)
                stream.ack(refs)
                consumed += len(refs)
        self.assertEqual(consumed, 8)
        self.assertEqual(stream.peak_resident_samples, 2)
        self.assertEqual(store.health()["resident_samples"], 0)
        metrics = stream.perf_metrics()
        self.assertEqual(metrics["colocated_capture_calls"], 4.0)
        self.assertGreater(metrics["colocated_capture_time_s"], 0.0)
        self.assertGreater(metrics["colocated_capture_samples_per_second"], 0.0)
        self.assertEqual(metrics["colocated_peak_resident_samples"], 2.0)
        self.assertEqual(
            stream.perf_metrics()["colocated_capture_calls"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
