# coding=utf-8
"""CPU gates for colocated online planning, run assembly, capture mapping, and
the bounded rollout stream driven through the real worker and feature store."""

from __future__ import annotations

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
from specforge.inference.rollout_worker import RolloutWorker
from specforge.launch import build_colocated_online_runtime
from specforge.runtime.contracts import PromptTask
from specforge.runtime.control_plane import DataFlowController
from specforge.runtime.control_plane.metadata_store import NoOpMetadataStore
from specforge.runtime.data_plane import LocalFeatureStore, LocalRolloutStream
from specforge.training import prompt_plan
from specforge.training.assembly import (
    TrainingRun,
    _prepare_prompts_coordinated,
    build_training_run,
)

HIDDEN = 4
LAYERS = (1, 3, 7)


def _prompts(count):
    return [
        {
            "task_id": f"source-{index}",
            "payload": {"input_ids": [index, index + 1], "loss_mask": [0, 1]},
        }
        for index in range(count)
    ]


def _colocated_config(**data_overrides):
    data = {"prompts_path": "prompts.jsonl"}
    data.update(data_overrides)
    return Config.model_validate(
        {
            "model": {
                "target_model_path": "target",
                "draft_model_config": "draft.json",
            },
            "data": data,
            "training": {"strategy": "dflash", "max_steps": 1},
            "deployment": {"mode": "local_colocated"},
        }
    )


class _PackedCapture:
    """Stand-in for the SGLang backend: one packed forward, views per row."""

    capture_layers = list(LAYERS)

    def capture_rows(self, rows):
        lengths = [len(row) for row in rows]
        total = sum(lengths)
        aux = torch.ones(total, HIDDEN * len(LAYERS))
        last = torch.ones(total, HIDDEN)
        return torch.split(aux, lengths), torch.split(last, lengths)


def _provider(capture_method="dspark"):
    return ServerStreamingProvider(
        modality="text",
        capture_method=capture_method,
        target_representation="hidden_state",
        layout=ServerCaptureLayout(
            aux_feature="hidden_states",
            last_hidden_feature="target_last_hidden_states",
            passthrough=(
                ("input_ids", "input_ids", ()),
                ("loss_mask", "loss_mask", ()),
            ),
            attention_mask_feature="attention_mask",
        ),
        build_collator=lambda: None,
    )


def _capture_config():
    return CaptureConfig.from_strategy(
        required_features={
            "input_ids",
            "loss_mask",
            "attention_mask",
            "hidden_states",
            "target_last_hidden_states",
        },
        aux_hidden_state_layer_ids=LAYERS,
        target_repr="hidden_state",
        target_hidden_size=HIDDEN,
    )


def _task(index):
    return PromptTask(
        task_id=f"task-{index}",
        run_id="run",
        source_id="source",
        payload={"input_ids": [index, index + 1], "loss_mask": [0, 1]},
        max_length=8,
    )


class PromptPlanTest(unittest.TestCase):
    def test_island_shards_partition_the_producer_permutation(self):
        prompts = _prompts(19)
        producer_order = prompt_plan.epoch_prompt_indices(prompts, 1, seed=17)

        shards = [
            prompt_plan.epoch_prompt_shard(
                prompts, 1, seed=17, shard_rank=rank, shard_count=2, batch_multiple=4
            )
            for rank in range(2)
        ]

        # 19 prompts / 2 islands = 9 each, truncated to complete 4-prompt batches.
        self.assertEqual([len(shard) for shard in shards], [8, 8])
        self.assertTrue(set(shards[0]).isdisjoint(shards[1]))
        for rank, shard in enumerate(shards):
            self.assertEqual(shard, producer_order[rank::2][:8])

    def test_multi_epoch_stream_matches_producer_task_identities(self):
        prompts = _prompts(6)

        stream = list(
            prompt_plan.iter_sharded_online_prompts(
                prompts,
                num_epochs=2,
                seed=3,
                shard_rank=0,
                shard_count=1,
                batch_multiple=2,
            )
        )

        expected = prompt_plan.epoch_online_prompts(
            prompts, 0, 2, seed=3
        ) + prompt_plan.epoch_online_prompts(prompts, 1, 2, seed=3)
        self.assertEqual(
            [item["task_id"] for item in stream], [item["task_id"] for item in expected]
        )
        self.assertEqual(stream[0]["metadata"]["epoch"], 0)
        self.assertEqual(stream[-1]["metadata"]["epoch"], 1)

    def test_single_epoch_keeps_source_task_ids(self):
        prompts = _prompts(4)
        stream = list(
            prompt_plan.iter_sharded_online_prompts(
                prompts,
                num_epochs=1,
                seed=0,
                shard_rank=0,
                shard_count=1,
                batch_multiple=1,
            )
        )
        self.assertEqual(
            sorted(item["task_id"] for item in stream),
            sorted(item["task_id"] for item in prompts),
        )

    def test_shard_epoch_size_drops_incomplete_batches(self):
        self.assertEqual(
            prompt_plan.shard_epoch_size(19, shard_count=2, batch_multiple=4), 8
        )
        self.assertEqual(
            prompt_plan.shard_epoch_size(3, shard_count=2, batch_multiple=2), 0
        )


class CoordinatedPromptPreparationTest(unittest.TestCase):
    def _prepare(self, *, world_size, rank, local_rank, cache_visible):
        config = _colocated_config(cache_dir="/cache", cache_key="prepared")
        prompts = _prompts(2)
        algorithm = builtin_algorithm_registry().resolve("dflash")
        with (
            mock.patch(
                "specforge.training.assembly._prepare_prompts", return_value=prompts
            ) as prepare,
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=world_size),
            mock.patch("torch.distributed.get_rank", return_value=rank),
            mock.patch("torch.distributed.all_gather_object") as gather,
            mock.patch("os.path.isfile", return_value=cache_visible),
            mock.patch("os.makedirs"),
            mock.patch("builtins.open", mock.mock_open()),
            mock.patch("os.replace"),
            mock.patch.dict("os.environ", {"LOCAL_RANK": str(local_rank)}),
        ):
            result = _prepare_prompts_coordinated(
                config, object(), algorithm=algorithm, draft_config=object()
            )
        self.assertIs(result, prompts)
        self.assertEqual(gather.call_count, 2)
        return prepare

    def test_non_owner_ranks_read_the_rank_zero_cache(self):
        prepare = self._prepare(world_size=8, rank=3, local_rank=3, cache_visible=True)
        prepare.assert_called_once()

    def test_node_local_rank_zero_prepares_when_the_cache_is_not_visible(self):
        prepare = self._prepare(
            world_size=16, rank=8, local_rank=0, cache_visible=False
        )
        prepare.assert_called_once()


class ColocatedRunAssemblyTest(unittest.TestCase):
    def _config_and_bundle(self, **training):
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
                    **training,
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
            target_hidden_size=HIDDEN,
            target_vocab_size=16,
            draft_vocab_size=16,
            capture_layers=[1],
            strategy_kwargs={},
        )
        return algorithm, config, bundle

    def _build(self, config, bundle, algorithm, *, resume_state=None):
        target_capture = mock.Mock()
        trainer = object()
        patches = [
            mock.patch(
                "specforge.training.assembly.build_model_bundle", return_value=bundle
            ),
            mock.patch(
                "specforge.training.assembly._prepare_prompts", return_value=_prompts(4)
            ),
            mock.patch(
                "specforge.training.assembly._configured_logger",
                return_value=mock.Mock(),
            ),
            mock.patch(
                "specforge.torch_compat.configure_flex_attention_inductor",
                return_value=False,
            ),
            mock.patch(
                "specforge.offline_capture.load_offline_capture",
                return_value=target_capture,
            ),
            mock.patch(
                "specforge.launch.build_colocated_online_runtime", return_value=trainer
            ),
        ]
        if resume_state is not None:
            patches.append(
                mock.patch(
                    "specforge.training.checkpoint.CheckpointManager.read_resume_state",
                    return_value=resume_state,
                )
            )
        with mock.patch.multiple if False else _nested(patches) as mocks:
            run = build_training_run(config, algorithm=algorithm)
        load_capture, build_runtime = mocks[4], mocks[5]
        self.assertIsInstance(run, TrainingRun)
        self.assertIs(run.trainer, trainer)
        return target_capture, load_capture, build_runtime

    def test_assembly_sizes_the_local_engine_and_plans_the_stream(self):
        algorithm, config, bundle = self._config_and_bundle()

        target_capture, load_capture, build_runtime = self._build(
            config, bundle, algorithm
        )

        engine = load_capture.call_args.kwargs
        self.assertEqual(engine["context_length"], 128)
        self.assertEqual(engine["max_running_requests"], 1)
        self.assertEqual(engine["max_total_tokens"], 128)
        self.assertEqual(engine["dp_size"], 1)
        target_capture.set_capture_layers.assert_called_once_with(
            [1], capture_method="dflash"
        )
        runtime = build_runtime.call_args.kwargs
        self.assertEqual(runtime["dataset_size"], 4)
        self.assertEqual(runtime["total_steps"], 1)
        self.assertEqual(runtime["checkpoint_extra"]["prompt_plan"], "epoch-shard-v1")
        self.assertEqual(runtime["checkpoint_extra"]["target_dp_size"], 1)
        self.assertEqual(len(list(runtime["prompts"])), 4)
        self.assertIsInstance(runtime["feature_source"], LocalSGLangCaptureAdapter)

    def test_resume_from_a_mismatched_prompt_plan_is_rejected(self):
        algorithm, config, bundle = self._config_and_bundle(resume_from="/tmp/ckpt")

        with self.assertRaisesRegex(ValueError, "batch_size"):
            self._build(
                config,
                bundle,
                algorithm,
                resume_state={"epoch": 0, "epoch_samples": 1, "batch_size": 2},
            )

    def test_resume_skips_the_consumed_prefix_of_the_stream(self):
        algorithm, config, bundle = self._config_and_bundle(resume_from="/tmp/ckpt")

        _, _, build_runtime = self._build(
            config, bundle, algorithm, resume_state={"epoch": 0, "epoch_samples": 1}
        )

        runtime = build_runtime.call_args.kwargs
        self.assertEqual(len(list(runtime["prompts"])), 3)
        self.assertEqual(runtime["resume_state"]["epoch_samples"], 1)

    def test_completed_checkpoint_resumes_with_an_empty_stream(self):
        algorithm, config, bundle = self._config_and_bundle(resume_from="/tmp/ckpt")

        _, _, build_runtime = self._build(
            config, bundle, algorithm, resume_state={"epoch": 1, "epoch_samples": 0}
        )

        self.assertEqual(list(build_runtime.call_args.kwargs["prompts"]), [])


class _nested:
    """Enter several patchers; expose the started mocks as a list."""

    def __init__(self, patchers):
        self._patchers = patchers

    def __enter__(self):
        return [patcher.start() for patcher in self._patchers]

    def __exit__(self, *exc):
        for patcher in reversed(self._patchers):
            patcher.stop()
        return False


class LocalCaptureAdapterTest(unittest.TestCase):
    def test_adapter_emits_the_server_streaming_layout(self):
        adapter = LocalSGLangCaptureAdapter(
            _PackedCapture(), provider=_provider(), synchronize_after_capture=False
        )

        [features] = adapter.generate_features([_task(1)], capture=_capture_config())

        self.assertEqual(tuple(features["hidden_states"].shape), (1, 2, 12))
        self.assertEqual(tuple(features["target_last_hidden_states"].shape), (1, 2, 4))
        self.assertEqual(tuple(features["input_ids"].shape), (1, 2))
        self.assertTrue(features["attention_mask"].all())
        self.assertEqual(features["__aux_layer_ids__"], LAYERS)

    def test_adapter_detaches_only_this_tp_ranks_rows(self):
        adapter = LocalSGLangCaptureAdapter(
            _PackedCapture(),
            provider=_provider(),
            synchronize_after_capture=False,
            batch_partition=TargetBatchPartition(rank=1, size=2),
        )

        features = adapter.generate_features(
            [_task(index) for index in range(4)], capture=_capture_config()
        )

        self.assertTrue(adapter.returns_local_batch)
        self.assertEqual([row["input_ids"][0, 0].item() for row in features], [2, 3])
        for row in features:
            hidden = row["hidden_states"]
            # clone() severs each local row from the packed capture allocation.
            self.assertEqual(
                hidden.untyped_storage().nbytes(),
                hidden.numel() * hidden.element_size(),
            )


class _TPPeer:
    """One target-TP peer: private controller, store, worker, and stream."""

    def __init__(self, rank, size, prompts):
        partition = TargetBatchPartition(rank=rank, size=size)
        self.controller = DataFlowController("run", metadata_store=NoOpMetadataStore())
        self.store = LocalFeatureStore(f"run-{rank}")
        adapter = LocalSGLangCaptureAdapter(
            _PackedCapture(),
            provider=_provider(),
            synchronize_after_capture=False,
            batch_partition=partition,
        )
        self.worker = RolloutWorker(
            self.controller,
            self.store,
            adapter,
            _capture_config(),
            run_id="run",
            batch_partition=partition,
            feature_source_returns_local_batch=True,
        )
        self.stream = LocalRolloutStream(
            controller=self.controller,
            worker=self.worker,
            feature_store=self.store,
            prompts=iter(prompts),
            local_batch_size=1,
            target_batch_size=size,
        )

    def consume_one(self):
        """Mimic the loader: get, materialize (get + release), ack."""
        refs = self.stream.get(1)
        if not refs:
            return None
        (ref,) = refs
        tensors, handle = self.store.get(ref, device=torch.device("cpu"))
        self.store.release(handle, reason="loaded")
        self.stream.ack(refs)
        return ref.source_task_id, tensors


class LocalRolloutStreamTest(unittest.TestCase):
    def test_tp_peers_capture_in_lockstep_and_train_disjoint_slices(self):
        prompts = _prompts(8)
        peers = [_TPPeer(rank, 2, prompts) for rank in range(2)]
        consumed = {0: [], 1: []}

        with peers[0].stream, peers[1].stream:
            while True:
                results = [peer.consume_one() for peer in peers]
                if results == [None, None]:
                    break
                for rank, result in enumerate(results):
                    self.assertIsNotNone(result)
                    task_id, tensors = result
                    consumed[rank].append(task_id)
                    self.assertEqual(tuple(tensors["hidden_states"].shape), (1, 2, 12))
                for peer in peers:
                    self.assertLessEqual(peer.store.health()["resident_samples"], 1)

        self.assertEqual(consumed[0], [f"source-{i}" for i in (0, 2, 4, 6)])
        self.assertEqual(consumed[1], [f"source-{i}" for i in (1, 3, 5, 7)])
        for peer in peers:
            status = peer.controller.status()
            self.assertEqual(
                (
                    status["prompts_pending"],
                    status["prompts_leased"],
                    status["prompts_failed"],
                ),
                (0, 0, 0),
            )
            self.assertEqual(peer.stream.capture_calls, 4)
            self.assertEqual(peer.stream.peak_staged_samples, 1)
            self.assertEqual(peer.store.health()["resident_samples"], 0)

    def test_short_prompt_stream_fails_loud_instead_of_dropping_the_tail(self):
        peer = _TPPeer(0, 2, _prompts(3))

        self.assertIsNotNone(peer.consume_one())
        with self.assertRaisesRegex(RuntimeError, "multiple of the 2-prompt"):
            peer.stream.get(1)

    def test_stream_rejects_requests_above_the_local_batch(self):
        peer = _TPPeer(0, 1, _prompts(2))
        with self.assertRaisesRegex(ValueError, "stages at most 1"):
            peer.stream.get(2)

    def test_perf_metrics_report_interval_capture_and_peak_staging(self):
        peer = _TPPeer(0, 1, _prompts(2))
        peer.consume_one()
        peer.consume_one()

        metrics = peer.stream.perf_metrics()

        self.assertEqual(metrics["colocated_capture_calls"], 2.0)
        self.assertGreater(metrics["colocated_capture_time_s"], 0.0)
        self.assertGreater(metrics["colocated_capture_samples_per_second"], 0.0)
        self.assertEqual(metrics["colocated_peak_staged_samples"], 1.0)
        self.assertEqual(peer.stream.perf_metrics()["colocated_capture_calls"], 0.0)

    def test_close_stops_the_worker_and_drops_staged_features(self):
        peer = _TPPeer(0, 1, _prompts(2))
        peer.stream.get(1)  # staged, not yet materialized
        self.assertEqual(peer.store.health()["resident_samples"], 1)

        peer.stream.close(reason="test")

        self.assertEqual(peer.store.health()["resident_samples"], 0)
        self.assertEqual(peer.stream.get(1), [])


class BuildColocatedOnlineRuntimeTest(unittest.TestCase):
    def test_builder_wires_a_bounded_stream_into_the_shared_trainer_assembly(self):
        algorithm = builtin_algorithm_registry().resolve("dflash")
        feature_source = LocalSGLangCaptureAdapter(
            _PackedCapture(),
            provider=algorithm.providers.server_streaming_for("text"),
            synchronize_after_capture=False,
        )

        with (
            mock.patch(
                "specforge.inference.batch_partition.TargetBatchPartition.from_distributed",
                return_value=TargetBatchPartition(rank=1, size=2),
            ),
            mock.patch(
                "specforge.launch._assemble_trainer", return_value="trainer"
            ) as assemble,
        ):
            trainer = build_colocated_online_runtime(
                algorithm=algorithm,
                prompts=iter(_prompts(4)),
                feature_source=feature_source,
                draft_model=object(),
                target_head=None,
                optimizer_factory=object(),
                run_id="run",
                output_dir="/tmp/colocated-test",
                target_hidden_size=HIDDEN,
                aux_hidden_state_layer_ids=list(LAYERS),
                batch_size=1,
                tp_size=2,
                dataset_size=2,
                checkpoint_extra={"prompt_plan": "epoch-shard-v1"},
            )

        self.assertEqual(trainer, "trainer")
        kwargs = assemble.call_args.kwargs
        stream = kwargs["ref_source"]["queue"]
        self.assertIsInstance(stream, LocalRolloutStream)
        self.assertEqual((stream.local_batch_size, stream.target_batch_size), (1, 2))
        self.assertFalse(kwargs["ref_source"]["prepositioned"])
        self.assertIs(kwargs["fit_context"], stream)
        self.assertFalse(kwargs["clone_on_fetch"])
        self.assertFalse(kwargs["durable_ack"])
        self.assertEqual(kwargs["num_epochs"], 1)
        self.assertEqual(kwargs["dataset_size"], 2)
        self.assertEqual(feature_source.batch_partition.size, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
