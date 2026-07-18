# coding=utf-8
"""Deterministic tests for process-facing windowed capture runtime loops."""

from __future__ import annotations

import dataclasses
import inspect
import os
import sqlite3
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass
from unittest import mock

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.inference.capture import CaptureConfig
from specforge.launch import (
    build_disagg_online_windowed_consumer,
    build_disagg_online_windowed_producer,
    build_disagg_windowed_capture_contract,
)
from specforge.runtime.contracts import FeatureSpec, PromptTask, SampleRef
from specforge.runtime.control_plane.metadata_store import SQLiteMetadataStore
from specforge.runtime.data_plane.windowed_capture import (
    CaptureFailedError,
    SQLiteWindowedCaptureRegistry,
    WindowedCaptureQueue,
    capture_contract_digest,
)
from specforge.runtime.data_plane.windowed_capture_runtime import (
    WindowedCaptureService,
    WindowedConsumerControl,
    start_windowed_consumer_control,
)


class _OwnerStore:
    lifetime_owner = True

    def __init__(self) -> None:
        self.live: dict[str, int] = {}
        self.reclaimed: list[tuple[str, int, str]] = []

    def adopt(self, ref: SampleRef) -> None:
        self.live[ref.sample_id] = int(ref.metadata["generation"])

    def reclaim(self, ref: SampleRef, *, reason: str) -> None:
        generation = int(ref.metadata["generation"])
        self.reclaimed.append((ref.sample_id, generation, reason))
        if self.live.get(ref.sample_id) == generation:
            self.live.pop(ref.sample_id)

    def gc(self):
        return {}


class _ReaderStore:
    lifetime_owner = False


@dataclass(frozen=True)
class _Failure:
    task_id: str
    reason: str
    retryable: bool


class _RefSource:
    def __init__(self, store: _OwnerStore, *, fail_once: bool = False) -> None:
        self.store = store
        self.fail_once = fail_once
        self.calls = 0
        self.generations: dict[str, int] = {}

    def produce_refs(self, tasks, *, capture):
        del capture
        self.calls += 1
        out = []
        for task in tasks:
            if self.fail_once and self.calls == 1:
                out.append(_Failure(task.task_id, "transient", True))
                continue
            generation = self.generations.get(task.task_id, 0) + 1
            self.generations[task.task_id] = generation
            sample_id = f"run:{task.task_id}"
            ref = SampleRef(
                sample_id=sample_id,
                run_id="run",
                source_task_id=task.task_id,
                feature_store_uri=f"fixture://run/{sample_id}?generation={generation}",
                feature_keys={"input_ids": f"{sample_id}/input_ids"},
                feature_specs={
                    "input_ids": FeatureSpec(
                        name="input_ids", shape=(1, 4), dtype="int64"
                    )
                },
                strategy="dflash",
                estimated_bytes=64,
                metadata={"generation": generation},
            )
            self.store.adopt(ref)
            out.append(ref)
        return out


def _capture() -> CaptureConfig:
    return CaptureConfig.from_strategy(
        required_features={"input_ids"},
        aux_hidden_state_layer_ids=(),
        target_repr="hidden_state",
        target_hidden_size=8,
    )


def _prompts(total: int) -> list[PromptTask]:
    return [
        PromptTask(
            task_id=f"source-{index}",
            run_id="run",
            source_id="fixture",
            payload={"input_ids": [index, index + 1]},
            max_length=8,
        )
        for index in range(total)
    ]


class TestWindowedCaptureService(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

    def registry(self, *, max_live_refs=4, consumers=("a",)):
        registry = SQLiteWindowedCaptureRegistry(
            os.path.join(self.tempdir.name, "window.db"),
            max_live_refs=max_live_refs,
            max_live_bytes=max_live_refs * 100,
            capture_reservation_bytes=100,
            poll_s=0.001,
        )
        registry.initialize_run(
            run_id="run",
            contract_digest=capture_contract_digest(_capture()),
            source_sample_ids=[task.task_id for task in _prompts(12)],
            expected_consumers=consumers,
        )
        self.addCleanup(registry.close)
        return registry

    def test_rejects_negative_batch_wait(self):
        registry = self.registry()
        store = _OwnerStore()
        with self.assertRaisesRegex(ValueError, "batch_wait_s"):
            WindowedCaptureService(
                registry,
                prompts=_prompts(12),
                feature_source=_RefSource(store),
                capture=_capture(),
                owner_store=store,
                batch_wait_s=-0.001,
            )

    def test_physical_generation_is_not_rewritten_by_window_generation(self):
        registry = self.registry(max_live_refs=1)
        registry.register_consumer("a")
        ticket = registry.request_acquire("a", 0)
        request = registry.claim_batch(1)[0]
        ref = SampleRef(
            sample_id="run:source-0",
            run_id="run",
            source_task_id="source-0",
            feature_store_uri="fixture://run/source-0?generation=41",
            feature_keys={},
            feature_specs={},
            strategy="dflash",
            estimated_bytes=1,
            metadata={"generation": 41, "window_generation": request.generation},
        )
        registry.mark_committing(request, ref)
        registry.complete_capture(request, ref)

        lease = registry.wait_ready(ticket, timeout_s=1.0)
        self.assertEqual(lease.ref.metadata["generation"], 41)
        self.assertEqual(lease.ref.metadata["window_generation"], request.generation)

    def test_registry_rejects_capture_ref_from_another_run(self):
        registry = self.registry(max_live_refs=1)
        registry.register_consumer("a")
        registry.request_acquire("a", 0)
        request = registry.claim_batch(1)[0]
        ref = _RefSource(_OwnerStore()).produce_refs(
            [_prompts(1)[0]], capture=_capture()
        )[0]
        ref = dataclasses.replace(
            ref,
            run_id="another-run",
            metadata={**ref.metadata, "window_generation": request.generation},
        )

        with self.assertRaisesRegex(ValueError, "does not match registry run"):
            registry.mark_committing(request, ref)

    def test_heartbeat_treats_concurrent_completion_as_terminal_success(self):
        registry = self.registry(max_live_refs=1)
        registry.register_consumer("a", cursor=12)
        control = start_windowed_consumer_control(
            registry,
            "a",
            lookbehind=0,
            lookahead=0,
            prefetch_depth=0,
            max_outstanding=1,
            heartbeat_interval_s=0.01,
        )
        registry.complete_consumer("a")
        time.sleep(0.03)

        control.ensure_healthy()
        control.close()

    def test_heartbeat_recovers_from_a_transient_sqlite_failure(self):
        registry = mock.Mock()
        registry.heartbeat.side_effect = [
            None,
            sqlite3.OperationalError("database is locked"),
            None,
        ]
        control = WindowedConsumerControl(
            registry=registry,
            consumer_id="a",
            heartbeat_interval_s=0.001,
        ).start()
        deadline = time.monotonic() + 1.0
        while registry.heartbeat.call_count < 3 and time.monotonic() < deadline:
            time.sleep(0.001)

        control.ensure_healthy()
        control.close()
        self.assertGreaterEqual(registry.heartbeat.call_count, 3)

    def test_external_ledger_failure_abandons_queue_lease(self):
        registry = self.registry(max_live_refs=1)
        registry.register_consumer("a")
        ticket = registry.request_acquire("a", 0)
        request = registry.claim_batch(1)[0]
        source = _RefSource(_OwnerStore())
        ref = source.produce_refs([_prompts(1)[0]], capture=_capture())[0]
        ref = SampleRef(
            **{
                **ref.__dict__,
                "metadata": {**ref.metadata, "window_generation": request.generation},
            }
        )
        registry.mark_committing(request, ref)
        registry.complete_capture(request, ref)
        registry.cancel_acquire(ticket)
        queue = WindowedCaptureQueue(
            registry,
            "a",
            idle_timeout_s=1.0,
            record_refs=lambda _refs: (_ for _ in ()).throw(RuntimeError("ledger")),
        )

        with self.assertRaisesRegex(RuntimeError, "ledger"):
            queue.get(1)
        self.assertEqual(registry.snapshot()["leases"], 0)

    def test_skewed_1p3c_and_partial_completion_stay_bounded(self):
        consumers = ("fast", "medium", "short")
        registry = self.registry(max_live_refs=4, consumers=consumers)
        registry.register_consumer(
            "fast", lookahead=3, prefetch_depth=4, max_outstanding=1
        )
        registry.register_consumer(
            "medium", lookahead=1, prefetch_depth=2, max_outstanding=1
        )
        registry.register_consumer("short", max_outstanding=1)
        store = _OwnerStore()
        source = _RefSource(store)
        service = WindowedCaptureService(
            registry,
            prompts=_prompts(12),
            feature_source=source,
            capture=_capture(),
            owner_store=store,
            capture_batch_size=3,
            consumer_registration_timeout_s=1.0,
            consumer_heartbeat_timeout_s=30.0,
            poll_s=0.001,
        )
        delivered = {consumer: [] for consumer in consumers}
        errors = []

        def consume(consumer: str, count: int, delay: float) -> None:
            try:
                queue = WindowedCaptureQueue(registry, consumer, idle_timeout_s=2.0)
                while len(delivered[consumer]) < count:
                    refs = queue.get(1)
                    if not refs:
                        break
                    delivered[consumer].append(refs[0].source_task_id)
                    queue.ack(refs)
                    if delay:
                        time.sleep(delay)
                if count < 12:
                    queue.complete(allow_partial=True)
                else:
                    self.assertEqual(queue.get(1), [])
            except BaseException as exc:
                errors.append(exc)

        service_result = []

        def produce() -> None:
            try:
                service_result.append(service.drive(max_rounds=100_000))
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=produce),
            threading.Thread(target=consume, args=("fast", 12, 0.0)),
            threading.Thread(target=consume, args=("medium", 12, 0.001)),
            threading.Thread(target=consume, args=("short", 3, 0.002)),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10.0)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        self.assertEqual(errors, [])
        self.assertEqual(delivered["fast"], [f"source-{i}" for i in range(12)])
        self.assertEqual(delivered["medium"], [f"source-{i}" for i in range(12)])
        self.assertEqual(delivered["short"], [f"source-{i}" for i in range(3)])
        snapshot = registry.snapshot()
        self.assertLessEqual(snapshot["peak_live_refs"], 4)
        self.assertLessEqual(snapshot["peak_live_bytes"], 400)
        self.assertEqual(snapshot["live_refs"], 0)
        self.assertEqual(snapshot["status"], "completed")
        self.assertTrue(service_result)
        self.assertEqual(store.live, {})

    def test_retryable_source_result_is_retried(self):
        registry = self.registry(max_live_refs=2)
        registry.register_consumer("a")
        store = _OwnerStore()
        source = _RefSource(store, fail_once=True)
        service = WindowedCaptureService(
            registry,
            prompts=_prompts(12),
            feature_source=source,
            capture=_capture(),
            owner_store=store,
            max_capture_retries=1,
            retry_backoff_s=0,
            consumer_registration_timeout_s=1.0,
            consumer_heartbeat_timeout_s=30.0,
            poll_s=0.001,
        )
        queue = WindowedCaptureQueue(registry, "a", idle_timeout_s=2.0)
        errors = []
        producer = threading.Thread(
            target=lambda: self._drive_or_record(service, errors)
        )
        producer.start()
        received = []
        while True:
            refs = queue.get(1)
            if not refs:
                break
            received.extend(ref.source_task_id for ref in refs)
            queue.ack(refs)
        producer.join(timeout=5.0)

        self.assertFalse(producer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(received, [f"source-{i}" for i in range(12)])
        self.assertGreaterEqual(service.capture_failures, 1)

    def test_non_retryable_capture_failure_terminates_and_reclaims(self):
        registry = self.registry(max_live_refs=2)
        registry.register_consumer("a")
        store = _OwnerStore()
        service = WindowedCaptureService(
            registry,
            prompts=_prompts(12),
            feature_source=_RefSource(store, fail_once=True),
            capture=_capture(),
            owner_store=store,
            max_capture_retries=0,
            retry_backoff_s=0,
            consumer_registration_timeout_s=1.0,
            consumer_heartbeat_timeout_s=30.0,
            poll_s=0.001,
        )
        queue = WindowedCaptureQueue(registry, "a", idle_timeout_s=2.0)
        errors = []
        producer = threading.Thread(
            target=lambda: self._drive_or_record(service, errors)
        )
        producer.start()

        with self.assertRaisesRegex(CaptureFailedError, "transient"):
            queue.get(1)
        queue.close("capture failed")
        producer.join(timeout=5.0)

        self.assertFalse(producer.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(service.capture_failures, 1)
        self.assertEqual(registry.snapshot()["status"], "completed_with_failures")
        self.assertEqual(store.live, {})

    @staticmethod
    def _drive_or_record(service, errors):
        try:
            service.drive(max_rounds=100_000)
        except BaseException as exc:
            errors.append(exc)


class TestWindowedLaunchBuilders(unittest.TestCase):
    def test_dflash_window_contract_does_not_default_to_logits(self):
        capture, _digest = build_disagg_windowed_capture_contract(
            strategy="dflash",
            target_hidden_size=8,
            target_model_version="fixture",
            tokenizer_version="fixture",
        )
        self.assertIsNone(capture.target_repr)

    def test_shared_assembler_forwards_loader_prefetch_depth(self):
        from specforge.launch import _assemble_trainer

        assembled = mock.Mock()
        assembled.controller = mock.sentinel.trainer
        assembled.loader = mock.sentinel.loader
        with mock.patch(
            "specforge.training.Trainer", return_value=assembled
        ) as trainer:
            result = _assemble_trainer(
                algorithm=builtin_algorithm_registry().resolve("dflash"),
                controller=mock.sentinel.controller,
                store=mock.sentinel.store,
                ref_source={"queue": mock.sentinel.queue},
                model=mock.sentinel.model,
                target_head=None,
                optimizer_factory=mock.sentinel.optimizer,
                run_id="run",
                output_dir="output",
                batch_size=1,
                accumulation_steps=1,
                num_epochs=1,
                max_steps=1,
                save_interval=0,
                eval_interval=0,
                tp_size=1,
                sp_ulysses_size=1,
                sp_ring_size=1,
                logger=None,
                log_interval=1,
                collate_fn=mock.sentinel.collate,
                dataloader_num_workers=3,
            )

        self.assertIs(result, assembled)
        self.assertEqual(trainer.call_args.kwargs["dataloader_num_workers"], 3)

    def test_producer_requires_stable_task_ids(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "stable task_id"):
                build_disagg_online_windowed_producer(
                    prompts=[{"payload": {"input_ids": [1]}}],
                    feature_store=_OwnerStore(),
                    feature_source=_RefSource(_OwnerStore()),
                    run_id="run",
                    consumer_ids=("a",),
                    registry_db_path=os.path.join(root, "window.db"),
                    max_live_refs=2,
                    target_hidden_size=8,
                    target_model_version="fixture",
                    tokenizer_version="fixture",
                    strategy="dflash",
                    target_repr="hidden_state",
                )

    def test_consumer_builder_ledgers_window_refs_and_validates_capacity(self):
        from specforge.launch import _assemble_trainer

        with tempfile.TemporaryDirectory() as root:
            owner = _OwnerStore()
            producer = build_disagg_online_windowed_producer(
                prompts=[
                    {
                        "task_id": "source-0",
                        "payload": {"input_ids": [1, 2]},
                    }
                ],
                feature_store=owner,
                feature_source=_RefSource(owner),
                run_id="run",
                consumer_ids=("a",),
                registry_db_path=os.path.join(root, "window.db"),
                max_live_refs=2,
                target_hidden_size=8,
                target_model_version="fixture",
                tokenizer_version="fixture",
                strategy="dflash",
                target_repr="hidden_state",
            )
            fake_trainer = mock.Mock()
            fake_loader = mock.Mock()
            fake_trainer.loader = fake_loader
            fake_loader.metrics.return_value = {
                "stages": {"store_get": {"count": 1}},
                "counters": {},
            }
            with mock.patch(
                "specforge.launch._assemble_trainer",
                return_value=fake_trainer,
            ) as assemble:
                runtime = build_disagg_online_windowed_consumer(
                    consumer_id="a",
                    registry_db_path=os.path.join(root, "window.db"),
                    max_live_refs=2,
                    contract_digest=producer.contract_digest,
                    total_samples=1,
                    feature_store=_ReaderStore(),
                    eagle3_model=object(),
                    optimizer_factory=mock.Mock(),
                    run_id="run",
                    output_dir=os.path.join(root, "output"),
                    metadata_db_path=os.path.join(root, "consumer.db"),
                    strategy="dflash",
                    max_outstanding=2,
                    batch_size=1,
                    accumulation_steps=2,
                    initialization_timeout_s=1.0,
                    heartbeat_interval_s=0.1,
                )
            try:
                self.assertIsNone(runtime.controller.sample_queue)
                self.assertTrue(assemble.call_args.kwargs["durable_ack"])
                unexpected = set(assemble.call_args.kwargs) - set(
                    inspect.signature(_assemble_trainer).parameters
                )
                self.assertEqual(unexpected, set())
                self.assertEqual(
                    runtime.accounting_snapshot()["input_pipeline"],
                    fake_loader.metrics.return_value,
                )
                self.assertEqual(runtime.accounting_snapshot()["queue"]["refs"], 0)
            finally:
                runtime.control.fail("test cleanup")
                runtime.close()
                producer.close()

    def test_consumer_resume_requires_checkpoint_for_durable_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            registry_path = os.path.join(root, "window.db")
            registry = SQLiteWindowedCaptureRegistry(
                registry_path,
                max_live_refs=2,
                poll_s=0.001,
            )
            digest = capture_contract_digest(_capture())
            registry.initialize_run(
                run_id="run",
                contract_digest=digest,
                source_sample_ids=("source-0",),
                expected_consumers=("a",),
            )
            registry.close()
            metadata_path = os.path.join(root, "consumer.db")
            metadata = SQLiteMetadataStore(metadata_path)
            ref = _RefSource(_OwnerStore()).produce_refs(
                [_prompts(1)[0]], capture=_capture()
            )[0]
            metadata.commit_sample(ref)
            metadata.record_train_ack(
                [ref.sample_id], global_step=1, optimizer_durable=True
            )
            metadata.close()

            with self.assertRaisesRegex(ValueError, "no resume_from checkpoint"):
                build_disagg_online_windowed_consumer(
                    consumer_id="a",
                    registry_db_path=registry_path,
                    max_live_refs=2,
                    contract_digest=digest,
                    total_samples=1,
                    feature_store=_ReaderStore(),
                    eagle3_model=object(),
                    optimizer_factory=mock.Mock(),
                    run_id="run",
                    output_dir=os.path.join(root, "output"),
                    metadata_db_path=metadata_path,
                    strategy="dflash",
                    resume=True,
                    initialization_timeout_s=1.0,
                )

    def test_producer_recovery_reclaims_committing_generation_and_replays(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "window.db")
            prompt = {"task_id": "source-0", "payload": {"input_ids": [1, 2]}}
            store = _OwnerStore()
            source = _RefSource(store)
            first = build_disagg_online_windowed_producer(
                prompts=[prompt],
                feature_store=store,
                feature_source=source,
                run_id="run",
                consumer_ids=("a",),
                registry_db_path=path,
                max_live_refs=2,
                target_hidden_size=8,
                target_model_version="fixture",
                tokenizer_version="fixture",
                strategy="dflash",
                target_repr="hidden_state",
            )
            first.registry.register_consumer("a")
            first.registry.request_acquire("a", 0)
            request = first.registry.claim_batch(1)[0]
            ref = source.produce_refs([_prompts(1)[0]], capture=_capture())[0]
            ref = dataclasses.replace(
                ref,
                metadata={
                    **ref.metadata,
                    "window_generation": request.generation,
                },
            )
            first.registry.mark_committing(request, ref)
            first.close()

            resumed = build_disagg_online_windowed_producer(
                prompts=[prompt],
                feature_store=store,
                feature_source=source,
                run_id="run",
                consumer_ids=("a",),
                registry_db_path=path,
                max_live_refs=2,
                target_hidden_size=8,
                target_model_version="fixture",
                tokenizer_version="fixture",
                strategy="dflash",
                target_repr="hidden_state",
                recover=True,
                consumer_registration_timeout_s=1.0,
                consumer_heartbeat_timeout_s=30.0,
                registry_poll_s=0.001,
            )
            resumed.registry.resume_consumer("a", durable_cursor=0)
            queue = WindowedCaptureQueue(resumed.registry, "a", idle_timeout_s=2.0)
            errors = []
            thread = threading.Thread(
                target=lambda: TestWindowedCaptureService._drive_or_record(
                    resumed.service, errors
                )
            )
            thread.start()
            refs = queue.get(1)
            queue.ack(refs)
            self.assertEqual(queue.get(1), [])
            thread.join(timeout=5.0)
            try:
                self.assertFalse(thread.is_alive())
                self.assertEqual(errors, [])
                self.assertEqual([item[1] for item in store.reclaimed], [1, 2])
                self.assertEqual(refs[0].metadata["generation"], 2)
                self.assertEqual(resumed.registry.snapshot()["status"], "completed")
            finally:
                resumed.close()


if __name__ == "__main__":
    unittest.main()
