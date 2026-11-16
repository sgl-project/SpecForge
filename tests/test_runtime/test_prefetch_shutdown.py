"""Real CPU loader/controller lifecycle, with explicit pending sample gates."""
import os
import tempfile
import threading
import time
import unittest
from concurrent import futures
from types import SimpleNamespace
from unittest import mock

import torch

from specforge.algorithms.common.hidden_states_data import build_collator
from specforge.runtime.contracts import TrainBatch
from specforge.runtime.data_plane import feature_dataloader as module
from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue
from specforge.training.controller import TrainerController, TrainerCore
from tests.test_runtime.test_trainer import FakeBackend, FakeStrategy


class Queue(SampleRefQueue):
    def __init__(self):
        super().__init__()
        self.get_calls = 0
        self.failures = []

    def get(self, n, timeout_s=0):
        self.get_calls += 1
        return super().get(n, timeout_s=timeout_s)

    def fail(self, refs, reason, retryable):
        self.failures.append(([ref.sample_id for ref in refs], retryable))
        return super().fail(refs, reason=reason, retryable=retryable)


class Store(LocalFeatureStore):
    def __init__(self, *, retain=False, block=False):
        super().__init__('bounded-loader')
        self.retain_on_release = retain
        self.block = block
        self.reads = []
        self.entered = threading.Event()
        self.proceed = threading.Event()

    def get(self, ref, *, device='cpu', names=None):
        self.reads.append(ref.sample_id)
        value = super().get(ref, device=device, names=names)
        if self.block and len(self.reads) == 1:
            self.entered.set()
            if not self.proceed.wait(3):
                raise RuntimeError('CPU test sample gate not released')
        return value

    def release(self, handle, *, reason='consumed'):
        if self.retain_on_release:
            with self._lock:
                self._active_leases.pop(handle.lease_token, None)
        else:
            super().release(handle, reason=reason)


def refs(store, n, *, direct=False):
    values = []
    for i in range(n):
        data = ({'input_ids': torch.arange(3).view(1, 3),
                 'loss_mask': torch.ones(1, 3, dtype=torch.int64),
                 'hidden_states': torch.ones(1, 3, 6),
                 'target_last_hidden_states': torch.ones(1, 3, 2)}
                if direct else {'x': torch.ones(2)})
        values.append(store.put(data, sample_id=f's{i}', metadata={
            'run_id': 'run', 'strategy': 'dflash' if direct else 'fake', 'num_tokens': 3}))
    return values


def configure(loader, limit):
    loader.configure_consumer_schedule(global_step=0, accumulation_steps=1,
        accumulation_remainder=0, log_interval=10, has_logger=True, max_batches=limit)


class TestRemainingBatchLimit(unittest.TestCase):
    def test_invalid_budget_or_nonfinite_drain_configuration_fails_early(self):
        loader = FeatureDataLoader(Store(), refs=[], strategy='fake')
        for limit in (-1, .5):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, 'max_batches'):
                configure(loader, limit)
        for setting in ('_PREFETCH_JOIN_TIMEOUT_S', '_PREFETCH_DRAIN_TIMEOUT_S'):
            for value in (0, -1, float('inf'), float('nan')):
                with self.subTest(setting=setting, value=value), mock.patch.object(module, setting, value):
                    with self.assertRaisesRegex(ValueError, 'finite and positive'):
                        FeatureDataLoader(Store(), refs=[], strategy='fake')

    def test_queue_never_leases_or_submits_beyond_limit(self):
        for workers in (0, 1, 4):
            for limit in (0, 1, 3):
                with self.subTest(workers=workers, limit=limit):
                    store, queue = Store(retain=True), Queue()
                    samples = refs(store, 12)
                    queue.put(samples)
                    loader = FeatureDataLoader(store, queue, batch_size=2,
                                               strategy='fake', num_workers=workers)
                    configure(loader, limit)
                    batches = list(loader)
                    loader.close()
                    self.assertEqual(len(batches), limit)
                    self.assertEqual(queue.get_calls, limit)
                    self.assertEqual(len(store.reads), 2 * limit)
                    self.assertEqual(queue.depth(), 12 - 2 * limit)
                    self.assertEqual(queue.in_flight(), 0)

    def test_refs_limit_applies_after_seek_before_worker_submission(self):
        store = Store(retain=True)
        samples = refs(store, 10)
        loader = FeatureDataLoader(store, refs=samples, batch_size=2,
                                   strategy='fake', num_workers=4)
        loader.seek(2)
        configure(loader, 1)
        self.assertEqual([batch.sample_ids for batch in loader], [['s4', 's5']])
        self.assertEqual(store.reads, ['s4', 's5'])

    def test_controller_threads_remaining_accumulation_and_resume_budget(self):
        store, queue = Store(retain=True), Queue()
        queue.put(refs(store, 20))
        loader = FeatureDataLoader(store, queue, strategy='fake', num_workers=4,
                                   ack=False)
        strategy = FakeStrategy()
        core = TrainerCore(strategy, FakeBackend(strategy.model), accumulation_steps=3)
        # One microbatch is already accumulated towards resumed optimizer step3.
        core.train_step(TrainBatch(['warmup'], 'fake', {'x': torch.ones(2)}, {}))
        with tempfile.TemporaryDirectory() as directory:
            controller = TrainerController(core, run_id='run', output_dir=directory,
                                           start_step=2, max_steps=4)
            self.assertEqual(controller.fit(loader), 4)
        loader.close()
        self.assertEqual(len(store.reads), 5)  # (4-2)*3 - remainder1
        self.assertEqual(queue.get_calls, 5)

    def test_controller_budget_recomputed_across_resumed_epochs(self):
        store = Store(retain=True)
        samples = refs(store, 3)
        loader = FeatureDataLoader(store, refs=samples, strategy='fake', num_workers=4)
        strategy = FakeStrategy()
        core = TrainerCore(strategy, FakeBackend(strategy.model), accumulation_steps=2)
        with tempfile.TemporaryDirectory() as directory:
            controller = TrainerController(core, run_id='run', output_dir=directory,
                start_step=1, max_steps=3, start_batch=1, start_samples=1, num_epochs=3)
            self.assertEqual(controller.fit(loader), 3)
        self.assertEqual(store.reads, ['s1', 's2', 's0', 's1'])


class TestCooperativeStop(unittest.TestCase):
    def assert_shutdown_errors_only(self, errors):
        for exc in errors:
            self.assertTrue(isinstance(exc, StopIteration) or
                            (type(exc) is RuntimeError and 'did not stop within' in str(exc)),
                            repr(exc))

    def pending(self, *, direct=False, retain=False, durable=False):
        store, queue = Store(retain=retain, block=True), Queue()
        queue.loader_close_retryable = durable
        samples = refs(store, 3, direct=direct)
        queue.put(samples)
        with mock.patch.dict(os.environ, {'SPECFORGE_TRAINER_DIRECT_BATCH': str(int(direct))}):
            loader = FeatureDataLoader(store, queue, batch_size=3, ack=False,
                num_workers=1, strategy='dflash' if direct else 'fake',
                collate_fn=build_collator() if direct else None)
        iterator, consumer_errors = iter(loader), []
        def consume():
            try:
                next(iterator)
            except BaseException as exc:
                consumer_errors.append(exc)
        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        self.assertTrue(store.entered.wait(1))
        return store, queue, samples, loader, iterator, consumer, consumer_errors

    def test_stop_at_sample_boundary_classifies_partial_consume_once_batch(self):
        for direct in (False, True):
            for durable in (False, True):
                with self.subTest(direct=direct, durable=durable):
                    store, queue, samples, loader, iterator, consumer, _ = self.pending(direct=direct, durable=durable)
                    errors = []
                    def close():
                        try:
                            loader.close()
                        except BaseException as exc:
                            errors.append(exc)
                    closer = threading.Thread(target=close, daemon=True)
                    closer.start()
                    try:
                        self.assertTrue(loader._prefetch_state.stop.wait(1))
                        self.assertTrue(closer.is_alive())
                        self.assertEqual(queue.failures, [])
                        self.assertEqual(store.health()['active_leases'], 1)
                    finally:
                        store.proceed.set()
                        closer.join(2)
                        consumer.join(2)
                        loader.close()
                        iterator.close()
                    self.assertFalse(closer.is_alive())
                    self.assertEqual(errors, [])
                    self.assertEqual(store.reads, ['s0'])
                    self.assertEqual({sid for ids, retry in queue.failures if not retry for sid in ids}, {'s0'})
                    self.assertEqual({sid for ids, retry in queue.failures if retry for sid in ids}, {'s1', 's2'})
                    self.assertEqual(queue.depth(), 2)
                    self.assertEqual(store.health()['active_leases'], 0)

    def test_late_sample_completion_drains_with_diagnostics_within_total_limit(self):
        store, queue, _, loader, iterator, consumer, _ = self.pending(retain=True)
        release = threading.Timer(.10, store.proceed.set)
        with mock.patch.object(module, '_PREFETCH_JOIN_TIMEOUT_S', .02), \
             mock.patch.object(module, '_PREFETCH_DRAIN_TIMEOUT_S', .4, create=True):
            release.start()
            try:
                loader.close()
            finally:
                store.proceed.set()
                release.join()
                consumer.join(1)
                loader.close()
                iterator.close()
        snapshot = loader.prefetch_shutdown_snapshot()
        self.assertEqual(snapshot['status'], 'STOPPED')
        self.assertGreaterEqual(snapshot['diagnostic_intervals'], 1)
        self.assertEqual(store.reads, ['s0'])
        self.assertEqual(queue.depth(), 3)

    def test_total_deadline_retains_running_read_then_reentrant_close_recovers(self):
        store, queue, _, loader, iterator, consumer, _ = self.pending(retain=True)
        try:
            with mock.patch.object(module, '_PREFETCH_JOIN_TIMEOUT_S', .02), \
                 mock.patch.object(module, '_PREFETCH_DRAIN_TIMEOUT_S', .07, create=True):
                started = time.monotonic()
                with self.assertRaisesRegex(RuntimeError, 'did not stop') as caught:
                    loader.close()
                self.assertLess(time.monotonic() - started, .5)
            snapshot = loader.prefetch_shutdown_snapshot()
            self.assertEqual(snapshot['status'], 'TIMED_OUT')
            self.assertIn('s0', str(snapshot))
            self.assertIn('get', str(snapshot['thread_stacks']))
            self.assertIn('progress=', str(caught.exception))
            self.assertEqual(queue.failures, [])
            self.assertEqual(store.health()['active_leases'], 1)
            self.assertIsNotNone(loader._prefetch_state)
            deadline = snapshot['deadline_monotonic']
            # Reentry cannot grant another budget to an undrained read.
            with self.assertRaisesRegex(RuntimeError, 'did not stop'):
                loader.close()
            self.assertEqual(loader.prefetch_shutdown_snapshot()['deadline_monotonic'], deadline)
        finally:
            store.proceed.set()
            state = loader._prefetch_state
            state.thread.join(1)
            self.assertFalse(futures.wait(state.live_fetches(), timeout=1).not_done)
            consumer.join(1)
            loader.close()
            loader.close()
            iterator.close()
        self.assertEqual(loader.prefetch_shutdown_snapshot()['status'], 'STOPPED')
        self.assertEqual(store.reads, ['s0'])
        self.assertEqual(queue.depth(), 3)
        self.assertEqual(store.health()['active_leases'], 0)

    def test_completed_future_timeout_error_is_not_swallowed_as_poll_timeout(self):
        store, queue = Store(), Queue()
        queue.put(refs(store, 3))
        loader = FeatureDataLoader(store, queue, batch_size=3, num_workers=1,
                                   strategy='fake', ack=False)
        # An emergency release prevents a broken regression from hanging tests.
        emergency = threading.Timer(.3, loader.close)
        with mock.patch.object(store, 'get', side_effect=TimeoutError('actual-fetch-timeout')):
            emergency.start()
            try:
                with self.assertRaisesRegex(TimeoutError, 'actual-fetch-timeout'):
                    next(iter(loader))
            finally:
                emergency.cancel()
                emergency.join()
                loader.close()
        self.assertEqual({sid for ids, retry in queue.failures if not retry for sid in ids}, {'s0'})
        self.assertEqual({sid for ids, retry in queue.failures if retry for sid in ids}, {'s1', 's2'})

    def test_release_failure_keeps_touched_sample_terminal_and_tail_retryable(self):
        store, queue = Store(), Queue()
        queue.put(refs(store, 3))
        loader = FeatureDataLoader(store, queue, batch_size=3, num_workers=1,
                                   strategy='fake', ack=False)
        release = store.release
        def failed_release(handle, *, reason):
            release(handle, reason=reason)
            raise RuntimeError('release-failed-after-consume')
        with mock.patch.object(store, 'release', failed_release):
            with self.assertRaisesRegex(RuntimeError, 'release-failed-after-consume'):
                next(iter(loader))
        loader.close()
        self.assertEqual(store.reads, ['s0'])
        self.assertEqual({sid for ids, retry in queue.failures if not retry for sid in ids}, {'s0'})
        self.assertEqual({sid for ids, retry in queue.failures if retry for sid in ids}, {'s1', 's2'})
        self.assertEqual(store.health()['active_leases'], 0)

    def test_submitter_join_and_fetch_share_one_absolute_deadline(self):
        class BlockingQueue(Queue):
            entered = threading.Event()
            proceed = threading.Event()
            def get(self, n, timeout_s=0):
                if self.get_calls == 1:
                    self.entered.set()
                    self.proceed.wait(3)
                return super().get(n, timeout_s=timeout_s)
        store, queue = Store(block=True, retain=True), BlockingQueue()
        queue.put(refs(store, 2))
        loader = FeatureDataLoader(store, queue, num_workers=1, ack=False, strategy='fake')
        iterator = iter(loader)
        errors = []
        def consume():
            try:next(iterator)
            except BaseException as exc:errors.append(exc)
        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        self.assertTrue(store.entered.wait(1) and queue.entered.wait(1))
        unblock_submitter = threading.Timer(.07, queue.proceed.set)
        try:
            with mock.patch.object(module, '_PREFETCH_JOIN_TIMEOUT_S', .02), \
                 mock.patch.object(module, '_PREFETCH_DRAIN_TIMEOUT_S', .12):
                unblock_submitter.start()
                started = time.monotonic()
                with self.assertRaisesRegex(RuntimeError, 'prefetch fetch did not stop'):
                    loader.close()
                self.assertLess(time.monotonic() - started, .18)
                self.assertEqual(queue.failures, [])
        finally:
            queue.proceed.set()
            store.proceed.set()
            unblock_submitter.join()
            state = loader._prefetch_state
            state.thread.join(1)
            self.assertFalse(futures.wait(state.live_fetches(), timeout=1).not_done)
            consumer.join(1)
            loader.close()
            iterator.close()
        self.assert_shutdown_errors_only(errors)

    def test_progress_does_not_extend_the_absolute_deadline(self):
        class GatedStore(Store):
            def __init__(self):
                super().__init__(retain=True)
                self.entered_samples = [threading.Event() for _ in range(3)]
                self.release_samples = [threading.Event() for _ in range(3)]
            def get(self, ref, **kwargs):
                value = super().get(ref, **kwargs)
                i = int(ref.sample_id[1:])
                self.entered_samples[i].set()
                self.release_samples[i].wait(3)
                return value
        store, queue = GatedStore(), Queue()
        queue.put(refs(store, 3))
        loader = FeatureDataLoader(store, queue, num_workers=3, ack=False, strategy='fake')
        iterator = iter(loader)
        errors = []
        def consume():
            try:next(iterator)
            except BaseException as exc:errors.append(exc)
        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        self.assertTrue(all(event.wait(1) for event in store.entered_samples))
        timers = [threading.Timer(delay, store.release_samples[i].set)
                  for i, delay in enumerate((.03, .07))]
        try:
            with mock.patch.object(module, '_PREFETCH_JOIN_TIMEOUT_S', .02), \
                 mock.patch.object(module, '_PREFETCH_DRAIN_TIMEOUT_S', .12):
                for timer in timers:timer.start()
                started = time.monotonic()
                with self.assertRaisesRegex(RuntimeError, 'did not stop'):
                    loader.close()
                self.assertLess(time.monotonic() - started, .18)
                snapshot = loader.prefetch_shutdown_snapshot()
                self.assertEqual(snapshot['materialized_samples'], 2)
                self.assertEqual(snapshot['status'], 'TIMED_OUT')
                self.assertEqual(store.health()['active_leases'], 1)
        finally:
            for event in store.release_samples:event.set()
            for timer in timers:timer.join()
            state = loader._prefetch_state
            state.thread.join(1)
            self.assertFalse(futures.wait(state.live_fetches(), timeout=1).not_done)
            consumer.join(1)
            loader.close()
            iterator.close()
        self.assert_shutdown_errors_only(errors)

    def test_real_mooncake_close_guard_survives_loader_timeout_and_late_get(self):
        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
        from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore
        backend = _FakeMooncakeStore()
        store = MooncakeFeatureStore(store=backend, store_id='loader-guard', retain_on_release=True)
        sample = store.put({'x': torch.ones(2)}, sample_id='s0',
                           metadata={'run_id': 'run', 'strategy': 'fake'})
        queue, entered, proceed = Queue(), threading.Event(), threading.Event()
        queue.put([sample])
        loader = FeatureDataLoader(store, queue, num_workers=1, ack=False, strategy='fake')
        original_get = backend.get_into
        def pending_get(*args):
            entered.set()
            proceed.wait(3)
            return original_get(*args)
        iterator = iter(loader)
        errors = []
        def consume():
            try:next(iterator)
            except BaseException as exc:errors.append(exc)
        consumer = threading.Thread(target=consume, daemon=True)
        with mock.patch.object(backend, 'get_into', pending_get):
            consumer.start()
            self.assertTrue(entered.wait(1))
            try:
                with mock.patch.object(module, '_PREFETCH_JOIN_TIMEOUT_S', .02), \
                     mock.patch.object(module, '_PREFETCH_DRAIN_TIMEOUT_S', .04):
                    with self.assertRaisesRegex(RuntimeError, 'did not stop'):
                        loader.close()
                self.assertEqual(store._active_get_calls, 1)
                self.assertTrue(backend._d)
                with self.assertRaisesRegex(RuntimeError, 'live readers'):
                    store.close()
                self.assertFalse(store._closed)
                self.assertEqual(queue.failures, [])
            finally:
                proceed.set()
                state = loader._prefetch_state
                state.thread.join(1)
                self.assertFalse(futures.wait(state.live_fetches(), timeout=1).not_done)
                consumer.join(1)
                loader.close()
                iterator.close()
                store.close()
        self.assertEqual(store._active_get_calls, 0)
        self.assertTrue(store._closed)
        self.assert_shutdown_errors_only(errors)


class TestExplicitIteratorClose(unittest.TestCase):
    def test_progress_failure_does_not_replace_training_or_iterator_failure(self):
        for training_failure in (True, False):
            with self.subTest(training_failure=training_failure):
                class Source:
                    def __iter__(self):return self
                    def __next__(self):return TrainBatch(['s'], 'fake', {'x': torch.ones(2)}, {})
                    def close(self):raise RuntimeError('iterator-close-error')
                strategy = FakeStrategy()
                core = TrainerCore(strategy, FakeBackend(strategy.model))
                with tempfile.TemporaryDirectory() as directory:
                    controller = TrainerController(core, run_id='run', output_dir=directory, max_steps=1)
                    progress = SimpleNamespace(update=lambda _: None,
                        close=mock.Mock(side_effect=RuntimeError('progress-close-error')))
                    with mock.patch.object(controller, '_make_progress_bar', return_value=progress):
                        if training_failure:
                            with mock.patch.object(core, 'train_step', side_effect=ValueError('primary-training-error')):
                                with self.assertRaisesRegex(ValueError, 'primary-training-error') as caught:
                                    controller.fit(Source())
                        else:
                            with self.assertRaisesRegex(RuntimeError, 'iterator-close-error') as caught:
                                controller.fit(Source())
                self.assertIn('progress-close-error', ' '.join(getattr(caught.exception, '__notes__', [])))

    def test_training_failure_preserved_when_iterator_close_also_fails(self):
        class BrokenIterator:
            closed = 0
            def __iter__(self):return self
            def __next__(self):return TrainBatch(['s'], 'fake', {'x': torch.ones(2)}, {})
            def close(self):
                self.closed += 1
                raise RuntimeError('iterator-close-error')
        source, strategy = BrokenIterator(), FakeStrategy()
        core = TrainerCore(strategy, FakeBackend(strategy.model))
        with tempfile.TemporaryDirectory() as directory:
            controller = TrainerController(core, run_id='run', output_dir=directory, max_steps=1)
            with mock.patch.object(core, 'train_step', side_effect=ValueError('primary-training-error')):
                with self.assertRaisesRegex(ValueError, 'primary-training-error') as caught:
                    controller.fit(source)
        self.assertEqual(source.closed, 1)
        self.assertIn('iterator-close-error', ' '.join(getattr(caught.exception, '__notes__', [])))

    def test_success_cannot_ignore_iterator_close_failure(self):
        class BrokenIterator:
            def __iter__(self):return self
            def __next__(self):return TrainBatch(['s'], 'fake', {'x': torch.ones(2)}, {})
            def close(self):raise RuntimeError('iterator-close-error')
        strategy = FakeStrategy()
        core = TrainerCore(strategy, FakeBackend(strategy.model))
        with tempfile.TemporaryDirectory() as directory:
            controller = TrainerController(core, run_id='run', output_dir=directory, max_steps=1)
            with self.assertRaisesRegex(RuntimeError, 'iterator-close-error'):
                controller.fit(BrokenIterator())


if __name__ == '__main__':
    unittest.main()
