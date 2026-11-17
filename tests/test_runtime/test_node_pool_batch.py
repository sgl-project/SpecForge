"""Real CPU SQLite/mmap tests for bounded metadata windows and failure claims."""
from contextlib import contextmanager
from concurrent.futures import CancelledError
import multiprocessing as mp
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import torch

from specforge.runtime.data_plane.node_content_pool import NodeContentPool, PoolFetchFailed, PoolPin
from specforge.runtime.data_plane.node_pool_batch import process_gate


def crossed_windows(directory, barrier, results, fetch_count, reverse):
    os.environ['SPECFORGE_TRAINER_NODE_POOL_BATCHED'] = '1'
    pool = NodeContentPool(directory, 65536)
    def fetch():
        assert not pool._db.in_transaction
        with fetch_count.get_lock():
            fetch_count.value += 1
        time.sleep(.03)
        return torch.arange(8, dtype=torch.uint8)
    barrier.wait(20)
    keys = ['B', 'A'] if reverse else ['A', 'B']
    with pool.acquire_many([(key, 8, fetch) for key in keys]) as window:
        results.put([pin.tensor.tolist() for pin in window.pins])
        barrier.wait(20)
    pool.close()


def foreign_waiting_for_publication(directory, ready, results):
    os.environ['SPECFORGE_TRAINER_NODE_POOL_BATCHED'] = '0'
    pool = NodeContentPool(directory, 65536)
    def fetch():
        ready.set()
        deadline = time.monotonic() + 10
        while pool._db.execute("SELECT state FROM entries WHERE key='local'").fetchone() != ('READY',):
            if time.monotonic() > deadline:
                raise TimeoutError('local writer was not published before foreign wait')
            time.sleep(.01)
        return torch.ones(8, dtype=torch.uint8)
    pin = pool.acquire('foreign', 8, fetch)
    results.put(True)
    pin.release()
    pool.close()


class TestNodePoolBatch(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, {'SPECFORGE_TRAINER_NODE_POOL_BATCHED': '1',
            'SPECFORGE_TRAINER_NODE_POOL_WINDOW_BYTES': str(64 << 20),
            'SPECFORGE_TRAINER_NODE_POOL_WINDOW_BLOCKS': '8',
            'SPECFORGE_TRAINER_NODE_POOL_WINDOW_INFLIGHT': '1'})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.assertIsNone(process_gate().current())

    def request(self, key, value=1):
        return key, 8, lambda: torch.full((8,), value, dtype=torch.uint8)

    def test_all_hit_and_miss_transactions_are_batched_and_old_off_path_remains(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = NodeContentPool(directory, 65536)
            requests = [self.request(str(i), i) for i in range(8)]
            sql = []
            pool._db.set_trace_callback(sql.append)
            with pool.acquire_many(requests) as window:
                self.assertEqual([p.tensor[0].item() for p in window.pins], list(range(8)))
            self.assertEqual(sql.count('BEGIN IMMEDIATE'), 3)
            self.assertEqual(sql.count('COMMIT'), 3)
            sql.clear()
            with pool.acquire_many([(k, n, lambda: self.fail('hot fetch')) for k, n, _ in requests]):
                pass
            self.assertEqual(sql.count('COMMIT'), 2)
            with self.assertRaisesRegex(RuntimeError, 'requires acquire_many'):
                pool.acquire('bypass', 8, lambda: self.fail('window bypass'))
            pool.close()
            with mock.patch.dict(os.environ, {'SPECFORGE_TRAINER_NODE_POOL_BATCHED': '0'}):
                old = NodeContentPool(directory, 65536)
            for key, n, fetch in requests:
                old.acquire(key, n, fetch).release()
            sql = []
            old._db.set_trace_callback(sql.append)
            for key, n, _ in requests:
                old.acquire(key, n, lambda: self.fail('hot fetch')).release()
            self.assertEqual(sql.count('COMMIT'), 16)
            old.close()

    def test_mixed_hit_miss_and_duplicate_keys_have_one_token_and_no_tx_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = NodeContentPool(directory, 65536)
            with pool.acquire_many([self.request('hit')]):
                pass
            calls, releases = [], []
            def fetch():
                self.assertFalse(pool._db.in_transaction)
                calls.append(1)
                return torch.arange(8, dtype=torch.uint8), lambda: releases.append(1)
            with pool.acquire_many([self.request('hit'), ('new', 8, fetch), ('new', 8, fetch)]) as window:
                self.assertIs(window.pins[1], window.pins[2])
                self.assertEqual(pool.health()['node_pool_active_pins'], 2)
                self.assertEqual(window.pins[2].tensor.tolist(), list(range(8)))
            self.assertEqual(calls, [1])
            self.assertEqual(releases, [1])
            self.assertEqual(pool.health()['node_pool_active_pins'], 0)
            self.assertEqual(pool._batch_high_water['registered_pin_tokens'], 2)
            pool.close()

    def test_two_process_crossed_windows_singleflight_without_deadlock(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx = mp.get_context('spawn')
            barrier, results, count = ctx.Barrier(2), ctx.Queue(), ctx.Value('i', 0)
            workers = [ctx.Process(target=crossed_windows, args=(directory, barrier, results, count, reverse)) for reverse in (False, True)]
            for worker in workers:
                worker.start()
            values = [results.get(timeout=25) for _ in workers]
            for worker in workers:
                worker.join(25)
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual(values[0], values[1])
            self.assertEqual(count.value, 2)
            self.assertFalse((Path(directory)/'data.mmap').exists())

    def test_own_writers_publish_before_waiting_foreign_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = NodeContentPool(directory, 65536)
            ctx = mp.get_context('spawn')
            ready, results = ctx.Event(), ctx.Queue()
            child = ctx.Process(target=foreign_waiting_for_publication, args=(directory, ready, results))
            child.start()
            self.assertTrue(ready.wait(20))
            with pool.acquire_many([('foreign', 8, lambda: self.fail('duplicate foreign fetch')), self.request('local')]) as window:
                self.assertTrue(all(pin is not None for pin in window.pins))
            self.assertTrue(results.get(timeout=20))
            child.join(20)
            self.assertEqual(child.exitcode, 0)
            pool.close()

    def test_capacity_fallback_does_not_evict_a_window_pin(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = NodeContentPool(directory, 4096)
            with pool.acquire_many([self.request('A'), self.request('B')]) as window:
                self.assertIsNotNone(window.pins[0])
                self.assertIsNone(window.pins[1])
                self.assertEqual(pool.health()['node_pool_active_pins'], 1)
                self.assertEqual(window.snapshot()['window_extent_bytes'], 8192)
            self.assertEqual(pool.health()['node_pool_fallbacks'], 1)
            pool.close()

    def test_partial_fill_failure_settles_completed_and_unstarted_writers(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = NodeContentPool(directory, 65536)
            error = RuntimeError('fetch sentinel')
            window = pool.acquire_many([self.request('completed'), ('failed', 8, lambda: (_ for _ in ()).throw(error)),
                                        ('unstarted', 8, lambda: self.fail('tail fetch'))])
            with self.assertRaises(RuntimeError) as caught:
                with window:
                    self.fail('acquisition should fail')
            self.assertIs(caught.exception, error)
            self.assertEqual(pool._db.execute("SELECT state,extent FROM entries WHERE key='completed'").fetchone(), ('READY', 4096))
            for key in ('failed','unstarted'):
                self.assertEqual(pool._db.execute('SELECT state,extent FROM entries WHERE key=?',(key,)).fetchone(), ('FAILED',0))
            self.assertEqual(pool.health()['node_pool_active_pins'], 0)
            self.assertEqual(pool.health()['node_pool_inflight_bytes'], 0)
            self.assertIsNone(process_gate().current())
            with pool.acquire_many([('completed',8,lambda:self.fail('completed fill discarded'))]):
                pass
            pool.close()

    def test_cancel_after_one_copy_fences_tail_and_preserves_completed_content(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = NodeContentPool(directory, 65536)
            stop = threading.Event()
            def first():
                stop.set()
                return torch.ones(8,dtype=torch.uint8)
            with self.assertRaises(CancelledError):
                with pool.acquire_many([('first',8,first),('tail',8,lambda:self.fail('cancelled tail'))], cancelled=stop.is_set):
                    pass
            self.assertEqual(pool._db.execute("SELECT state FROM entries WHERE key='first'").fetchone(), ('READY',))
            self.assertEqual(pool._db.execute("SELECT state,extent FROM entries WHERE key='tail'").fetchone(), ('FAILED',0))
            self.assertIsNone(process_gate().current())
            pool.close()

    def test_release_failure_retains_tokens_and_gate_until_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = NodeContentPool(directory, 65536)
            window = pool.acquire_many([self.request('a')])
            window.__enter__()
            pin = window.pins[0]
            token = pin.token
            original = pool._transaction
            @contextmanager
            def failed_release(scope='unspecified'):
                with original(scope) as db:
                    yield db
                    if scope=='batch_release':
                        raise RuntimeError('release commit failed')
            with mock.patch.object(pool,'_transaction',failed_release):
                with self.assertRaisesRegex(RuntimeError,'release commit failed'):
                    window.close()
            self.assertEqual(pin.token,token)
            self.assertIsNotNone(pin.tensor)
            self.assertEqual(pool.health()['node_pool_active_pins'],1)
            with self.assertRaisesRegex(RuntimeError,'requires proven cleanup'):
                with pool.acquire_many([self.request('b')]):pass
            window.close()
            self.assertIsNone(pin.token)
            self.assertIsNone(process_gate().current())
            pool.close()

    def test_admission_rollback_leaves_no_pin_or_writer_and_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            pool=NodeContentPool(directory,65536)
            with pool.acquire_many([self.request('hot')]):pass
            error=RuntimeError('admission sentinel')
            original=pool._transaction
            @contextmanager
            def failure(scope='unspecified'):
                with original(scope) as db:
                    yield db
                    if scope=='batch_lookup':raise error
            with mock.patch.object(pool,'_transaction',failure):
                with self.assertRaises(RuntimeError) as caught:
                    with pool.acquire_many([self.request('hot'),('new',8,lambda:self.fail('fetch before commit'))]):pass
            self.assertIs(caught.exception,error)
            self.assertEqual(pool.health()['node_pool_active_pins'],0)
            self.assertEqual(pool.health()['node_pool_inflight_bytes'],0)
            self.assertIsNone(pool._db.execute("SELECT key FROM entries WHERE key='new'").fetchone())
            self.assertIsNone(process_gate().current())
            with pool.acquire_many([self.request('new')]):pass
            pool.close()

    def test_source_cleanup_failure_does_not_mask_primary_or_lose_callback(self):
        with tempfile.TemporaryDirectory() as directory:
            pool=NodeContentPool(directory,65536);primary=RuntimeError('copy sentinel');calls=[]
            def release():
                calls.append(1)
                if len(calls)==1:raise RuntimeError('release secondary')
            with mock.patch('specforge.runtime.data_plane.node_pool_batch.torch.frombuffer',side_effect=primary):
                with self.assertRaises(RuntimeError) as caught:
                    with pool.acquire_many([('bad',8,lambda:(torch.ones(8,dtype=torch.uint8),release))]):pass
            self.assertIs(caught.exception,primary)
            self.assertTrue(any('release secondary' in note for note in primary.__notes__))
            self.assertEqual(calls,[1,1])
            self.assertIsNone(process_gate().current())
            pool.close()

    def test_real_copy_preparation_error_drops_shared_view_with_live_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            pool=NodeContentPool(directory,65536)
            captured=None
            try:
                # Valid contiguous8-byte CPU source; dtype reinterpretation of
                # a scalar fails after the destination frombuffer view exists.
                with pool.acquire_many([('scalar',8,lambda:torch.tensor(1,dtype=torch.int64))]):pass
            except RuntimeError as exc:
                captured=exc
            self.assertIsNotNone(captured)
            traceback=captured.__traceback__
            seen=False
            while traceback is not None:
                if traceback.tb_frame.f_code.co_name=='_fill_owned':
                    seen=True
                    self.assertIsNone(traceback.tb_frame.f_locals['shared'])
                traceback=traceback.tb_next
            self.assertTrue(seen)
            pool.close()
            self.assertTrue(pool._mapping.closed)

    def test_release_many_wrong_key_or_generation_does_not_unpin_real_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            pool=NodeContentPool(directory,65536)
            with pool.acquire_many([self.request('real')]) as window:
                pin=window.pins[0];token=pin.token
                for key,generation in [('wrong',pin.generation),('real','wrong-generation')]:
                    stale=PoolPin(pool,key,generation,token,None)
                    pool.release_many([stale])
                    self.assertEqual(pin.token,token)
                    self.assertEqual(pool.health()['node_pool_active_pins'],1)
                self.assertEqual(window.snapshot()['registered_pin_tokens'],1)
            self.assertEqual(pool.health()['node_pool_active_pins'],0)
            pool.close()

    def test_window_reentry_is_rejected_before_new_pin_or_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            pool=NodeContentPool(directory,65536)
            window=pool.acquire_many([self.request('once')])
            with window:
                with self.assertRaisesRegex(RuntimeError,'one-shot'):window.__enter__()
                self.assertEqual(pool.health()['node_pool_active_pins'],1)
            with self.assertRaisesRegex(RuntimeError,'one-shot'):
                with window:pass
            self.assertEqual(pool.health()['node_pool_active_pins'],0)
            self.assertIsNone(process_gate().current())
            with pool.acquire_many([self.request('fresh')]):pass
            pool.close()

    def test_late_completion_retains_process_gate_across_multiple_pool_objects(self):
        with tempfile.TemporaryDirectory() as directory:
            a=NodeContentPool(directory+'/a',65536)
            b=NodeContentPool(directory+'/b',65536,wait_timeout_s=.03)
            window=a.acquire_many([self.request('pin')]);window.__enter__()
            with self.assertRaises(TimeoutError):
                with b.acquire_many([self.request('other')]):pass
            window.defer(0);window.close()
            self.assertEqual(a.health()['node_pool_active_pins'],1)
            with self.assertRaisesRegex(RuntimeError,'requires proven cleanup'):
                with b.acquire_many([self.request('other')]):pass
            # Represents actual late completion; no thread/native operation is
            # killed and the gate remains charged until this explicit boundary.
            window.pins[0].release();window.resolve_deferred(0)
            with b.acquire_many([self.request('other')]):pass
            self.assertEqual(a._batch_high_water['active_windows'],1)
            self.assertEqual(b._batch_high_water['active_windows'],1)
            a.close();b.close()

    def test_oversize_fallback_and_hard_limits_precede_any_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            pool=NodeContentPool(directory,65536)
            called=[]
            with pool.acquire_many([('oversize',(8<<20)+1,lambda:called.append(1))]) as window:
                self.assertEqual(window.pins,[None])
                self.assertEqual(window.snapshot()['oversize_fallback_bytes'],(8<<20)+1)
                self.assertLessEqual(window.snapshot()['window_extent_bytes'],64<<20)
            self.assertEqual(called,[])
            for requests in ([('too-big',(64<<20)+1,lambda:called.append(1))], [self.request(str(i)) for i in range(9)]):
                with self.assertRaises(ValueError):pool.acquire_many(requests)
            self.assertEqual(called,[])
            self.assertEqual(pool.health()['node_pool_active_pins'],0)
            pool.close()


if __name__=='__main__':
    unittest.main()
