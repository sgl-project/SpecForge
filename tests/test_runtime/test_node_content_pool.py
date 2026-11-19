"""Spawned processes share actual mapped bytes and one node capacity ledger."""
import multiprocessing as mp
import os
import tempfile
import threading
import time
import unittest

import torch

from specforge.runtime.data_plane.node_content_pool import NodeContentPool, PoolFetchFailed


def consume(directory, barrier, result, count):
    pool = NodeContentPool(directory, 8192)
    barrier.wait()
    def fetch():
        with count.get_lock():
            count.value += 1
        time.sleep(0.15)
        return torch.arange(128, dtype=torch.uint8)
    pin = pool.acquire('shared', 128, fetch)
    result.put((bytes(pin.tensor.tolist()), pool.health()))
    barrier.wait()
    pin.release()
    pool.close()


def exit_fetcher(directory, ready, finish):
    pool = NodeContentPool(directory, 4096)
    def fetch():
        ready.set()
        finish.wait(10)
        os._exit(7)
    pool.acquire('unfinished', 8, fetch)


class TestNodeContentPool(unittest.TestCase):
    def test_spawn_singleflight_and_one_real_storage(self):
        with tempfile.TemporaryDirectory() as directory:
            context = mp.get_context('spawn')
            barrier = context.Barrier(3)
            result = context.Queue()
            count = context.Value('i', 0)
            workers = [context.Process(target=consume, args=(directory, barrier, result, count)) for _ in range(2)]
            for worker in workers:
                worker.start()
            barrier.wait(timeout=20)
            rows = [result.get(timeout=20) for _ in workers]
            self.assertEqual(rows[0][0], rows[1][0])
            self.assertEqual(count.value, 1)
            self.assertTrue(all(row[1]['node_pool_storage_bytes'] == 4096 for row in rows))
            self.assertTrue(any(row[1].get('node_pool_singleflight_waits') == 1 for row in rows))
            self.assertTrue(any(row[1].get('node_pool_cross_process_hits') == 1 for row in rows))
            barrier.wait(timeout=20)
            for worker in workers:
                worker.join(timeout=20)
                self.assertEqual(worker.exitcode, 0)
            self.assertFalse(os.path.exists(os.path.join(directory, 'data.mmap')))

    def test_lookup_pin_blocks_eviction_and_duplicate_late_release(self):
        with tempfile.TemporaryDirectory() as directory:
            a = NodeContentPool(directory, 4096)
            b = NodeContentPool(directory, 4096)
            pin = a.acquire('a', 4, lambda: torch.arange(4, dtype=torch.uint8))
            self.assertIsNone(b.acquire('b', 4, lambda: torch.zeros(4, dtype=torch.uint8)))
            old_generation, old_token = pin.generation, pin.token
            pin.release()
            pin.release()
            replacement = b.acquire('b', 4, lambda: torch.ones(4, dtype=torch.uint8))
            a.release('a', old_generation, old_token)
            self.assertEqual(a.health()['node_pool_active_pins'], 1)
            self.assertEqual(int(replacement.tensor.sum()), 4)
            replacement.release()
            a.close()
            b.close()

    def test_fetch_failure_is_visible_and_no_shared_inflight_bytes_remain(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = NodeContentPool(directory, 4096)
            with self.assertRaisesRegex(RuntimeError, 'fetch broke'):
                pool.acquire('bad', 4, lambda: (_ for _ in ()).throw(RuntimeError('fetch broke')))
            self.assertEqual(pool.health()['node_pool_inflight_bytes'], 0)
            pool.close()

    def test_fetcher_exit_wakes_waiters_and_never_reuses_a_live_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            context = mp.get_context('spawn')
            ready, finish = context.Event(), context.Event()
            pool = NodeContentPool(directory, 4096)
            child = context.Process(target=exit_fetcher, args=(directory, ready, finish))
            child.start()
            self.assertTrue(ready.wait(20))
            self.assertIsNone(pool.acquire('other', 8, lambda: torch.ones(8, dtype=torch.uint8)))
            outcome = []
            def wait():
                try:
                    pool.acquire('unfinished', 8, lambda: self.fail('duplicate fetch'))
                except BaseException as exc:
                    outcome.append(exc)
            worker = threading.Thread(target=wait)
            worker.start()
            deadline = time.monotonic() + 5
            while pool.health()['node_pool_singleflight_waits'] == 0:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.01)
            finish.set()
            child.join(20)
            worker.join(20)
            self.assertEqual(child.exitcode, 7)
            self.assertIsInstance(outcome[0], PoolFetchFailed)
            self.assertEqual(pool.health()['node_pool_inflight_bytes'], 0)
            replacement = pool.acquire('after-exit', 8, lambda: torch.ones(8, dtype=torch.uint8))
            replacement.release()
            pool.close()
