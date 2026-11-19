"""CPU observations of real SQLite transactions, directory limits and mmap pins.

The large-capacity test writes directory rows and a single data page, never a
128-GiB data set. It validates directory capacity, not cache performance.
"""
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest import mock

import torch

from specforge.runtime.data_plane import node_content_pool as module
from specforge.runtime.data_plane.node_content_pool import NodeContentPool, node_pool_budgets
from specforge.runtime.data_plane.pool_diagnostics import Diagnostics


class TestPoolBudget(unittest.TestCase):
    def test_total_budget_scales_and_covers_one_mib_entries(self):
        previous = None
        for gib in (8, 32, 64, 128):
            with self.subTest(gib=gib):
                budget = node_pool_budgets(gib << 30)
                self.assertEqual(budget['metadata_budget_bytes'], (gib << 30) // 128)
                self.assertEqual(budget['terminal_proof_budget_bytes'], 14 << 20)
                self.assertEqual(budget['data_capacity_bytes'] + budget['metadata_budget_bytes'], gib << 30)
                self.assertEqual(sum(budget[name] for name in ('directory_budget_bytes', 'terminal_proof_budget_bytes', 'metadata_runtime_headroom_bytes')), budget['metadata_budget_bytes'])
                self.assertGreaterEqual(budget['max_entries'], budget['data_capacity_bytes'] // (1 << 20))
                # Arbitrary small tails can exhaust entries before data bytes.
                self.assertLess(budget['max_entries'], budget['data_capacity_bytes'] // 4096)
                self.assertLessEqual(budget['max_page_count'] * 8200 + (64 << 10), budget['directory_budget_bytes'])
                if previous:
                    self.assertGreater(budget['max_entries'], previous['max_entries'])
                    self.assertGreater(budget['max_page_count'], previous['max_page_count'])
                previous = budget

    def test_small_budgeted_pool_scales_proof_and_tiny_test_exception_is_explicit(self):
        for total in (64 << 20, 128 << 20, 1 << 30, 4 << 30):
            with self.subTest(total=total):
                budget = node_pool_budgets(total)
                self.assertGreater(budget['terminal_proof_budget_bytes'], 0)
                self.assertLessEqual(budget['terminal_proof_budget_bytes'], 14 << 20)
                self.assertGreaterEqual(budget['directory_budget_bytes'], 8 * 8200 + (64 << 10))
                self.assertEqual(budget['budget_mode'], 'total-including-metadata')
        with tempfile.TemporaryDirectory() as directory:
            pool = NodeContentPool(directory, 64 << 20)
            self.assertLessEqual(pool.health()['node_pool_metadata_file_bytes'], pool.directory_budget)
            self.assertEqual(pool.terminal_proof_budget, node_pool_budgets(64 << 20)['terminal_proof_budget_bytes'])
            pool.close()
        budget = node_pool_budgets(4096)
        self.assertEqual(budget['budget_mode'], 'tiny-cpu-test-unbudgeted-directory')
        self.assertEqual(budget['terminal_proof_budget_bytes'], 0)
        self.assertIsNone(budget['max_entries'])
        for invalid in (-1, 0, 4095):
            with self.assertRaises(ValueError):
                node_pool_budgets(invalid)

    def test_baseline_directory_version_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            db = sqlite3.connect(Path(directory) / 'directory.sqlite')
            db.execute('CREATE TABLE config(version TEXT, capacity INTEGER)')
            db.execute('INSERT INTO config VALUES (?,?)', ('node-content-v1', 4096))
            db.commit()
            db.close()
            # Constructor rejection deliberately leaves no completed pool to
            # close; clean its partially-created Python handles in this fixture.
            rejected = NodeContentPool.__new__(NodeContentPool)
            try:
                with self.assertRaisesRegex(ValueError, 'layout/capacity differs'):
                    rejected.__init__(directory, 4096)
            finally:
                rejected._db.close()
                rejected._gate.close()


class TestPoolDiagnostics(unittest.TestCase):
    def setUp(self):
        self.trace_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.trace_directory.cleanup)
        with mock.patch('specforge.runtime.data_plane.pool_diagnostics.atexit.register'):
            self.diag = Diagnostics(self.trace_directory.name)
        self.addCleanup(self.diag.file.close)
        patcher = mock.patch.object(module, 'get_diagnostics', return_value=self.diag)
        patcher.start()
        self.addCleanup(patcher.stop)

    def metrics(self):
        return self.diag.snapshot()['metrics']

    def test_exact_entry_high_water_uses_existing_reserve_count(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = NodeContentPool(directory, 8 << 30)
            pool.max_entries = 2
            for key in ('a', 'b', 'c'):
                pin = pool.acquire(key, 8, lambda: torch.arange(8, dtype=torch.uint8))
                pin.release()
            self.assertEqual(self.metrics()['node_pool.entry_count_high_water'], 2)
            self.assertEqual(self.metrics()['node_pool.evictions_entry_cap'], 1)
            pool.close()

    def test_actual_commits_match_sqlite_trace_including_health_and_close(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = NodeContentPool(directory, 8192)
            sql = []
            pool._db.set_trace_callback(sql.append)
            first = pool.acquire('a', 8, lambda: torch.arange(8, dtype=torch.uint8))
            first.release()
            hit = pool.acquire('a', 8, lambda: self.fail('hit fetched'))
            self.assertEqual(hit.tensor.tolist(), list(range(8)))
            hit.release()
            pool.health()
            pool.close()
            counts = self.metrics()
            self.assertEqual(counts['node_pool.begin_success'], sql.count('BEGIN IMMEDIATE'))
            self.assertEqual(counts['node_pool.commit_calls'], sql.count('COMMIT'))
            self.assertEqual(counts['node_pool.commit_success'], sql.count('COMMIT'))
            scopes = ('lookup', 'fill', 'release', 'health', 'close')
            self.assertEqual(sum(counts[f'node_pool.{scope}.commit_success'] for scope in scopes), sql.count('COMMIT'))
            self.assertEqual(counts['node_pool.health.commit_success'], 2)
            self.assertEqual(counts['node_pool.close.commit_success'], 1)
            self.assertGreater(counts['node_pool.begin_ns'], 0)
            self.assertGreater(counts['node_pool.transaction_body_ns'], 0)
            self.assertGreater(counts['node_pool.commit_ns'], 0)
            self.assertEqual(counts['node_pool.fetch_calls'], 1)
            self.assertEqual(counts['node_pool.shared_cpu_copy_calls'], 1)
            self.assertEqual(counts['node_pool.shared_cpu_copy_completed_bytes'], 8)
            self.assertGreater(counts['node_pool.fetch_ns'], 0)
            self.assertGreater(counts['node_pool.shared_cpu_copy_ns'], 0)
            # Instrumentation adds no BEGIN/COMMIT to the real operation counts.
            self.assertEqual(sql.count('COMMIT'), 8)

    def test_rollback_preserves_original_exception_and_is_not_a_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = NodeContentPool(directory, 4096)
            sentinel = RuntimeError('transaction body sentinel')
            with self.assertRaises(RuntimeError) as caught:
                with pool._transaction('fixture_error'):
                    raise sentinel
            self.assertIs(caught.exception, sentinel)
            self.assertEqual(self.metrics()['node_pool.rollback_calls'], 1)
            self.assertEqual(self.metrics().get('node_pool.commit_calls', 0), 0)
            with self.assertRaises(RuntimeError) as caught:
                pool.acquire('bad-fetch', 8, lambda: (_ for _ in ()).throw(sentinel))
            self.assertIs(caught.exception, sentinel)
            self.assertEqual(self.metrics()['node_pool.fetch_calls'], 1)
            self.assertGreater(self.metrics()['node_pool.fetch_ns'], 0)
            self.assertEqual(self.metrics().get('node_pool.shared_cpu_copy_calls', 0), 0)
            self.assertEqual(pool.health()['node_pool_inflight_bytes'], 0)
            pool.close()

    def test_periodic_metadata_sampling_observes_live_journal_without_new_transactions(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = NodeContentPool(directory, 8 << 30)
            for _ in range(63):
                pool.release('missing', 'generation', 'token')
            with pool._transaction('fixture_journal') as db:
                db.execute("UPDATE counters SET value=value+1 WHERE name='hits'")
            counts = self.metrics()
            self.assertEqual(counts['node_pool.metadata_periodic_samples'], 1)
            self.assertEqual(counts['node_pool.commit_success'], 64)
            self.assertGreater(counts['node_pool.journal_file_bytes_observed_max'], 0)
            self.assertLessEqual(counts['node_pool.metadata_file_bytes_observed_max'], pool.directory_budget)
            self.assertFalse((Path(directory) / 'directory.sqlite-journal').exists())
            pool.close()

    def test_real_large_directory_reaches_scaled_entry_cap_with_bounded_files(self):
        # Real directory rows exceed the baseline's fixed3520 limit by >60x.
        # Sparse mmap occupies no data pages until the one final 8-byte fetch.
        with tempfile.TemporaryDirectory() as directory:
            pool = NodeContentPool(directory, 128 << 30)
            self.assertGreater(pool.max_entries, 200000)
            with pool._transaction('fixture_directory') as db:
                db.executemany('INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?)',
                               ((f'namespace/load_epoch/teacher/content/{index:064x}', f'{index:032x}', index * 4096, 4096, 8, 'READY', 'fixture', os.getpid(), float(index), '')
                                for index in range(pool.max_entries)))
            pin = pool.acquire('new-content', 8, lambda: torch.ones(8, dtype=torch.uint8))
            health = pool.health()
            self.assertEqual(health['node_pool_evictions'], 1)
            self.assertEqual(health['node_pool_diagnostic_observed_maxima']['entry_count'], pool.max_entries)
            self.assertEqual(self.metrics()['node_pool.evictions_entry_cap'], 1)
            self.assertLessEqual(health['node_pool_metadata_file_bytes'], pool.directory_budget)
            self.assertLessEqual((Path(directory) / 'data.mmap').stat().st_blocks * 512, 4096)
            self.assertEqual(pool._db.execute('PRAGMA max_page_count').fetchone()[0], pool.max_page_count)
            pin.release()
            # Force all directory entries ineligible, without writing data.
            with pool._transaction('fixture_ineligible') as db:
                db.execute("UPDATE entries SET state='WRITING'")
                pool._sample_metadata(db, 'fixture_full_journal')
                self.assertLessEqual(pool._observed_metadata['metadata_file_bytes'], pool.directory_budget)
            self.assertIsNone(pool.acquire('cannot-evict', 8, lambda: self.fail('entry cap bypassed')))
            self.assertEqual(self.metrics()['node_pool.no_eligible_victim_entry_cap'], 1)
            self.assertEqual(self.metrics()['node_pool.fallbacks_no_reservation'], 1)
            with pool._transaction('fixture_end_writers') as db:
                db.execute("UPDATE entries SET state='READY'")
            pool.close()
            self.assertFalse((Path(directory) / 'data.mmap').exists())

    def test_sqlite_begin_wait_and_python_lock_wait_are_separate_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            a = NodeContentPool(directory, 4096)
            b = NodeContentPool(directory, 4096)
            entered = threading.Event()
            errors = []
            def release(pool):
                entered.set()
                try:
                    pool.release('absent', 'generation', 'token')
                except BaseException as exc:
                    errors.append(exc)
            with a._transaction('fixture_held_sqlite'):
                worker = threading.Thread(target=release, args=(b,))
                worker.start()
                self.assertTrue(entered.wait(5))
                time.sleep(0.05)
            worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertGreater(self.metrics()['node_pool.begin_ns'], 10_000_000)
            entered.clear()
            with a._lock:
                worker = threading.Thread(target=release, args=(a,))
                worker.start()
                self.assertTrue(entered.wait(5))
                time.sleep(0.05)
            worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertGreater(self.metrics()['node_pool.python_lock_wait_ns'], 10_000_000)
            a.close()
            b.close()

    def test_capacity_fragmentation_and_oversize_reasons_preserve_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = NodeContentPool(directory, 5 * 4096)
            pins = [pool.acquire(str(index), 8, lambda: torch.ones(8, dtype=torch.uint8)) for index in range(5)]
            self.assertIsNone(pool.acquire('full', 8, lambda: self.fail('pin evicted')))
            self.assertEqual(self.metrics()['node_pool.no_eligible_victim_capacity'], 1)
            for index in (1, 3):
                pins[index].release()
            with pool._transaction('fixture_holes') as db:
                db.execute("DELETE FROM entries WHERE key IN ('1','3')")
            self.assertIsNone(pool.acquire('fragmented', 2 * 4096, lambda: self.fail('pinned holes merged')))
            self.assertEqual(self.metrics()['node_pool.no_eligible_victim_fragmentation'], 1)
            self.assertEqual(self.metrics()['node_pool.reason_classification_calls'], 2)
            self.assertGreater(self.metrics()['node_pool.reason_classification_ns'], 0)
            self.assertIsNone(pool.acquire('oversize', 6 * 4096, lambda: self.fail('oversize fetched')))
            self.assertEqual(self.metrics()['node_pool.fallbacks_oversize'], 1)
            for pin in pins:
                pin.release()
            replacement = pool.acquire('after-unpin', 2 * 4096, lambda: torch.ones(2 * 4096, dtype=torch.uint8))
            self.assertIsNotNone(replacement)
            self.assertGreater(self.metrics()['node_pool.evictions_fragmentation'], 0)
            self.assertGreater(self.metrics()['node_pool.offset_rows_returned'], 0)
            self.assertGreater(self.metrics()['node_pool.offset_rows_visited'], 0)
            replacement.release()
            pool.close()

    def test_real_writing_poll_count_exceeds_first_wait_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            pool = NodeContentPool(directory, 4096)
            entered, finish = threading.Event(), threading.Event()
            pins, errors = [], []
            def fetch():
                entered.set()
                self.assertTrue(finish.wait(10))
                return torch.arange(8, dtype=torch.uint8)
            def acquire(callback):
                try:
                    pins.append(pool.acquire('shared', 8, callback))
                except BaseException as exc:
                    errors.append(exc)
            writer = threading.Thread(target=acquire, args=(fetch,))
            waiter = threading.Thread(target=acquire, args=(lambda: self.fail('duplicate fetch'),))
            writer.start()
            self.assertTrue(entered.wait(10))
            waiter.start()
            deadline = time.monotonic() + 10
            while self.metrics().get('node_pool.writing_poll_iterations', 0) < 3:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.005)
            finish.set()
            writer.join(10)
            waiter.join(10)
            self.assertFalse(writer.is_alive())
            self.assertFalse(waiter.is_alive())
            self.assertEqual(errors, [])
            health = pool.health()
            self.assertEqual(health['node_pool_singleflight_waits'], 1)
            self.assertGreaterEqual(self.metrics()['node_pool.writing_poll_sleeps'], 3)
            self.assertEqual(self.metrics()['node_pool.writing_poll_iterations'], self.metrics()['node_pool.writing_poll_sleeps'])
            for pin in pins:
                pin.release()
            pool.close()

    def test_diagnostics_disabled_adds_no_sampling_queries(self):
        with tempfile.TemporaryDirectory() as directory:
            disabled = Diagnostics()
            with mock.patch.object(module, 'get_diagnostics', return_value=disabled):
                pool = NodeContentPool(directory, 4096)
            sql = []
            pool._db.set_trace_callback(sql.append)
            pin = pool.acquire('a', 8, lambda: torch.zeros(8, dtype=torch.uint8))
            pin.release()
            pool.health()
            pool.close()
            self.assertEqual(disabled.snapshot()['metrics'], {})
            self.assertFalse(any('PRAGMA page_count' in statement for statement in sql))


if __name__ == '__main__':
    unittest.main()
