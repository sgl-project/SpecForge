"""Public store recovery after a contended release BEGIN and a delayed receipt."""
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from specforge.runtime.data_plane.node_pool_batch import process_gate
from tests.test_runtime.test_segment_manifest import Fixture, features


def _lock_directory(path):
    script = """
import sqlite3, sys
with sqlite3.connect(sys.argv[1], isolation_level=None) as db:
    db.execute('BEGIN IMMEDIATE')
    print('locked', flush=True)
    sys.stdin.readline()
    db.execute('COMMIT')
"""
    return subprocess.Popen([sys.executable, '-S', '-c', script, str(path)],
                            text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)


class BatchReleaseRecovery(unittest.TestCase):
    def test_claimed_receipt_and_real_busy_release(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            'SPECFORGE_TRAINER_NODE_POOL': '1',
            'SPECFORGE_TRAINER_NODE_POOL_DIR': tmp + '/pool',
            'SPECFORGE_TRAINER_NODE_POOL_BYTES': str(1 << 20),
            'SPECFORGE_TRAINER_NODE_POOL_BATCHED': '1',
        }):
            fixture = Fixture(tmp)
            a = fixture.publish('consumed-A', n=9)
            b = fixture.publish('failed-B', n=10)
            backend = fixture.backend
            reader = fixture.reader(receive_buffers='pinned')
            pool = reader._node_pool
            reader.adopt(a)
            reader.adopt(b)
            calls, effective = Counter(), Counter()
            def counted(fn):
                original = getattr(backend, fn)
                def invoke(key, *args, **kwargs):
                    before = key in backend._d
                    calls[fn, key] += 1
                    result = original(key, *args, **kwargs)
                    if before != (key in backend._d):
                        effective[fn, key] += 1
                    return result
                return invoke
            with patch.object(backend, 'put_from', counted('put_from')), \
                 patch.object(backend, 'remove', counted('remove')), \
                 patch.object(backend, 'close', create=True) as close_backend:
                _, handle = reader.get(a, names=['hidden_states'])
                reader.release(handle)
                reader.abort_ref(a)
                proofs = reader._terminal_proofs
                manifest = a.metadata['segment_manifest']
                row = lambda: proofs.db.execute('SELECT * FROM proofs WHERE sample_id=?',
                                                (a.sample_id,)).fetchone()
                identity = proofs.identity(reader.store_id, a.sample_id, manifest)
                claimed = (*identity, 'CLAIMED', reader._terminal_proof_owner, os.getpid())
                self.assertEqual(row(), claimed)
                ack, receipt = manifest['ack_key'], manifest['receipt_key']
                self.assertEqual(effective['put_from', ack], 1)
                destination = torch.empty(b.feature_specs['hidden_states'].shape)
                expected = features(10)['aux'].unsqueeze(0)
                db_path = Path(tmp, 'pool', 'directory.sqlite')
                self.assertEqual(Path(pool._db.execute('pragma database_list').fetchone()[2]), db_path)
                pool._db.execute('pragma busy_timeout=40')
                select = 'SELECT client,key,generation,token FROM pins ORDER BY token'
                before = []
                sql = []
                release_window = []
                original = pool._transaction
                @contextmanager
                def contend(scope):
                    if scope != 'batch_release':
                        with original(scope) as db:
                            yield db
                        return
                    before.extend(pool._db.execute(select))
                    window = process_gate().current()
                    release_window.append(window)
                    # All reads and destination copies completed before the release.
                    torch.testing.assert_close(destination, expected)
                    with _lock_directory(db_path) as writer:
                        try:
                            self.assertEqual(writer.stdout.readline().strip(), 'locked')
                            pool._db.set_trace_callback(sql.append)
                            try:
                                with original(scope):
                                    self.fail('busy BEGIN unexpectedly entered transaction body')
                            except sqlite3.OperationalError as error:
                                self.assertEqual(error.sqlite_errorcode, sqlite3.SQLITE_BUSY)
                                self.assertFalse(pool._db.in_transaction)
                                self.assertEqual(sql, ['BEGIN IMMEDIATE'])
                                self.assertIs(process_gate().current(), window)
                                self.assertTrue(window.closing)
                                self.assertEqual(set(before), {
                                    (pool.client, p.key, p.generation, p.token) for p in window.pins})
                                self.assertEqual(row(), claimed)
                                raise
                        finally:
                            pool._db.set_trace_callback(None)
                            writer.stdin.write('release\n')
                            writer.stdin.flush()
                            self.assertEqual(writer.wait(timeout=10), 0)
                with patch.object(pool, '_transaction', contend):
                    with self.assertRaises(sqlite3.OperationalError) as primary:
                        reader.get_into(b, {'hidden_states':destination})
                self.assertEqual(primary.exception.sqlite_errorcode, sqlite3.SQLITE_BUSY)
                self.assertEqual(list(pool._db.execute(select)), before)
                self.assertTrue(before)
                self.assertIs(process_gate().current(), release_window[0])
                print('BUSY', json.dumps(dict(directory=str(db_path), sql=sql, pins=before,
                                              claimant=claimed, failed=b.sample_id)), flush=True)
                with self.assertRaisesRegex(RuntimeError, 'unresolved terminal ACK claims'):
                    reader.close()
                self.assertIsNone(process_gate().current())
                self.assertEqual(list(pool._db.execute(select)), [])
                self.assertFalse(reader._closed)
                self.assertFalse(reader._capture_read_intents)
                self.assertEqual(row(), claimed)
                self.assertEqual(calls['put_from', ack], 1)
                close_backend.assert_not_called()
                with self.assertRaisesRegex(RuntimeError, 'closing'):
                    reader.get(b)
                fixture.owner.progress()
                self.assertIn(receipt, backend._d)
                self.assertEqual(reader.drain_pending_removals()['release_pending'], 0)
                self.assertEqual(row(), (*identity, 'PROVED', '', 0))
                self.assertEqual(proofs.owned_claims(reader._terminal_proof_owner), 0)
                self.assertEqual(calls['put_from', ack], 1)
                self.assertEqual(effective['put_from', ack], 1)
                self.assertEqual(effective['remove', ack], 1)
                # Fake backend lacks the force keyword; the adapter falls back after TypeError.
                self.assertEqual(calls['remove', receipt], 2)
                self.assertEqual(effective['remove', receipt], 1)
                self.assertEqual(calls['put_from', b.metadata['segment_manifest']['ack_key']], 0)
                self.assertEqual(pool.health()['node_pool_inflight_bytes'], 0)
                self.assertEqual(pool.health()['node_pool_clients'], 1)
                reader.close()
                reader.close()
                self.assertTrue(reader._closed)
                self.assertFalse(db_path.exists())
                close_backend.assert_not_called()
                print('RECOVERED',json.dumps(dict(sample=a.sample_id, proof='PROVED',
                      ack_calls=calls['put_from', ack], ack_publications=effective['put_from', ack],
                      ack_deletes=effective['remove',ack], receipt_delete_calls=calls['remove',receipt],
                      receipt_deletes=effective['remove',receipt], failed_sample_ack_calls=0)), flush=True)
                cleanup = fixture.reader(node_pool_enabled=False)
                cleanup.adopt(b)
                fixture.drain(cleanup, b)
                cleanup.close()
                fixture.close()


    def test_unknown_receive_keeps_its_window_while_global_drain_proves_other_claim(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            'SPECFORGE_TRAINER_NODE_POOL': '1',
            'SPECFORGE_TRAINER_NODE_POOL_DIR': tmp + '/pool',
            'SPECFORGE_TRAINER_NODE_POOL_BYTES': str(1 << 20),
            'SPECFORGE_TRAINER_NODE_POOL_BATCHED': '1',
        }):
            fixture = Fixture(tmp)
            a, b = fixture.publish('A'), fixture.publish('B', n=10)
            reader = fixture.reader(receive_buffers='pinned')
            reader._owns_store = True
            reader.adopt(a)
            reader.abort_ref(a)
            reader.adopt(b)
            with patch.object(fixture.backend, 'get_into', side_effect=RuntimeError('unknown DMA')):
                with self.assertRaisesRegex(RuntimeError, 'unknown DMA'):
                    reader.get_into(b, {'hidden_states':torch.empty(b.feature_specs['hidden_states'].shape)})
            window = process_gate().current()
            self.assertTrue(window.receive_only())
            intent, = reader._capture_read_intents.values()
            self.assertEqual(intent['fence'].wire['sample_id'], b.sample_id)
            self.assertTrue(intent['unsafe_fetch'])
            fixture.owner.progress()
            with self.assertRaisesRegex(RuntimeError, 'receive drain unknown'):
                reader.drain_pending_removals()
            self.assertNotIn(a.sample_id, reader._release_pending)
            self.assertEqual(reader._terminal_proofs.health()['terminal_proof_records'], 1)
            self.assertTrue(intent['unsafe_fetch'])
            self.assertIs(window, process_gate().current())
            self.assertGreater(reader._receive_pool.health()['receive_pool_quarantined_bytes'], 0)
            with patch.object(fixture.backend, 'close', return_value=0, create=True) as stop:
                reader.close()
                stop.assert_called_once_with()
            self.assertFalse(reader._capture_read_intents)
            self.assertIsNone(process_gate().current())
            cleanup = fixture.reader(node_pool_enabled=False)
            cleanup.adopt(b)
            fixture.drain(cleanup, b)
            cleanup.close()
            fixture.close()
