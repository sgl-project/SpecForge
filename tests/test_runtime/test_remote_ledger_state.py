"""Bounded ledger phases retain exact owner charges through statement/commit failures."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest

from specforge.runtime.data_plane.segment_manifest import RemoteObjectLedger


class LedgerStateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.a = RemoteObjectLedger(self.tmp.name, 8 << 20, 'a')
        self.addCleanup(self.a.db.close)

    def states(self):
        return dict(self.a.db.execute('SELECT key,state FROM objects'))

    def test_iterables_and_transaction_bounds(self):
        keys = [str(i) for i in range(130)]
        self.a.reserve(dict.fromkeys(keys, 1))
        sql = []
        self.a.db.set_trace_callback(sql.append)
        self.a.state(iter(()), 'WRITING')
        self.assertEqual(sql, [])
        self.a.state({keys[0]}, 'WRITING')
        self.assertEqual(len(sql), 1)
        sql.clear()
        self.a.state(iter(keys), 'READY')
        self.assertEqual(sql.count('BEGIN IMMEDIATE'), 3)
        self.assertEqual(sql.count('COMMIT'), 3)
        self.assertEqual(self.states(), dict.fromkeys(keys, 'READY'))

    def test_failed_second_chunk_recovery(self):
        keys = [str(i) for i in range(135)]
        self.a.reserve(dict.fromkeys(keys, 1))
        self.a.db.execute("""CREATE TRIGGER fail BEFORE UPDATE ON objects
          WHEN NEW.key='70' BEGIN SELECT RAISE(ABORT,'exact injected update failure'); END""")
        with self.assertRaisesRegex(sqlite3.IntegrityError, 'exact injected update failure'):
            self.a.state(keys, 'QUARANTINED')
        self.assertFalse(self.a.db.in_transaction)
        self.assertEqual(self.states(), {k:'QUARANTINED' if i<64 else 'RESERVED' for i,k in enumerate(keys)})
        self.a.db.execute('DROP TRIGGER fail')
        self.a.state(keys[64:], 'QUARANTINED')
        self.assertEqual(self.a.health()['remote_quarantined_bytes'], len(keys)*4096)

    def test_commit_failure_retains_charge_and_primary(self):
        self.a.reserve({'first':100,'second':200})
        con=self.a.db
        primary=sqlite3.OperationalError('unknown commit result')
        for committed in (False, True):
            class Failing:
                def __getattr__(self,name):
                    return getattr(con,name)
                def execute(self, sql, *params):
                    if sql == 'COMMIT':
                        if committed:
                            con.execute(sql)
                        raise primary
                    return con.execute(sql,*params)
            self.a.db=Failing()
            try:
                with self.assertRaises(sqlite3.OperationalError) as failure:
                    self.a.state(['first','second'],'DELETING')
                self.assertIs(failure.exception,primary)
            finally:
                self.a.db=con
            self.assertFalse(con.in_transaction)
            self.assertEqual(set(self.states().values()), {'DELETING' if committed else 'RESERVED'})
            self.assertEqual(self.a.health()['remote_owner_charged_bytes'],8192)

    def test_two_writers_keep_owner_fences(self):
        b=RemoteObjectLedger(self.tmp.name, 8<<20, 'b')
        self.addCleanup(b.db.close)
        owned_a=[f'a-{i}' for i in range(80)]
        owned_b=[f'b-{i}' for i in range(80)]
        self.a.reserve(dict.fromkeys(owned_a,1))
        b.reserve(dict.fromkeys(owned_b,1))
        barrier=threading.Barrier(2)
        def change(ledger,state):
            barrier.wait()
            ledger.state(dict.fromkeys(owned_a+owned_b),state)
        with ThreadPoolExecutor(2) as ex:
            first=ex.submit(change,self.a,'READY')
            second=ex.submit(change,b,'WRITING')
            first.result();second.result()
        self.assertEqual(self.states(),{**dict.fromkeys(owned_a,'READY'),**dict.fromkeys(owned_b,'WRITING')})

