"""Real spawn-process exclusion and exact persisted proof identity."""
import multiprocessing as mp
import os
import tempfile
import time
import unittest

from specforge.runtime.data_plane.terminal_receipt_proofs import TerminalReceiptProofs


def manifest():
    return {'token':'attempt', 'digest':'a' * 64, 'owner_epoch':'epoch',
            'ack_key':'exact/ack', 'receipt_key':'exact/receipt'}


def claim_worker(path, writer, ready, busy, finish, count, results):
    ledger = TerminalReceiptProofs(path, 1 << 20)
    claimant = 'a' if writer else 'b'
    if not writer:
        ready.wait(10)
    result = ledger.claim('store','sample',manifest(),claimant)
    if result == 'OWNER':
        with count.get_lock():
            count.value += 1  # The only client permitted to submit a native ACK.
        ready.set()
        finish.wait(10)
        ledger.prove('store','sample',manifest(),claimant)
    else:
        if result == 'BUSY':
            busy.set()
        deadline = time.monotonic() + 10
        while result != 'PROVED' and time.monotonic() < deadline:
            time.sleep(0.01)
            result = ledger.claim('store','sample',manifest(),claimant)
    results.put(result)
    ledger.close()


class TestTerminalReceiptProofs(unittest.TestCase):
    def test_spawned_claim_prevents_ack_after_another_process_proves_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            context = mp.get_context('spawn')
            ready, busy, finish = context.Event(), context.Event(), context.Event()
            count, results = context.Value('i', 0), context.Queue()
            workers = [context.Process(target=claim_worker, args=(directory+'/proof.sqlite', writer, ready, busy, finish, count, results)) for writer in (True,False)]
            for worker in workers:
                worker.start()
            self.assertTrue(busy.wait(20))
            finish.set()
            self.assertEqual(sorted([results.get(timeout=20), results.get(timeout=20)]), ['OWNER','PROVED'])
            for worker in workers:
                worker.join(20)
                self.assertEqual(worker.exitcode, 0)
            self.assertEqual(count.value, 1)
            proof = TerminalReceiptProofs(directory+'/proof.sqlite', 1 << 20)
            self.assertEqual(proof.claim('store','sample',manifest(),'fresh'), 'PROVED')
            self.assertEqual(proof.health()['terminal_proof_active_claims'], 0)
            proof.close()

    def test_different_attempt_has_no_terminal_proof_and_wrong_digest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            proof = TerminalReceiptProofs(directory+'/proof.sqlite', 1 << 20)
            self.assertEqual(proof.claim('store','sample',manifest(),'a'), 'OWNER')
            proof.prove('store','sample',manifest(),'a')
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                proof.claim('store','sample',{**manifest(),'digest':'b' * 64},'fresh')
            self.assertEqual(proof.claim('store','sample',{**manifest(),'token':'new-attempt'},'fresh'), 'OWNER')
            proof.unsafe('store','sample',{**manifest(),'token':'new-attempt'},'fresh')
            with self.assertRaisesRegex(RuntimeError, 'completion is unknown'):
                proof.claim('store','sample',{**manifest(),'token':'new-attempt'},'other')
            proof.db.execute("UPDATE proofs SET phase='UNKNOWN' WHERE token='new-attempt'")
            with self.assertRaisesRegex(ValueError, 'unknown terminal'):
                proof.claim('store','sample',{**manifest(),'token':'new-attempt'},'other')
            proof.close()
