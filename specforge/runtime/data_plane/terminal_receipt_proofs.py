"""Durable exact-attempt receipt proofs shared by clients on a Trainer node.

A persistent claim spans ACK publication through receipt verification. A second
client cannot issue a late ACK after the first client has retired the receipt.
Completed rows are small control-plane audit evidence, not live data pages.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path


class TerminalReceiptProofs:
    def __init__(self, path, budget_bytes):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.budget = int(budget_bytes)
        if self.budget < (128 << 10):
            raise ValueError('terminal receipt proof budget must include SQLite journal headroom')
        self.lock = threading.RLock()
        self.closed = False
        self.db = sqlite3.connect(self.path, isolation_level=None, timeout=30,
                                  check_same_thread=False)
        self.db.execute('PRAGMA page_size=4096')
        if self.db.execute('PRAGMA page_size').fetchone()[0] != 4096:
            raise ValueError('terminal proofs require 4096-byte SQLite pages')
        self.db.execute('PRAGMA cache_size=-64')
        self.db.execute('PRAGMA journal_mode=DELETE')
        self.db.execute('PRAGMA synchronous=FULL')
        # DB plus rollback page records/header fit in this fixed allocation.
        self.db.execute(f'PRAGMA max_page_count={(self.budget - (64 << 10)) // 8192}')
        self.db.execute('CREATE TABLE IF NOT EXISTS config(version TEXT,budget INTEGER)')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            config = self.db.execute('SELECT version,budget FROM config').fetchone()
            if config is None:
                self.db.execute('INSERT INTO config VALUES (?,?)', ('terminal-proof-v1', self.budget))
            elif config != ('terminal-proof-v1', self.budget):
                raise ValueError('terminal proof layout/budget differs across clients')
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise
        self.db.execute('''CREATE TABLE IF NOT EXISTS proofs(
            store_id TEXT, sample_id TEXT, token TEXT, digest TEXT,
            owner_epoch TEXT, ack_key TEXT, receipt_key TEXT,
            phase TEXT, claimant TEXT, pid INTEGER,
            PRIMARY KEY(store_id,sample_id,token))''')

    @staticmethod
    def identity(store_id, sample_id, manifest):
        return (str(store_id), str(sample_id), manifest['token'], manifest['digest'],
                manifest['owner_epoch'], manifest['ack_key'], manifest['receipt_key'])

    def claim(self, store_id, sample_id, manifest, claimant):
        identity = self.identity(store_id, sample_id, manifest)
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                row = self.db.execute('SELECT * FROM proofs WHERE store_id=? AND sample_id=? AND token=?', identity[:3]).fetchone()
                if row is None:
                    self.db.execute('INSERT INTO proofs VALUES (?,?,?,?,?,?,?,?,?,?)',
                                    (*identity, 'CLAIMED', claimant, os.getpid()))
                    result = 'OWNER'
                else:
                    if row[:7] != identity:
                        raise ValueError('terminal receipt proof attempt/digest identity mismatch')
                    if row[7] not in ('CLAIMED', 'PROVED', 'UNSAFE'):
                        raise ValueError('unknown terminal receipt proof state')
                    if row[7] == 'PROVED':
                        result = 'PROVED'
                    elif row[7] == 'UNSAFE':
                        raise RuntimeError('terminal ACK completion is unknown; exact claim retained')
                    elif row[8] == claimant and row[9] == os.getpid():
                        result = 'OWNER'
                    else:
                        try:
                            os.kill(row[9], 0)
                        except ProcessLookupError:
                            raise RuntimeError('terminal ACK claimant exited before its receipt proof') from None
                        result = 'BUSY'
                self.db.execute('COMMIT')
                return result
            except BaseException:
                self.db.execute('ROLLBACK')
                raise

    def prove(self, store_id, sample_id, manifest, claimant):
        identity = self.identity(store_id, sample_id, manifest)
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                row = self.db.execute('SELECT * FROM proofs WHERE store_id=? AND sample_id=? AND token=?', identity[:3]).fetchone()
                if row is None or row[:7] != identity or row[7:] != ('CLAIMED', claimant, os.getpid()):
                    raise RuntimeError('receipt proof was not produced by the exact ACK claimant')
                # FULL commit completes before anyone may remove the receipt.
                self.db.execute("UPDATE proofs SET phase='PROVED',claimant='',pid=0 WHERE store_id=? AND sample_id=? AND token=?", identity[:3])
                self.db.execute('COMMIT')
            except BaseException:
                self.db.execute('ROLLBACK')
                raise

    def unsafe(self, store_id, sample_id, manifest, claimant):
        identity = self.identity(store_id, sample_id, manifest)
        with self.lock:
            self.db.execute("UPDATE proofs SET phase='UNSAFE' WHERE store_id=? AND sample_id=? AND token=? AND claimant=? AND phase='CLAIMED'", (*identity[:3], claimant))

    def owned_claims(self, claimant):
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM proofs WHERE claimant=? AND phase!='PROVED'", (claimant,)).fetchone()[0]

    def health(self):
        with self.lock:
            counts = dict(self.db.execute('SELECT phase,COUNT(*) FROM proofs GROUP BY phase'))
            file_bytes = 0
            for path in (self.path, Path(str(self.path) + '-journal')):
                try:
                    file_bytes += path.stat().st_size
                except FileNotFoundError:
                    pass  # Another client's completed transaction unlinked it.
            return {'terminal_proof_path': str(self.path), 'terminal_proof_budget_bytes': self.budget,
                    'terminal_proof_records': counts.get('PROVED', 0),
                    'terminal_proof_active_claims': counts.get('CLAIMED', 0),
                    'terminal_proof_unsafe_claims': counts.get('UNSAFE', 0),
                    'terminal_proof_file_bytes': file_bytes}

    def close(self):
        with self.lock:
            if not self.closed:
                self.db.close()
                self.closed = True
