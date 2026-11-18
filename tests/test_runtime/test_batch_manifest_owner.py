"""Exact manifest generations, fail-closed batch probes, and scalar compatibility."""
import copy
import os
import tempfile
import unittest
from unittest.mock import patch

from tests.test_runtime.test_segment_manifest import Fixture


class BatchManifestOwnerTests(unittest.TestCase):
    def test_probe_order_and_shared_dependencies(self):
        for native_batch in (True, False):
            with self.subTest(batch=native_batch), tempfile.TemporaryDirectory() as root, patch.dict(
                    os.environ, {'SPECFORGE_TRAINER_NODE_POOL_DIR':root+'/pool'}):
                f = Fixture(root)
                b = f.backend
                probes = []
                if native_batch:
                    def probe(keys):
                        probes.append(list(keys))
                        return [int(key in b._d) for key in keys]
                    b.batch_is_exist = probe
                # Seed Teacher cache, then retain two manifests sharing its prefix.
                seed = f.publish('seed',h_prefix=True)
                a = f.publish('a',l=5,h_prefix=True)
                self.assertEqual(f.owner.stats['ready_unique_keys'],0)
                later = f.publish('b',l=5,h_prefix=True)
                m = later.metadata['segment_manifest']
                shared = [o['key'] for desc in m['features'].values() for o in desc if o['kind']=='prefix']
                self.assertEqual(len(shared),4)
                self.assertEqual(probes,[shared] if native_batch else [])
                self.assertEqual(f.owner.stats['ready_scalar_calls'],0 if native_batch else 4)
                self.assertEqual(f.owner.stats['block_reused'],4)
                self.assertEqual(f.owner.stats['attach_visits'],f.owner.stats['manifest_dependencies'])
                r = f.reader(node_pool_enabled=False)
                for ref in (seed,a,later):r.adopt(ref)
                f.drain(r,a)
                self.assertTrue(all(key in b._d for key in shared))
                f.drain(r,later);f.drain(r,seed)
                self.assertTrue(all(key not in b._d for key in shared))
                self.assertEqual(f.owner.stats['attach_visits'],f.owner.stats['detach_visits'])
                r.close();f.close();f.owner.close()

    def test_failed_probe_precedes_reservation_and_never_retries_scalar(self):
        with tempfile.TemporaryDirectory() as root:
            f=Fixture(root)
            a=f.publish('a')
            before=copy.deepcopy(f.owner.objects)
            ledger=f.owner.ledger.health()
            manifests=set(f.owner.manifests)
            original=set(f.backend._d)
            primary=RuntimeError('batch transport error')
            for result in ([],[1,1,1,0],[1,1,1,-1],primary):
                with self.subTest(result=result):
                    def probe(keys):
                        self.assertEqual(len(keys),4)
                        if result is primary:raise primary
                        return result
                    with patch.object(f.backend,'batch_is_exist',probe,create=True):
                        with self.assertRaises(RuntimeError) as caught:f.publish('b',l=5)
                    if result is primary:self.assertIs(caught.exception,primary)
                    self.assertEqual(f.owner.stats['ready_scalar_calls'],0)
                    self.assertEqual(f.owner.objects,before)
                    self.assertEqual(f.owner.ledger.health(),ledger)
                    self.assertEqual(set(f.owner.manifests),manifests)
                    self.assertEqual(set(f.backend._d),original)
            f.owner.cancel(a.metadata['segment_manifest']['token'])
            f.owner.progress();f.close()

    def test_deferred_cancel_only_removes_old_physical_generation(self):
        with tempfile.TemporaryDirectory() as root:
            f=Fixture(root)
            a=f.publish('a')
            m=a.metadata['segment_manifest']
            old={s['key'] for desc in m['features'].values() for s in desc}
            f.backend.lease_defer=True
            f.owner.cancel(m['token']);f.owner.progress()
            later=f.publish('b',l=5)
            m2=later.metadata['segment_manifest']
            new={s['key'] for desc in m2['features'].values() for s in desc}
            self.assertFalse(old & new)
            f.backend.lease_defer=False;f.owner.progress()
            self.assertTrue(old.isdisjoint(f.backend._d))
            self.assertTrue(new <= set(f.backend._d))
            f.owner.cancel(m2['token']);f.owner.progress()
            self.assertEqual(f.owner.health()['remote_owner_charged_bytes'],0)
            f.close()

