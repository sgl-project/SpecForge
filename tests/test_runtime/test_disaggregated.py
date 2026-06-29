# coding=utf-8
"""Disaggregated FeatureStore: cross-process boundary, B5 use-after-free, B9 auth."""

import tempfile
import unittest

import torch

from specforge.runtime.contracts import assert_no_tensors
from specforge.runtime.data_plane.disaggregated import AuthPolicy, SharedDirFeatureStore


class _FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class TestDisaggregatedStore(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def test_producer_and_consumer_share_only_the_directory(self):
        # Two *separate* store instances (= two processes) share only `root`.
        # The consumer resolves the sample from the ref + filesystem alone.
        producer = SharedDirFeatureStore(self.root, store_id="st")
        t = torch.randn(1, 4, 8)
        ref = producer.put(
            {"hidden_state": t}, sample_id="s0", metadata={"strategy": "eagle3"}
        )
        self.assertTrue(ref.feature_store_uri.startswith("disagg://"))
        assert_no_tensors(ref)  # control plane stays metadata-only across the boundary

        consumer = SharedDirFeatureStore(self.root, store_id="st")  # different instance
        out, handle = consumer.get(ref)
        self.assertTrue(torch.allclose(out["hidden_state"], t))
        consumer.release(handle)
        # after the consumer frees it, it is gone for everyone (shared backend)
        with self.assertRaises(KeyError):
            producer.get(ref)

    def test_consumer_release_of_stale_handle_does_not_free_fresh_reput(self):
        # finding [1]/[2]: a consumer-only instance releasing a stale handle must
        # NOT delete the freshly re-put generation's data.
        producer = SharedDirFeatureStore(self.root, store_id="st")
        consumer = SharedDirFeatureStore(self.root, store_id="st")
        ref1 = producer.put({"x": torch.zeros(1, 4)}, sample_id="s0", metadata={})
        _, h_old = consumer.get(ref1)  # consumer leases gen1
        ref2 = producer.put(
            {"x": torch.ones(1, 4)}, sample_id="s0", metadata={}
        )  # gen2
        consumer.release(h_old)  # stale gen1 handle; must not touch gen2
        out, _ = consumer.get(ref2)  # gen2 must still be intact
        self.assertEqual(out["x"].sum().item(), 4.0)  # ones(1,4) intact

    def test_use_after_free_get_raises(self):
        store = SharedDirFeatureStore(self.root)
        ref = store.put({"x": torch.randn(1, 4)}, sample_id="s0", metadata={})
        store.abort("s0", reason="terminal")
        with self.assertRaises(KeyError):  # B5: no stale data after free
            store.get(ref)

    def test_stale_generation_is_rejected(self):
        store = SharedDirFeatureStore(self.root)
        ref_old = store.put({"x": torch.zeros(1, 4)}, sample_id="s0", metadata={})
        store.put(
            {"x": torch.ones(1, 4)}, sample_id="s0", metadata={}
        )  # re-put -> gen+1
        with self.assertRaises(KeyError):  # old handle must not alias new data
            store.get(ref_old)

    def test_clone_on_fetch_is_independent(self):
        store = SharedDirFeatureStore(self.root)
        ref = store.put({"x": torch.zeros(1, 4)}, sample_id="s0", metadata={})
        out, h = store.get(ref)
        out["x"].add_(1.0)  # mutate the fetched copy
        store.release(h)
        ref2 = store.put({"x": torch.zeros(1, 4)}, sample_id="s1", metadata={})
        out2, _ = store.get(ref2)
        self.assertEqual(out2["x"].sum().item(), 0.0)  # store data untouched

    def test_release_refcounted_last_lease_frees(self):
        store = SharedDirFeatureStore(self.root)
        ref = store.put({"x": torch.randn(1, 4)}, sample_id="s0", metadata={})
        _, h1 = store.get(ref)
        _, h2 = store.get(ref)
        store.release(h1)
        self.assertEqual(store.health()["resident_samples"], 1)  # h2 holds it
        store.release(h2)
        with self.assertRaises(KeyError):
            store.get(ref)

    def test_gc_force_frees_past_max_hold(self):
        clock = _FakeClock()
        store = SharedDirFeatureStore(self.root, max_hold_age_s=10.0, clock=clock)
        store.put({"x": torch.randn(1, 4)}, sample_id="old", metadata={})
        clock.advance(50.0)
        report = store.gc()
        self.assertEqual(report["force_freed"], 1)
        self.assertEqual(store.health()["resident_samples"], 0)


class TestDisaggregatedAuth(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()

    def test_auth_required_blocks_wrong_credential(self):
        auth = AuthPolicy("s3cret")
        # attach-time gate: wrong/missing credential cannot open the store
        with self.assertRaises(PermissionError):
            SharedDirFeatureStore(self.root, auth=auth, credential="wrong")
        with self.assertRaises(PermissionError):
            SharedDirFeatureStore(self.root, auth=auth, credential=None)

    def test_auth_enforced_on_data_path(self):
        auth = AuthPolicy("s3cret")
        store = SharedDirFeatureStore(self.root, auth=auth, credential="s3cret")
        ref = store.put({"x": torch.randn(1, 4)}, sample_id="s0", metadata={})
        out, _ = store.get(ref)
        self.assertIn("x", out)
        self.assertTrue(store.health()["auth_required"])

    def test_no_auth_is_open(self):
        store = SharedDirFeatureStore(self.root)  # colocated default
        self.assertFalse(store.health()["auth_required"])
        store.put({"x": torch.randn(1, 4)}, sample_id="s0", metadata={})


if __name__ == "__main__":
    unittest.main(verbosity=2)
