# coding=utf-8
"""Contract tests for MooncakeFeatureStore using an in-memory fake backend.

The fake stands in for ``MooncakeDistributedStore`` (the subset of its API the
backend uses), so the FeatureStore contract — generation guard, clone-on-fetch,
consume-once free, retain mode, auth, hard-pin, max-hold gc, and the fallible-free
retry seam — is verified locally without a running Mooncake master. A real
end-to-end test against ``mooncake`` is gated below on the package import.
"""

import importlib.util
import unittest

import torch

from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.disaggregated import AuthPolicy
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore


class _FakeMooncakeStore:
    """In-memory stand-in for MooncakeDistributedStore (API subset)."""

    def __init__(self) -> None:
        self._d = {}
        self.last_config = None
        self.fail_remove = False
        self.lease_defer = False  # remove() returns ok but keeps bytes (Mooncake lease)
        self.put_calls = 0
        self.remove_calls = 0

    def is_exist(self, key):
        return 1 if key in self._d else 0

    def put(self, key, value, config=None):
        self.last_config = config
        self._d[key] = bytes(value)
        self.put_calls += 1
        return 0

    def get(self, key):
        return self._d.get(key, b"")

    def remove(self, key):
        self.remove_calls += 1
        if self.fail_remove:
            return -1
        if self.lease_defer:
            return 0  # report success but keep the object (lease-deferred free)
        self._d.pop(key, None)
        return 0

    def get_size(self, key):
        return len(self._d.get(key, b""))


class _FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def _tensors():
    torch.manual_seed(0)
    return {
        "hidden_state": torch.randn(4, 8),
        "target": torch.randn(4, 8),
        "input_ids": torch.arange(4).unsqueeze(0),
    }


def _meta():
    return {"run_id": "run0", "num_tokens": 4}


def _store(**kw):
    return MooncakeFeatureStore(store=_FakeMooncakeStore(), store_id="run0", **kw)


class TestMooncakeFeatureStore(unittest.TestCase):
    def test_put_get_roundtrip_bit_exact(self):
        fs = _store()
        src = _tensors()
        ref = fs.put(src, sample_id="s0", metadata=_meta())
        self.assertTrue(ref.feature_store_uri.startswith("mooncake://"))
        out, handle = fs.get(ref)
        for k in src:
            self.assertTrue(torch.equal(out[k], src[k]), f"{k} not bit-exact")
        self.assertEqual(handle.sample_id, "s0")

    def test_clone_on_fetch_independent(self):
        fs = _store()
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        out, _ = fs.get(ref)
        out["hidden_state"] += 1.0  # mutate the returned copy
        again, _ = fs.get(ref)
        self.assertFalse(torch.equal(out["hidden_state"], again["hidden_state"]))

    def test_hard_pin_config_on_put(self):
        fake = _FakeMooncakeStore()
        fs = MooncakeFeatureStore(store=fake, store_id="run0", hard_pin=True)
        fs.put(_tensors(), sample_id="s0", metadata=_meta())
        self.assertTrue(getattr(fake.last_config, "with_hard_pin", False))
        self.assertTrue(fs.health()["hard_pin"])

    def test_get_after_release_raises(self):
        fs = _store()
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        _, handle = fs.get(ref)
        fs.release(handle)
        with self.assertRaises(KeyError):
            fs.get(ref)

    def test_get_after_release_raises_even_if_remote_lingers(self):
        # Mooncake's remove() is lease-deferred: it can report success while the
        # bytes linger under a read-lease. The ref must still not resolve (B5).
        fake = _FakeMooncakeStore()
        fake.lease_defer = True
        fs = MooncakeFeatureStore(store=fake, store_id="run0")
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        _, handle = fs.get(ref)
        fs.release(handle)
        self.assertEqual(fake.is_exist("run0/s0"), 1)  # bytes still physically there
        with self.assertRaises(KeyError):
            fs.get(ref)  # but the ref is logically freed -> KeyError

    def test_abort_frees(self):
        fs = _store()
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        fs.abort("s0")
        with self.assertRaises(KeyError):
            fs.get(ref)
        self.assertEqual(fs.health()["resident_samples"], 0)

    def test_stale_generation_rejected_after_reput(self):
        fs = _store()
        ref1 = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        fs.put(_tensors(), sample_id="s0", metadata=_meta())  # re-put -> new gen
        with self.assertRaises(KeyError):
            fs.get(ref1)  # stale ref refused (B5)

    def test_retain_on_release_keeps_data(self):
        fs = _store(retain_on_release=True)
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        _, handle = fs.get(ref)
        fs.release(handle)
        out, _ = fs.get(ref)  # still available for the next epoch
        self.assertIn("hidden_state", out)

    def test_consume_once_free_on_last_lease(self):
        fs = _store()
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        _, h1 = fs.get(ref)
        _, h2 = fs.get(ref)
        fs.release(h1)
        self.assertEqual(fs.health()["resident_samples"], 1)  # still leased
        fs.release(h2)
        with self.assertRaises(KeyError):
            fs.get(ref)  # freed on last lease

    def test_auth_required_disaggregated(self):
        auth = AuthPolicy(token="secret")
        with self.assertRaises(PermissionError):
            MooncakeFeatureStore(
                store=_FakeMooncakeStore(), auth=auth, credential="wrong"
            )
        fs = MooncakeFeatureStore(
            store=_FakeMooncakeStore(), store_id="run0", auth=auth, credential="secret"
        )
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        out, _ = fs.get(ref)
        self.assertIn("target", out)

    def test_max_resident_bytes_raises_when_behind(self):
        fs = _store(max_resident_bytes=16)  # far below one sample
        with self.assertRaises(MemoryError):
            fs.put(_tensors(), sample_id="s0", metadata=_meta())

    def test_gc_force_frees_past_max_hold(self):
        clock = _FakeClock()
        fs = _store(max_hold_age_s=10.0, clock=clock)
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        clock.advance(11.0)
        res = fs.gc()
        self.assertEqual(res["force_freed"], 1)
        with self.assertRaises(KeyError):
            fs.get(ref)

    def test_gc_spares_leased_even_if_old(self):
        clock = _FakeClock()
        fs = _store(max_hold_age_s=10.0, clock=clock)
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        _, _h = fs.get(ref)  # active lease
        clock.advance(11.0)
        res = fs.gc()
        self.assertEqual(res["force_freed"], 0)  # spared while leased

    def test_release_pending_retry_then_reconcile(self):
        fake = _FakeMooncakeStore()
        fs = MooncakeFeatureStore(
            store=fake, store_id="run0", max_release_attempts=3
        )
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        _, handle = fs.get(ref)
        fake.fail_remove = True
        fs.release(handle)  # remote free fails -> parked
        self.assertEqual(fs.health()["release_pending"], 1)
        fs.gc()  # retry, still failing
        self.assertEqual(fs.health()["release_pending"], 1)
        fake.fail_remove = False
        res = fs.gc()  # now the remove succeeds
        self.assertEqual(res["force_freed"], 1)
        self.assertEqual(fs.health()["release_pending"], 0)

    def test_equivalence_with_local_feature_store(self):
        src = _tensors()
        local = LocalFeatureStore("run0")
        mooncake = _store()
        lref = local.put({k: v.clone() for k, v in src.items()}, sample_id="s0", metadata=_meta())
        mref = mooncake.put({k: v.clone() for k, v in src.items()}, sample_id="s0", metadata=_meta())
        lout, _ = local.get(lref)
        mout, _ = mooncake.get(mref)
        self.assertEqual(set(lout), set(mout))
        for k in lout:
            self.assertTrue(torch.equal(lout[k], mout[k]), f"{k} differs local vs mooncake")


@unittest.skipUnless(
    importlib.util.find_spec("mooncake") is not None,
    "mooncake package not installed; real end-to-end store test skipped",
)
class TestMooncakeFeatureStoreReal(unittest.TestCase):
    """End-to-end against a real Mooncake master. Requires env:
    MOONCAKE_LOCAL_HOSTNAME, MOONCAKE_METADATA_SERVER, MOONCAKE_MASTER_SERVER_ADDR.
    Run on a Mooncake-enabled GPU host."""

    def _setup_kwargs(self):
        import os

        req = ("MOONCAKE_LOCAL_HOSTNAME", "MOONCAKE_METADATA_SERVER", "MOONCAKE_MASTER_SERVER_ADDR")
        if not all(os.environ.get(k) for k in req):
            self.skipTest(f"set {req} to run the real Mooncake e2e test")
        return {
            "local_hostname": os.environ["MOONCAKE_LOCAL_HOSTNAME"],
            "metadata_server": os.environ["MOONCAKE_METADATA_SERVER"],
            "master_server_addr": os.environ["MOONCAKE_MASTER_SERVER_ADDR"],
            "protocol": os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
        }

    def test_real_roundtrip(self):
        fs = MooncakeFeatureStore(setup_kwargs=self._setup_kwargs())
        src = _tensors()
        ref = fs.put(src, sample_id="s0", metadata=_meta())
        out, handle = fs.get(ref)
        for k in src:
            self.assertTrue(torch.equal(out[k], src[k]))
        fs.release(handle)
        with self.assertRaises(KeyError):
            fs.get(ref)


if __name__ == "__main__":
    unittest.main()
