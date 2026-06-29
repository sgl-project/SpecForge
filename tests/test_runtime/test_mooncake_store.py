# coding=utf-8
"""MooncakeFeatureStore contract tests with an in-memory fake backend."""

import ctypes
import importlib.util
import unittest

import torch

from specforge.runtime.data_plane.disaggregated import AuthPolicy
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore


class _FakeMooncakeStore:
    """In-memory stand-in for the Mooncake API subset used by the store."""

    def __init__(self) -> None:
        self._d = {}
        self.last_config = None
        self.fail_remove = False
        self.lease_defer = False
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

    def register_buffer(self, ptr, size):
        return 0

    def unregister_buffer(self, ptr):
        return 0

    def put_from(self, key, ptr, size, config=None):
        self.last_config = config
        self._d[key] = ctypes.string_at(ptr, size)
        self.put_calls += 1
        return 0

    def get_into(self, key, ptr, size):
        data = self._d.get(key)
        if not data:
            return -1
        n = min(size, len(data))
        ctypes.memmove(ptr, data, n)
        return n

    def remove(self, key):
        self.remove_calls += 1
        if self.fail_remove:
            return -1
        if self.lease_defer:
            return 0
        self._d.pop(key, None)
        return 0

    def get_size(self, key):
        return len(self._d.get(key, b""))


def _phys_resident(fake, sid="s0", store_id="run0"):
    exact = f"{store_id}/{sid}"
    prefix = exact + "/"
    return any(k == exact or k.startswith(prefix) for k in fake._d)


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
        out["hidden_state"] += 1.0
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
        fake = _FakeMooncakeStore()
        fake.lease_defer = True
        fs = MooncakeFeatureStore(store=fake, store_id="run0")
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        _, handle = fs.get(ref)
        fs.release(handle)
        self.assertTrue(_phys_resident(fake))
        with self.assertRaises(KeyError):
            fs.get(ref)

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
        fs.put(_tensors(), sample_id="s0", metadata=_meta())
        with self.assertRaises(KeyError):
            fs.get(ref1)

    def test_retain_on_release_keeps_data(self):
        fs = _store(retain_on_release=True)
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        _, handle = fs.get(ref)
        fs.release(handle)
        out, _ = fs.get(ref)
        self.assertIn("hidden_state", out)

    def test_consume_once_free_on_last_lease(self):
        fs = _store()
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        _, h1 = fs.get(ref)
        _, h2 = fs.get(ref)
        fs.release(h1)
        self.assertEqual(fs.health()["resident_samples"], 1)
        fs.release(h2)
        with self.assertRaises(KeyError):
            fs.get(ref)

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
        fs = _store(max_resident_bytes=16)
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
        _, _h = fs.get(ref)
        clock.advance(11.0)
        res = fs.gc()
        self.assertEqual(res["force_freed"], 0)

    def test_release_pending_retry_then_reconcile(self):
        fake = _FakeMooncakeStore()
        fs = MooncakeFeatureStore(store=fake, store_id="run0", max_release_attempts=3)
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        _, handle = fs.get(ref)
        fake.fail_remove = True
        fs.release(handle)
        self.assertEqual(fs.health()["release_pending"], 1)
        fs.gc()
        self.assertEqual(fs.health()["release_pending"], 1)
        fake.fail_remove = False
        res = fs.gc()
        self.assertEqual(res["force_freed"], 1)
        self.assertEqual(fs.health()["release_pending"], 0)

    def test_equivalence_with_local_feature_store(self):
        src = _tensors()
        local = LocalFeatureStore("run0")
        mooncake = _store()
        lref = local.put(
            {k: v.clone() for k, v in src.items()}, sample_id="s0", metadata=_meta()
        )
        mref = mooncake.put(
            {k: v.clone() for k, v in src.items()}, sample_id="s0", metadata=_meta()
        )
        lout, _ = local.get(lref)
        mout, _ = mooncake.get(mref)
        self.assertEqual(set(lout), set(mout))
        for k in lout:
            self.assertTrue(
                torch.equal(lout[k], mout[k]), f"{k} differs local vs mooncake"
            )

    def test_abort_raises_even_if_remote_lingers(self):
        fake = _FakeMooncakeStore()
        fake.lease_defer = True
        fs = MooncakeFeatureStore(store=fake, store_id="run0")
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        fs.abort("s0")
        self.assertTrue(_phys_resident(fake))
        with self.assertRaises(KeyError):
            fs.get(ref)

    def test_reput_with_failed_remove_warns_and_supersedes(self):
        fake = _FakeMooncakeStore()
        fs = MooncakeFeatureStore(store=fake, store_id="run0", zero_copy=False)
        ref1 = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        fake.fail_remove = True
        with self.assertLogs(
            "specforge.runtime.data_plane.mooncake_store", level="WARNING"
        ) as cm:
            ref2 = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        self.assertTrue(
            any("orphan" in line.lower() for line in cm.output),
            f"expected an orphan warning, got {cm.output}",
        )
        out, _ = fs.get(ref2)
        self.assertIn("hidden_state", out)
        with self.assertRaises(KeyError):
            fs.get(ref1)


class TestMooncakeFeatureStoreZeroCopy(unittest.TestCase):
    def test_zero_copy_is_the_default(self):
        fs = _store()
        self.assertTrue(fs._zero_copy)

    def test_falls_back_to_pickle_without_raw_api(self):
        class _ByteOnly(_FakeMooncakeStore):
            put_from = None
            get_into = None

        fs = MooncakeFeatureStore(store=_ByteOnly(), store_id="run0")
        self.assertFalse(fs._zero_copy)
        src = _tensors()
        ref = fs.put(src, sample_id="s0", metadata=_meta())
        out, _ = fs.get(ref)
        for k in src:
            self.assertTrue(torch.equal(out[k], src[k]))

    def test_one_object_per_tensor_with_generation_in_key(self):
        fake = _FakeMooncakeStore()
        fs = MooncakeFeatureStore(store=fake, store_id="run0")
        fs.put(_tensors(), sample_id="s0", metadata=_meta())
        self.assertEqual(
            set(fake._d),
            {"run0/s0/g1/hidden_state", "run0/s0/g1/target", "run0/s0/g1/input_ids"},
        )
        self.assertNotIn("run0/s0", fake._d)

    def test_no_pickle_on_the_wire(self):
        fake = _FakeMooncakeStore()
        fs = MooncakeFeatureStore(store=fake, store_id="run0")
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        blob = fake._d["run0/s0/g1/hidden_state"]
        self.assertEqual(len(blob), 4 * 8 * 4)
        self.assertFalse(blob[:2] == b"PK")

    def test_reput_success_supersedes_old_generation(self):
        fake = _FakeMooncakeStore()
        fs = MooncakeFeatureStore(store=fake, store_id="run0")
        ref1 = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        fs.put(_tensors(), sample_id="s0", metadata=_meta())
        self.assertNotIn("run0/s0/g1/hidden_state", fake._d)
        self.assertIn("run0/s0/g2/hidden_state", fake._d)
        with self.assertRaises(KeyError):
            fs.get(ref1)

    def test_short_read_is_rejected(self):
        fake = _FakeMooncakeStore()
        fs = MooncakeFeatureStore(store=fake, store_id="run0")
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        key = "run0/s0/g1/hidden_state"
        self.assertEqual(len(fake._d[key]), 4 * 8 * 4)
        fake._d[key] = fake._d[key][:-4]
        with self.assertRaises(KeyError):
            fs.get(ref)


def _shared_pair(**consumer_kw):
    fake = _FakeMooncakeStore()
    producer = MooncakeFeatureStore(store=fake, store_id="run0")
    consumer = MooncakeFeatureStore(store=fake, store_id="run0", **consumer_kw)
    return fake, producer, consumer


class TestMooncakeFeatureStoreCrossProcess(unittest.TestCase):
    def test_cross_process_put_then_get_bit_exact(self):
        fake, producer, consumer = _shared_pair(retain_on_release=True)
        src = _tensors()
        ref = producer.put(src, sample_id="s0", metadata=_meta())
        out, handle = consumer.get(ref)
        for k in src:
            self.assertTrue(torch.equal(out[k], src[k]), f"{k} not bit-exact")
        self.assertEqual(handle.sample_id, "s0")
        self.assertEqual(producer.health()["resident_samples"], 1)

    def test_cross_process_stale_generation_rejected(self):
        fake, producer, consumer = _shared_pair(retain_on_release=True)
        ref1 = producer.put(_tensors(), sample_id="s0", metadata=_meta())
        producer.put(_tensors(), sample_id="s0", metadata=_meta())
        with self.assertRaises(KeyError):
            consumer.get(ref1)

    def test_cross_process_abort_blocks_consumer_get(self):
        fake, producer, consumer = _shared_pair(retain_on_release=True)
        ref = producer.put(_tensors(), sample_id="s0", metadata=_meta())
        producer.abort("s0")
        self.assertFalse(_phys_resident(fake))
        with self.assertRaises(KeyError):
            consumer.get(ref)

    def test_cross_process_consume_once_free_by_consumer(self):
        fake, producer, consumer = _shared_pair()
        ref = producer.put(_tensors(), sample_id="s0", metadata=_meta())
        _, handle = consumer.get(ref)
        consumer.release(handle)
        self.assertFalse(_phys_resident(fake))
        with self.assertRaises(KeyError):
            producer.get(ref)

    @unittest.expectedFailure
    def test_cross_process_abort_under_lease_defer_is_known_gap(self):
        # Requires a shared tombstone index; see the module docstring.
        fake = _FakeMooncakeStore()
        fake.lease_defer = True
        producer = MooncakeFeatureStore(store=fake, store_id="run0")
        consumer = MooncakeFeatureStore(
            store=fake, store_id="run0", retain_on_release=True
        )
        ref = producer.put(_tensors(), sample_id="s0", metadata=_meta())
        producer.abort("s0")
        with self.assertRaises(KeyError):
            consumer.get(ref)


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

        req = (
            "MOONCAKE_LOCAL_HOSTNAME",
            "MOONCAKE_METADATA_SERVER",
            "MOONCAKE_MASTER_SERVER_ADDR",
        )
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
