# coding=utf-8
"""Contract tests for MooncakeFeatureStore using an in-memory fake backend.

The fake stands in for ``MooncakeDistributedStore`` (the subset of its API the
backend uses), so the FeatureStore contract — generation guard, clone-on-fetch,
consume-once free, retain mode, auth, hard-pin, max-hold gc, and the fallible-free
retry seam — is verified locally without a running Mooncake master. A real
end-to-end test against ``mooncake`` is gated below on the package import.
"""

import ctypes
import importlib.util
import unittest

import torch

from specforge.runtime.data_plane.disaggregated import AuthPolicy
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore


class _FakeMooncakeStore:
    """In-memory stand-in for MooncakeDistributedStore (API subset).

    Supports BOTH the zero-copy raw-buffer API (``put_from``/``get_into`` +
    ``register_buffer``, simulated with ctypes against the real buffer pointers)
    and the legacy byte API (``put``/``get``), so the contract is exercised on
    either transport without a running master.
    """

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

    # -- zero-copy raw-buffer API (ctypes-simulated) -----------------------
    def register_buffer(self, ptr, size):
        return 0

    def unregister_buffer(self, ptr):
        return 0

    def put_from(self, key, ptr, size, config=None):
        self.last_config = config
        self._d[key] = ctypes.string_at(ptr, size)  # DMA-equivalent read of src
        self.put_calls += 1
        return 0

    def get_into(self, key, ptr, size):
        data = self._d.get(key)
        if not data:
            return -1
        n = min(size, len(data))
        ctypes.memmove(ptr, data, n)  # DMA-equivalent write into dst
        return n

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


def _phys_resident(fake, sid="s0", store_id="run0"):
    """Transport-agnostic: do the sample's bytes physically exist in the fake?

    pickle: one key ``run0/s0``; zero-copy: ``run0/s0/g{gen}/{name}`` keys.
    """
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
        self.assertTrue(_phys_resident(fake))  # bytes still physically there
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
        fs = MooncakeFeatureStore(store=fake, store_id="run0", max_release_attempts=3)
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
        # Mirror of test_get_after_release_raises_even_if_remote_lingers, but for
        # abort(): Mooncake's remove() is lease-deferred (reports success while the
        # bytes linger under a read-lease). Within one process the abort tombstone
        # still makes the ref unresolvable immediately (B5), even though is_exist
        # is still 1.
        fake = _FakeMooncakeStore()
        fake.lease_defer = True
        fs = MooncakeFeatureStore(store=fake, store_id="run0")
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        fs.abort("s0")
        self.assertTrue(_phys_resident(fake))  # bytes physically linger
        with self.assertRaises(KeyError):
            fs.get(ref)  # logically aborted -> KeyError (no use-after-free)

    def test_reput_with_failed_remove_warns_and_supersedes(self):
        # Pickle path: re-put when the pre-remove of the prior hard-pinned blob
        # fails must still publish the new generation (overwriting the single key)
        # and warn loudly that a pinned blob may be orphaned. The stale ref is then
        # refused by the generation guard (the overwrite replaced the blob).
        # (Zero-copy uses distinct per-generation keys, so a failed remove leaves
        # the stale generation readable -- covered separately below.)
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
        out, _ = fs.get(ref2)  # the new generation is resident + served
        self.assertIn("hidden_state", out)
        with self.assertRaises(KeyError):
            fs.get(ref1)  # the superseded ref is refused (generation guard)


class TestMooncakeFeatureStoreZeroCopy(unittest.TestCase):
    """Zero-copy transport specifics (the default): per-tensor keys, no pickle."""

    def test_zero_copy_is_the_default(self):
        fs = _store()
        self.assertTrue(fs._zero_copy)

    def test_falls_back_to_pickle_without_raw_api(self):
        # a backend that lacks put_from/get_into must transparently use the blob
        # path (and still round-trip).
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
        # no pickle blob was written under the bare sample key
        self.assertNotIn("run0/s0", fake._d)

    def test_no_pickle_on_the_wire(self):
        # the stored bytes are the raw tensor buffer, not a torch.save pickle
        # (which would begin with the PK zip magic or the pickle protocol byte).
        fake = _FakeMooncakeStore()
        fs = MooncakeFeatureStore(store=fake, store_id="run0")
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        blob = fake._d["run0/s0/g1/hidden_state"]
        self.assertEqual(len(blob), 4 * 8 * 4)  # 4x8 float32, exactly the raw bytes
        self.assertFalse(blob[:2] == b"PK")  # not a torch.save zip archive

    def test_reput_success_supersedes_old_generation(self):
        fake = _FakeMooncakeStore()
        fs = MooncakeFeatureStore(store=fake, store_id="run0")
        ref1 = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        fs.put(_tensors(), sample_id="s0", metadata=_meta())  # gen 2; gen-1 removed
        self.assertNotIn("run0/s0/g1/hidden_state", fake._d)
        self.assertIn("run0/s0/g2/hidden_state", fake._d)
        with self.assertRaises(KeyError):
            fs.get(ref1)  # gen-1 keys gone -> stale ref refused (B5)

    def test_short_read_is_rejected(self):
        # A get_into that transfers fewer bytes than the spec-sized receive buffer
        # (a truncated / partially-written object the backend still reports
        # present) must raise -- never return a tensor whose uninitialized tail is
        # silent garbage. get_into returns the byte count, so a short count != nb
        # is the signal. Simulate by truncating the stored blob.
        fake = _FakeMooncakeStore()
        fs = MooncakeFeatureStore(store=fake, store_id="run0")
        ref = fs.put(_tensors(), sample_id="s0", metadata=_meta())
        key = "run0/s0/g1/hidden_state"
        self.assertEqual(len(fake._d[key]), 4 * 8 * 4)  # full 4x8 float32
        fake._d[key] = fake._d[key][:-4]  # drop 4 bytes -> short read on get_into
        with self.assertRaises(KeyError):
            fs.get(ref)


def _shared_pair(**consumer_kw):
    """A producer + consumer backed by ONE fake store = the real disagg topology.

    Two MooncakeFeatureStore instances (separate in-process generation/lease/freed
    indices) over a single shared backend, mirroring producer-on-node-0 /
    consumer-on-node-1. store_id must match so the keys line up.
    """
    fake = _FakeMooncakeStore()
    producer = MooncakeFeatureStore(store=fake, store_id="run0")
    consumer = MooncakeFeatureStore(store=fake, store_id="run0", **consumer_kw)
    return fake, producer, consumer


class TestMooncakeFeatureStoreCrossProcess(unittest.TestCase):
    """Cross-instance (disaggregated) contract: the consumer never put(), so it
    resolves refs purely from the shared backend + the generation carried on the
    ref, with its own empty in-process index."""

    def test_cross_process_put_then_get_bit_exact(self):
        fake, producer, consumer = _shared_pair(retain_on_release=True)
        src = _tensors()
        ref = producer.put(src, sample_id="s0", metadata=_meta())
        out, handle = consumer.get(ref)  # separate instance, empty local index
        for k in src:
            self.assertTrue(torch.equal(out[k], src[k]), f"{k} not bit-exact")
        self.assertEqual(handle.sample_id, "s0")
        # the producer owns the sample; the consumer resolved it cross-instance
        # from the shared backend + the generation carried on the ref.
        self.assertEqual(producer.health()["resident_samples"], 1)

    def test_cross_process_stale_generation_rejected(self):
        # producer re-puts (gen bumps on the shared blob); the consumer's original
        # ref is stale and must be refused via the on-disk generation guard, even
        # though the consumer's in-process index never saw either put.
        fake, producer, consumer = _shared_pair(retain_on_release=True)
        ref1 = producer.put(_tensors(), sample_id="s0", metadata=_meta())
        producer.put(_tensors(), sample_id="s0", metadata=_meta())  # re-put -> gen 2
        with self.assertRaises(KeyError):
            consumer.get(ref1)

    def test_cross_process_abort_blocks_consumer_get(self):
        # With a normal (immediate) remove, producer.abort physically deletes the
        # blob, so a separate consumer's get() raises (B5 holds cross-process via
        # physical removal, not the per-process tombstone).
        fake, producer, consumer = _shared_pair(retain_on_release=True)
        ref = producer.put(_tensors(), sample_id="s0", metadata=_meta())
        producer.abort("s0")
        self.assertFalse(_phys_resident(fake))
        with self.assertRaises(KeyError):
            consumer.get(ref)

    def test_cross_process_consume_once_free_by_consumer(self):
        # Consume-once consumer (retain_on_release=False) frees the shared blob on
        # release; the producer can then no longer resolve the ref.
        fake, producer, consumer = _shared_pair()  # consumer frees on release
        ref = producer.put(_tensors(), sample_id="s0", metadata=_meta())
        _, handle = consumer.get(ref)
        consumer.release(handle)
        self.assertFalse(_phys_resident(fake))
        with self.assertRaises(KeyError):
            producer.get(ref)

    @unittest.expectedFailure
    def test_cross_process_abort_under_lease_defer_is_known_gap(self):
        # KNOWN LIMITATION (deferred to the M7 shared metadata index): under a
        # lease-deferred remove, producer.abort marks only ITS OWN _freed and the
        # bytes linger, so a separate consumer (empty _freed) still resolves the
        # aborted ref -> stale. The cross-process tombstone needs a shared index.
        # Encoded as expectedFailure so it flips to a hard failure once M7 closes
        # it (prompting removal of this marker).
        fake = _FakeMooncakeStore()
        fake.lease_defer = True
        producer = MooncakeFeatureStore(store=fake, store_id="run0")
        consumer = MooncakeFeatureStore(
            store=fake, store_id="run0", retain_on_release=True
        )
        ref = producer.put(_tensors(), sample_id="s0", metadata=_meta())
        producer.abort("s0")
        with self.assertRaises(KeyError):
            consumer.get(ref)  # SHOULD raise; currently returns stale -> xfail


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
