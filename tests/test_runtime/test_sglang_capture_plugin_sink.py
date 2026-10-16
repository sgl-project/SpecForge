# coding=utf-8
"""The capture plugin's sink: NVLink publication and the store path it leaves alone.

Mirrors the v0.5.18 patch sink's NVLink tests in ``test_spec_capture_sink.py``.
The arena's allocator and free protocol run on any host. Publishing into an
arena needs CUDA and uses a stand-in TransferEngine whose fabric allocation is
an ordinary device tensor.
"""

import ctypes
import os
import socket
import sys
import threading
import time
import types
import unittest
from unittest import mock

import torch

sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(__file__),
        os.pardir,
        os.pardir,
        "plugins",
        "sglang-spec-capture",
    ),
)

from specforge_sglang_capture import sink  # noqa: E402

ARENA_BYTES = sink.NVLINK_ARENA_BYTES_ENV


def _spec(sample_id, **features):
    return {
        "store_id": "test",
        "sample_id": sample_id,
        "features": features or {"aux": "hidden_states"},
    }


class _BufferStore:
    """Records host ``batch_put_from`` writes as raw bytes keyed by object key."""

    def __init__(self):
        self.values = {}

    def register_buffer(self, ptr, size):
        return 0

    def unregister_buffer(self, ptr):
        return 0

    def batch_put_from(self, keys, pointers, sizes, config):
        for key, ptr, size in zip(keys, pointers, sizes):
            self.values[key] = ctypes.string_at(ptr, size)
        return [0] * len(keys)


def _engine_module(backing):
    """A ``mooncake.engine`` whose TransferEngine allocates ``backing``."""
    engine = types.SimpleNamespace(
        initialize=lambda *args: 0,
        get_rpc_port=lambda: 15459,
        allocate_managed_buffer=lambda size: backing.data_ptr(),
    )
    module = types.ModuleType("mooncake.engine")
    module.TransferEngine = lambda: engine
    return module


def _receive_frees(arena, expected, timeout=5.0):
    """Drain the arena's control connections until ``expected`` frees land."""
    freed, deadline = 0, time.monotonic() + timeout
    while freed < expected and time.monotonic() < deadline:
        freed += arena.receive_frees(timeout=0.1)
    return freed


def _object_bytes(backing, feature, nbytes):
    offset = feature["address"] - backing.data_ptr()
    return backing[offset : offset + nbytes]


class SpecCaptureSinkTest(unittest.TestCase):
    def setUp(self):
        self.sink = sink.SpecCaptureSink()
        self.addCleanup(self.sink._executor.shutdown)

    def _nvlink(self, backing):
        """Patch the environment and Mooncake for NVLink into ``backing``."""
        env = {
            "MOONCAKE_PROTOCOL": "nvlink",
            "MOONCAKE_LOCAL_HOSTNAME": "127.0.0.1",
            ARENA_BYTES: str(backing.numel()),
        }
        patches = (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.dict(sys.modules, {"mooncake.engine": _engine_module(backing)}),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def _arena(self):
        arena = self.sink._nvlink
        self.addCleanup(arena._listener.close)
        return arena

    def test_nvlink_arena_size_is_required(self):
        with mock.patch.dict(os.environ, {"MOONCAKE_PROTOCOL": "nvlink"}, clear=True):
            with self.assertRaisesRegex(ValueError, ARENA_BYTES):
                sink.NvlinkArena(torch.device("cpu"))

    def test_nvlink_always_publishes_device_memory(self):
        with mock.patch.dict(os.environ, {"MOONCAKE_PROTOCOL": "nvlink"}, clear=True):
            self.assertTrue(sink.gpu_put_enabled())
            with mock.patch.dict(os.environ, {"SGLANG_SPEC_CAPTURE_GPU_PUT": "0"}):
                with self.assertRaisesRegex(ValueError, "conflicts"):
                    sink.gpu_put_enabled()

    def test_store_protocols_publish_through_the_store(self):
        aux = torch.arange(6, dtype=torch.float32).reshape(3, 2)
        for protocol in ("tcp", "rdma"):
            with self.subTest(protocol=protocol):
                store = self.sink._store = _BufferStore()
                env = {
                    "MOONCAKE_PROTOCOL": protocol,
                    "SGLANG_SPEC_CAPTURE_GPU_PUT": "0",
                }
                with mock.patch.dict(os.environ, env, clear=True):
                    (result,) = self.sink.put_samples([(_spec("a"), aux, None)])
                self.assertEqual(
                    store.values, {"test/a/g1/hidden_states": aux.numpy().tobytes()}
                )
                self.assertNotIn("nvlink", result)
                self.assertEqual(
                    result["features"],
                    {"hidden_states": {"shape": [1, 3, 2], "dtype": "float32"}},
                )
        self.assertIsNone(self.sink._nvlink)

    def test_nvlink_never_connects_to_the_store(self):
        env = {"MOONCAKE_PROTOCOL": "nvlink", ARENA_BYTES: "4096"}
        with (
            mock.patch.dict(os.environ, env, clear=True),
            mock.patch.object(
                self.sink, "_connect", side_effect=AssertionError("store used")
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "needs CUDA captures"):
                self.sink.put_samples([(_spec("a"), torch.ones(2, 2), None)])
        self.assertIsNone(self.sink._nvlink)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_nvlink_publishes_addresses_into_the_arena(self):
        backing = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
        self._nvlink(backing)
        spec = _spec("nv", aux="hidden_states", last_hidden="target")
        spec["passthrough"] = [
            {"name": "input_ids", "data": [3, 1, 2], "shape": [1, 3], "dtype": "int64"}
        ]
        aux = torch.randn(3, 8, device="cuda", dtype=torch.bfloat16)
        last = torch.randn(3, 4, device="cuda")
        (result,) = self.sink.put_samples([(spec, aux, last)])
        arena = self._arena()
        self.assertEqual(
            result["nvlink"], {"session": "127.0.0.1:15459", "control": arena.control}
        )
        expected = {
            "hidden_states": aux.unsqueeze(0),
            "target": last.unsqueeze(0),
            "input_ids": torch.tensor([[3, 1, 2]], device="cuda"),
        }
        for name, tensor in expected.items():
            nbytes = tensor.numel() * tensor.element_size()
            self.assertTrue(
                torch.equal(
                    _object_bytes(backing, result["features"][name], nbytes),
                    tensor.reshape(-1).view(torch.uint8),
                ),
                name,
            )
        host, port = result["nvlink"]["control"].rsplit(":", 1)
        with socket.create_connection((host, int(port)), timeout=5) as client:
            client.sendall(b'["test/nv/g1/target", "missing"]\n')
            self.assertEqual(_receive_frees(arena, 1), 1)
        self.assertEqual(arena.health()["objects"], 2)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_nvlink_copies_after_the_ready_event(self):
        backing = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
        self._nvlink(backing)
        producer = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            torch.cuda._sleep(50_000_000)  # queue the fill behind a spin
            source = torch.full((4096, 8), 7.0, device="cuda")
            ready = torch.cuda.Event()
            ready.record()
        future = self.sink.submit_samples(
            [(_spec("ev"), source, None)], ready_event=ready
        )
        (result,) = future.result(timeout=30)
        self._arena()
        published = _object_bytes(
            backing, result["features"]["hidden_states"], source.numel() * 4
        )
        self.assertTrue(torch.all(published.view(torch.float32) == 7).item())

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_nvlink_republishes_a_key_only_on_replace(self):
        backing = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
        self._nvlink(backing)
        first = torch.ones(2, 4, device="cuda")
        self.sink.put_samples([(_spec("r", last_hidden="target"), None, first)])
        arena = self._arena()
        retry = _spec("r", aux="hidden_states", last_hidden="target")
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            # aux is placed first; the clash on target releases it again.
            self.sink.put_samples([(retry, first * 2, first * 3)])
        self.assertEqual(set(arena._objects), {"test/r/g1/target"})
        self.assertEqual(arena.health()["free_bytes"], arena.capacity - 512)
        retry["replace"] = True
        (result,) = self.sink.put_samples([(retry, first * 2, first * 3)])
        self.assertEqual(arena.health()["objects"], 2)
        for name, value in (("hidden_states", 2), ("target", 3)):
            published = _object_bytes(backing, result["features"][name], 32)
            self.assertTrue(torch.all(published.view(torch.float32) == value).item())


class NvlinkArenaTest(unittest.TestCase):
    """Allocation and the free protocol, without CUDA or Mooncake."""

    def setUp(self):
        self.arena = object.__new__(sink.NvlinkArena)
        self.arena.capacity = 4096
        self.arena._spans = [(0, 4096)]
        self.arena._objects = {}
        self.arena._listener = socket.create_server(("127.0.0.1", 0))
        self.arena._listener.setblocking(False)
        self.arena._peers = {}
        self.addCleanup(self.arena._listener.close)
        self.control = self.arena._listener.getsockname()

    def test_freed_space_is_reused_and_coalesced(self):
        sizes = (("a", 100), ("b", 1000), ("c", 1))
        offsets = [self.arena._allocate(key, n) for key, n in sizes]
        self.assertEqual(offsets, [0, 512, 1536])
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            self.arena._allocate("a", 1)
        self.assertEqual(self.arena.free(["a", "c", "missing"]), 2)
        self.assertEqual(self.arena._allocate("d", 512), 0)
        self.assertEqual(self.arena.free(["b", "d"]), 2)
        self.assertEqual(self.arena._spans, [(0, 4096)])

    def test_a_full_arena_waits_for_frees_then_fails(self):
        self.arena._allocate("all", 4096)

        def free_later():
            with socket.create_connection(self.control, timeout=5) as client:
                time.sleep(0.1)
                client.sendall(b'["all"]\n')

        threading.Thread(target=free_later, daemon=True).start()
        self.assertEqual(self.arena._allocate("next", 4096), 0)
        self.arena._WAIT_S = 0.05
        with self.assertRaisesRegex(MemoryError, "no room"):
            self.arena._allocate("more", 1)

    def test_frees_arrive_as_json_lines_across_reads(self):
        for key in ("k1", "k2", "k3"):
            self.arena._allocate(key, 10)
        with socket.create_connection(self.control, timeout=5) as client:
            client.sendall(b'["k1", "unknown"]\n["k')
            self.assertEqual(_receive_frees(self.arena, 1), 1)
            client.sendall(b'2"]\n')
            self.assertEqual(_receive_frees(self.arena, 1), 1)
        self.assertEqual(set(self.arena._objects), {"k3"})
        self.arena.receive_frees(timeout=0.1)  # the closed connection is dropped
        self.assertEqual(self.arena._peers, {})


if __name__ == "__main__":
    unittest.main()
