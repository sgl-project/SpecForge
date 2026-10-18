# coding=utf-8
"""The capture plugin's sink: NVLink publication and the store path it leaves alone.

Mirrors the v0.5.18 patch sink's NVLink tests in ``test_spec_capture_sink.py``.
The arena's allocator and free protocol run on any host. Publishing into an
arena needs CUDA and uses a stand-in TransferEngine whose fabric allocation is
an ordinary device tensor.
"""

import ctypes
import json
import os
import socket
import sys
import threading
import time
import types
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
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

from specforge_sglang_capture import _capture_internal_state  # noqa: E402
from specforge_sglang_capture import sink  # noqa: E402

from specforge.inference.adapters.server_capture import SGLangServerCaptureAdapter
from specforge.runtime.data_plane.mooncake_arena import (
    REF_METADATA_KEY,
    ArenaObjectClient,
)
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from tests.test_runtime.test_server_capture import (
    _capture_schema,
    _eagle3_contract,
    _task,
)

# Pin the launcher/client contract rather than deriving the name from the sink.
ARENA_BYTES = "SGLANG_SPEC_CAPTURE_ARENA_BYTES"


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
        self.addCleanup(self._close_control)
        self.addCleanup(self.sink._executor.shutdown)

    def _close_control(self):
        if self.sink._nvlink is not None:
            for peer in self.sink._nvlink._peers:
                peer.close()
        if self.sink._control_listener is not None:
            self.sink._control_listener.close()

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
                self.assertNotIn("arena", result)
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
        control = self.sink.control_endpoint()
        listener = self.sink._control_listener
        self.assertIsNone(self.sink._nvlink)
        (result,) = self.sink.put_samples([(spec, aux, last)])
        arena = self._arena()
        self.assertIs(arena._listener, listener)
        self.assertEqual(result["arena"]["control"], control)
        self.assertEqual(
            result["arena"], {"session": "127.0.0.1:15459", "control": arena.control}
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
        host, port = result["arena"]["control"].rsplit(":", 1)
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

    def _plugin_client(self):
        backing = torch.zeros(1 << 20, dtype=torch.uint8, device="cuda")
        self._nvlink(backing)
        self.sink.aux_layer_ids = [2, 5, 8]
        scheduler = SimpleNamespace(
            _specforge_capture_runtime=SimpleNamespace(is_writer=True, sink=self.sink)
        )
        client = ArenaObjectClient(local_hostname="127.0.0.1")

        def close_client():
            for connection in client._connections.values():
                connection.close()

        self.addCleanup(close_client)
        store = MooncakeFeatureStore(store=client, store_id="run0")
        self.addCleanup(store.close)

        def server_info(url, **kwargs):
            self.assertEqual(url, "http://plugin:30000/server_info")
            state = SimpleNamespace(internal_state={})
            _capture_internal_state(state, scheduler)
            response = mock.Mock()
            response.json.return_value = {"internal_states": [state.internal_state]}
            return response

        discovery = mock.patch("requests.get", side_effect=server_info)
        get_info = discovery.start()
        self.addCleanup(discovery.stop)
        expected = {}

        def post(url, json_body, timeout):
            self.assertEqual(url, "http://plugin:30000/generate")
            # Discovery must precede the first write and allocate no GPU arena.
            get_info.assert_called_once()
            self.assertIsNotNone(self.sink._control_listener)
            self.assertIsNone(self.sink._nvlink)
            rows = []
            for ids, params in zip(
                json_body["input_ids"], json_body["sampling_params"]
            ):
                spec = json.loads(params["custom_params"]["spec_capture"])
                aux = torch.arange(len(ids) * 24, device="cuda", dtype=torch.float32)
                aux = aux.view(len(ids), 24)
                last = torch.full((len(ids), 8), 7.0, device="cuda")
                (result,) = self.sink.put_samples([(spec, aux, last)])
                expected[spec["features"]["aux"]] = aux[None].cpu()
                expected[spec["features"]["last_hidden"]] = last[None].cpu()
                for item in spec["passthrough"]:
                    expected[item["name"]] = torch.tensor(
                        item["data"], dtype=torch.int64
                    ).reshape(item["shape"])
                rows.append({"meta_info": {"spec_capture": [result]}})
            return rows

        adapter = SGLangServerCaptureAdapter(
            "http://plugin:30000",
            store,
            run_id="run0",
            algorithm="eagle3",
            schema=_capture_schema("eagle3"),
            post_fn=post,
            server_capture="plugin",
        )

        def read(session, ptr, address, nbytes):
            # Emulate the transport into a host receive buffer, using the
            # actual CUDA arena addresses returned by the plugin.
            self.assertEqual(session, self.sink._nvlink.session)
            offset = address - backing.data_ptr()
            data = backing[offset : offset + nbytes].cpu()
            ctypes.memmove(ptr, data.data_ptr(), nbytes)
            return 0

        transfer = mock.patch.object(
            client,
            "_transfer_engine",
            return_value=SimpleNamespace(transfer_sync_read=read),
        )
        transfer.start()
        self.addCleanup(transfer.stop)
        return store, adapter, expected

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_plugin_results_are_read_and_freed_by_current_arena_client(self):
        store, adapter, expected = self._plugin_client()
        (ref,) = adapter.produce_refs([_task(0, 4)], capture=_eagle3_contract())
        locator = ref.metadata[REF_METADATA_KEY]
        self.assertEqual(locator["control"], self.sink.control_endpoint())
        self.assertEqual(set(locator["addresses"]), set(expected))
        tensors, handle = store.get(ref)
        for name, tensor in expected.items():
            torch.testing.assert_close(tensors[name], tensor)
        store.release(handle)
        self.assertEqual(
            _receive_frees(self.sink._nvlink, len(expected)), len(expected)
        )
        self.assertEqual(self.sink._nvlink.health()["free_bytes"], 1 << 20)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_first_lost_plugin_response_is_cleaned_without_a_ref(self):
        store, adapter, _ = self._plugin_client()
        post = adapter.post_fn

        def post_then_lose(*args, **kwargs):
            post(*args, **kwargs)
            raise TimeoutError("response lost after publication")

        adapter.post_fn = post_then_lose
        with self.assertRaisesRegex(TimeoutError, "response lost"):
            adapter.produce_refs([_task(0, 4)], capture=_eagle3_contract())
        objects = len(self.sink._nvlink._objects)
        self.assertGreater(objects, 0)
        self.assertEqual(store.discard_external_attempts(), 1)
        self.assertEqual(_receive_frees(self.sink._nvlink, objects), objects)
        self.assertEqual(self.sink._nvlink.health()["free_bytes"], 1 << 20)


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

    def _allocate(self, key, nbytes):
        return self.arena._allocate(
            key, nbytes, deadline=time.monotonic() + self.arena._WAIT_S
        )

    def test_freed_space_is_reused_and_coalesced(self):
        sizes = (("a", 100), ("b", 1000), ("c", 1))
        offsets = [self._allocate(key, n) for key, n in sizes]
        self.assertEqual(offsets, [0, 512, 1536])
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            self._allocate("a", 1)
        self.assertEqual(self.arena.free(["a", "c", "missing"]), 2)
        self.assertEqual(self._allocate("d", 512), 0)
        self.assertEqual(self.arena.free(["b", "d"]), 2)
        self.assertEqual(self.arena._spans, [(0, 4096)])

    def test_a_full_arena_waits_for_frees_then_fails(self):
        self._allocate("all", 4096)

        def free_later():
            with socket.create_connection(self.control, timeout=5) as client:
                time.sleep(0.1)
                client.sendall(b'["all"]\n')

        threading.Thread(target=free_later, daemon=True).start()
        self.assertEqual(self._allocate("next", 4096), 0)
        self.arena._WAIT_S = 0.05
        with self.assertRaisesRegex(MemoryError, "no room"):
            self._allocate("more", 1)

    def test_frees_arrive_as_json_lines_across_reads(self):
        for key in ("k1", "k2", "k3"):
            self._allocate(key, 10)
        with socket.create_connection(self.control, timeout=5) as client:
            client.sendall(b'["k1", "unknown"]\n["k')
            self.assertEqual(_receive_frees(self.arena, 1), 1)
            client.sendall(b'2"]\n')
            self.assertEqual(_receive_frees(self.arena, 1), 1)
        self.assertEqual(set(self.arena._objects), {"k3"})
        self.arena.receive_frees(timeout=0.1)  # the closed connection is dropped
        self.assertEqual(self.arena._peers, {})

    def test_publish_shares_one_deadline_and_rolls_back_on_timeout(self):
        self._allocate("old-a", 2048)
        self._allocate("old-b", 2048)
        self.arena._base = 0
        self.arena._bytes = torch.zeros(4096, dtype=torch.uint8)
        self.arena._stream = mock.Mock()
        now = [0.0]
        releases = [(20.0, "old-a"), (40.0, "old-b")]
        waits = []

        def receive_frees(timeout=0.0):
            if not timeout:
                return 0
            waits.append(timeout)
            release_at, key = releases[0]
            if release_at <= now[0] + timeout:
                now[0] = release_at
                releases.pop(0)
                return self.arena.free([key])
            now[0] += timeout
            return 0

        with (
            mock.patch.object(sink.time, "monotonic", side_effect=lambda: now[0]),
            mock.patch.object(self.arena, "receive_frees", side_effect=receive_frees),
            mock.patch.object(torch.cuda, "stream", return_value=nullcontext()),
        ):
            with self.assertRaisesRegex(MemoryError, "batch deadline"):
                self.arena.publish(
                    ["new-a", "new-b"],
                    [torch.ones(2048, dtype=torch.uint8) for _ in range(2)],
                    [],
                )
        self.assertEqual(now[0], 30.0)
        self.assertEqual(waits, [30.0, 10.0])
        self.assertEqual(self.arena._objects, {"old-b": (2048, 2048)})
        self.assertEqual(self.arena._spans, [(0, 2048)])
        self.arena._stream.synchronize.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
