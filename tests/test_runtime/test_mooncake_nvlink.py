"""MOONCAKE_PROTOCOL=nvlink: server-owned objects read over MNNVL (no GPU).

A stub capture server returns the sink's NVLink result rows (object addresses
plus the control endpoint) and a fake TransferEngine reads those addresses, so
the real client stack runs end to end: adapter -> ref metadata ->
``MooncakeFeatureStore`` -> ``NvlinkObjectClient`` reads and frees.
"""

import ctypes
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

import torch

from specforge.config import Config
from specforge.inference.adapters.server_capture import SGLangServerCaptureAdapter
from specforge.runtime.data_plane.mooncake_nvlink import (
    REF_METADATA_KEY,
    NvlinkObjectClient,
)
from specforge.runtime.data_plane.mooncake_store import (
    MOONCAKE_OBJECT_NOT_FOUND,
    MooncakeFeatureStore,
)
from specforge.training.disaggregated import _mooncake_store
from tests.test_runtime.test_server_capture import (
    _capture_schema,
    _eagle3_contract,
    _StubCaptureServer,
    _task,
)

SESSION = "capture-host:15459"


class _ServerArena:
    """The capture server's objects, keyed like the sink; frees via HTTP."""

    def __init__(self):
        self.objects = {}
        self.fail_frees = False
        self.freed = []
        arena = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if arena.fail_frees:
                    self.send_response(500)
                    self.end_headers()
                    return
                keys = body["keys"]
                arena.freed.extend(keys)
                freed = sum(arena.objects.pop(key, None) is not None for key in keys)
                data = json.dumps({"freed": freed}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self._http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._http.serve_forever, daemon=True).start()
        self.control = f"http://127.0.0.1:{self._http.server_address[1]}"

    def close(self):
        self._http.shutdown()
        self._http.server_close()

    def put_from(self, key, ptr, size, config=None):
        buffer = torch.empty(max(size, 1), dtype=torch.uint8)
        ctypes.memmove(buffer.data_ptr(), ptr, size)
        self.objects[key] = buffer
        return 0


class _NvlinkCaptureServer(_StubCaptureServer):
    """Adds what the sink returns under MOONCAKE_PROTOCOL=nvlink."""

    def __call__(self, url, json_body, timeout):
        rows = super().__call__(url, json_body, timeout)
        for row in rows:
            result = row["meta_info"]["spec_capture"]
            prefix = f"{result['store_id']}/{result['sample_id']}/g{result['gen']}"
            result["nvlink"] = {"session": SESSION, "control": self.backend.control}
            for name, meta in result["features"].items():
                meta["address"] = self.backend.objects[f"{prefix}/{name}"].data_ptr()
        return rows


class _FakeTransferEngine:
    def __init__(self):
        self.sessions = []

    def transfer_sync_read(self, session, ptr, address, nbytes):
        self.sessions.append(session)
        ctypes.memmove(ptr, address, nbytes)
        return 0


class NvlinkObjectClientTest(unittest.TestCase):
    def setUp(self):
        self.arena = _ServerArena()
        self.addCleanup(self.arena.close)
        self.server = _NvlinkCaptureServer(self.arena)
        self.client = NvlinkObjectClient(local_hostname="127.0.0.1")
        self.engine = _FakeTransferEngine()
        patcher = mock.patch.object(
            NvlinkObjectClient, "_transfer_engine", return_value=self.engine
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.store = MooncakeFeatureStore(store=self.client, store_id="run0")
        self.adapter = SGLangServerCaptureAdapter(
            "http://server:30000",
            self.store,
            run_id="run0",
            algorithm="eagle3",
            schema=_capture_schema("eagle3"),
            post_fn=self.server,
        )

    def _refs(self, *lengths):
        tasks = [_task(i, length) for i, length in enumerate(lengths)]
        return self.adapter.produce_refs(tasks, capture=_eagle3_contract())

    def test_reads_server_objects_and_frees_them_when_consumed(self):
        refs = self._refs(6, 9)
        for ref in refs:
            locator = ref.metadata[REF_METADATA_KEY]
            self.assertEqual(locator["session"], SESSION)
            self.assertEqual(set(locator["addresses"]), set(ref.feature_specs))
            out, handle = self.store.get(ref)
            for name, expected in self.server.expected[ref.sample_id].items():
                self.assertTrue(torch.equal(out[name], expected), name)
            self.store.release(handle)
        self.assertEqual(set(self.engine.sessions), {SESSION})
        self.assertEqual(self.arena.objects, {})

    def test_unlocated_keys_are_freed_on_every_known_server(self):
        self._refs(4)
        self.assertEqual(self.client.remove("run0/lost/g1/hidden_state"), 0)
        self.assertIn("run0/lost/g1/hidden_state", self.arena.freed)
        self.assertEqual(self.client.is_exist("run0/lost/g1/hidden_state"), 0)
        self.assertEqual(
            self.client.get_into("run0/lost/g1/hidden_state", 0, 8),
            MOONCAKE_OBJECT_NOT_FOUND,
        )

    def test_failed_free_stays_pending_until_a_retry_succeeds(self):
        (ref,) = self._refs(5)
        self.arena.fail_frees = True
        self.store.abort(ref.sample_id, reason="optimizer-boundary-durable-ack")
        self.assertEqual(self.store.health()["release_pending"], 1)
        self.assertTrue(self.arena.objects)
        self.arena.fail_frees = False
        report = self.store.retry_sample_removals([ref.sample_id])
        self.assertEqual(report["remaining_ids"], [])
        self.assertEqual(self.arena.objects, {})

    def test_server_writes_are_never_put_from_trainer_roles(self):
        with self.assertRaisesRegex(RuntimeError, "written by the capture server"):
            self.client.put_from("key", 0, 1)


class NvlinkStoreFactoryTest(unittest.TestCase):
    def _cfg(self, receive_buffers):
        return Config.model_validate(
            {
                "model": {"target_model_path": "target"},
                "data": {"prompts_path": "prompts.jsonl"},
                "deployment": {
                    "mode": "disaggregated",
                    "disaggregated": {
                        "control_dir": "/control",
                        "backend": "mooncake",
                        "server_urls": ["http://capture:30000"],
                        "receive_buffers": receive_buffers,
                    },
                },
            }
        )

    def test_nvlink_uses_device_receives_and_needs_no_store_endpoints(self):
        env = {"MOONCAKE_PROTOCOL": "nvlink", "MOONCAKE_LOCAL_HOSTNAME": "10.0.0.2"}
        with mock.patch.dict("os.environ", env, clear=True):
            store = _mooncake_store(self._cfg("cuda"))
            self.assertIsInstance(store._store, NvlinkObjectClient)
            self.assertEqual(store.receive_buffers, "cuda")
            with self.assertRaisesRegex(ValueError, "receive_buffers=cuda"):
                _mooncake_store(self._cfg("pinned"))


if __name__ == "__main__":
    unittest.main()
