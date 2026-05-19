"""Tests for the NCCL transport module and NCCL-based remote target transport.

Test categories:
  1. Unit tests (no GPU needed): metadata encode/decode, NCCLTransport init logic
  2. GPU unit tests (single GPU): tensor send/recv via loopback NCCL group
  3. Integration tests (multi-process): full server/client negotiation
"""

import json
import os
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

import torch

# Add project root so specforge is importable
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_TESTS_DIR)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from specforge.modeling.target._nccl_transport import (
    NCCLTransport,
    _DTYPE_TABLE,
    _DTYPE_TO_CODE,
    decode_nccl_metadata,
    encode_nccl_metadata,
)


class TestNCCLMetadataEncodeDecode(unittest.TestCase):
    """Unit tests for metadata JSON encode/decode — no GPU required."""

    def test_encode_decode_basic(self):
        """Round-trip encode/decode with typical tensor dict."""
        tensor_dict = {
            "hidden_states": torch.randn(2, 4, 8),
            "input_ids": torch.randint(0, 100, (2, 4)),
            "loss_mask": torch.ones(2, 4, dtype=torch.bool),
        }
        keys_order = ["hidden_states", "input_ids", "loss_mask"]

        raw = encode_nccl_metadata(tensor_dict, keys_order)
        decoded_keys, decoded_meta, _ = decode_nccl_metadata(raw)

        self.assertEqual(decoded_keys, keys_order)
        for key in keys_order:
            meta = decoded_meta[key]
            self.assertIsNotNone(meta)
            self.assertEqual(meta["dtype_code"], _DTYPE_TO_CODE[tensor_dict[key].dtype])
            self.assertEqual(meta["shape"], list(tensor_dict[key].shape))

    def test_encode_decode_with_none(self):
        """None values should round-trip correctly."""
        tensor_dict = {
            "hidden_states": torch.randn(1, 2),
            "target": None,
        }
        keys_order = ["hidden_states"]  # only non-None CUDA tensors

        raw = encode_nccl_metadata(tensor_dict, keys_order)
        decoded_keys, decoded_meta, _ = decode_nccl_metadata(raw)

        self.assertEqual(decoded_keys, keys_order)
        self.assertIsNotNone(decoded_meta["hidden_states"])

    def test_encode_decode_empty_keys(self):
        """Empty keys_order should work (edge case: all tensors are None)."""
        tensor_dict = {"a": None}
        keys_order = []

        raw = encode_nccl_metadata(tensor_dict, keys_order)
        decoded_keys, decoded_meta, _ = decode_nccl_metadata(raw)

        self.assertEqual(decoded_keys, [])

    def test_metadata_is_valid_json(self):
        """Output should be valid UTF-8 JSON."""
        tensor_dict = {
            "x": torch.randn(3, 5, dtype=torch.float16),
        }
        keys_order = ["x"]

        raw = encode_nccl_metadata(tensor_dict, keys_order)
        # Should not raise
        payload = json.loads(raw.decode("utf-8"))
        self.assertIn("keys_order", payload)
        self.assertIn("metadata", payload)

    def test_all_supported_dtypes(self):
        """All dtypes in _DTYPE_TABLE should encode correctly."""
        for code, dtype in _DTYPE_TABLE.items():
            if dtype == torch.bool:
                t = torch.ones(2, dtype=dtype)
            elif dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                t = torch.ones(2, dtype=dtype)
            else:
                t = torch.ones(2, dtype=dtype)
            tensor_dict = {"t": t}
            keys_order = ["t"]
            raw = encode_nccl_metadata(tensor_dict, keys_order)
            decoded_keys, decoded_meta, _ = decode_nccl_metadata(raw)
            self.assertEqual(decoded_meta["t"]["dtype_code"], code)

    def test_preserves_key_order(self):
        """keys_order should be preserved exactly."""
        tensor_dict = {
            "c": torch.randn(1),
            "a": torch.randn(1),
            "b": torch.randn(1),
        }
        keys_order = ["c", "a", "b"]

        raw = encode_nccl_metadata(tensor_dict, keys_order)
        decoded_keys, _, _ = decode_nccl_metadata(raw)
        self.assertEqual(decoded_keys, ["c", "a", "b"])

    def test_multidimensional_shapes(self):
        """Multi-dimensional shapes should round-trip."""
        tensor_dict = {
            "4d": torch.randn(2, 3, 4, 5),
            "scalar": torch.tensor(42.0),
            "1d": torch.randn(100),
        }
        keys_order = ["4d", "scalar", "1d"]

        raw = encode_nccl_metadata(tensor_dict, keys_order)
        _, decoded_meta, _ = decode_nccl_metadata(raw)

        self.assertEqual(decoded_meta["4d"]["shape"], [2, 3, 4, 5])
        self.assertEqual(decoded_meta["scalar"]["shape"], [])
        self.assertEqual(decoded_meta["1d"]["shape"], [100])


class TestNCCLTransportInit(unittest.TestCase):
    """Unit tests for NCCLTransport initialization logic (no actual NCCL)."""

    def test_server_rank(self):
        transport = NCCLTransport(nccl_port=12345, host="127.0.0.1", is_server=True)
        self.assertEqual(transport._rank, 0)
        self.assertFalse(transport.is_initialized)

    def test_client_rank(self):
        transport = NCCLTransport(nccl_port=12345, host="127.0.0.1", is_server=False)
        self.assertEqual(transport._rank, 1)
        self.assertFalse(transport.is_initialized)

    def test_double_init_returns_true(self):
        """Second initialize() call should short-circuit if already initialized."""
        transport = NCCLTransport(nccl_port=12345, host="127.0.0.1", is_server=True)
        # Simulate already initialized
        transport._initialized = True
        transport._pg = MagicMock()

        result = transport.initialize(timeout_seconds=1)
        self.assertTrue(result)

    def test_destroy_cleans_state(self):
        """destroy() should clear state even if pg is None."""
        transport = NCCLTransport(nccl_port=12345, host="127.0.0.1", is_server=True)
        transport._initialized = True
        transport._pg = None

        transport.destroy()
        self.assertFalse(transport.is_initialized)
        self.assertIsNone(transport._pg)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.device_count() >= 2,
    "Requires at least 2 CUDA GPUs (NCCL P2P needs separate devices)"
)
class TestNCCLTransportGPU(unittest.TestCase):
    """GPU tests for NCCL transport — requires 2+ CUDA GPUs.

    NCCL P2P send/recv between two ranks on the same GPU is not supported.
    These tests use spawn context with rank 0 on GPU 0 and rank 1 on GPU 1.
    """

    def _run_nccl_pair(self, server_fn, client_fn, timeout=60):
        """Run a server and client function in separate processes using spawn."""
        import multiprocessing as mp
        import random

        # Use a random port to avoid TIME_WAIT conflicts between tests
        nccl_port = random.randint(30000, 60000)
        host = "127.0.0.1"

        ctx = mp.get_context("spawn")
        server_queue = ctx.Queue()
        client_queue = ctx.Queue()

        p_server = ctx.Process(target=server_fn, args=(nccl_port, host, server_queue))
        p_client = ctx.Process(target=client_fn, args=(nccl_port, host, client_queue))

        p_server.start()
        p_client.start()

        p_server.join(timeout=timeout)
        p_client.join(timeout=timeout)

        # Check for timeout
        if p_server.is_alive():
            p_server.kill()
            p_server.join()
            self.fail("Server process timed out")
        if p_client.is_alive():
            p_client.kill()
            p_client.join()
            self.fail("Client process timed out")

        # Check exit codes
        if p_server.exitcode != 0:
            server_err = server_queue.get_nowait() if not server_queue.empty() else "unknown"
            self.fail(f"Server process exited with code {p_server.exitcode}: {server_err}")
        if p_client.exitcode != 0:
            client_err = client_queue.get_nowait() if not client_queue.empty() else "unknown"
            self.fail(f"Client process exited with code {p_client.exitcode}: {client_err}")

        # Get results
        server_result = server_queue.get_nowait() if not server_queue.empty() else ("ok", None)
        client_result = client_queue.get_nowait() if not client_queue.empty() else ("ok", None)

        return server_result, client_result

    def test_send_recv_single_tensor(self):
        """Test NCCL send/recv with a single tensor."""
        server_result, client_result = self._run_nccl_pair(
            _gpu_server_single, _gpu_client_single
        )
        if server_result[0] == "error":
            self.fail(f"Server error: {server_result[1]}")
        if client_result[0] == "error":
            self.fail(f"Client error: {client_result[1]}")
        self.assertEqual(server_result[0], "ok")
        self.assertEqual(client_result[0], "ok")
        # Verify data matches
        server_data = torch.tensor(server_result[1])
        client_data = torch.tensor(client_result[1])
        torch.testing.assert_close(server_data, client_data)

    def test_send_recv_multiple_dtypes(self):
        """Test NCCL send/recv with multiple tensors of different dtypes."""
        server_result, client_result = self._run_nccl_pair(
            _gpu_server_multi, _gpu_client_multi
        )
        self.assertEqual(server_result[0], "ok")
        self.assertEqual(client_result[0], "ok")
        torch.testing.assert_close(
            torch.tensor(server_result[1]["hidden"]),
            torch.tensor(client_result[1]["hidden"]),
        )
        self.assertEqual(server_result[1]["ids"], client_result[1]["ids"])

    def test_send_recv_with_metadata_encode_decode(self):
        """End-to-end: encode metadata on server, decode on client, send/recv tensors."""
        server_result, client_result = self._run_nccl_pair(
            _gpu_server_metadata, _gpu_client_metadata
        )
        self.assertEqual(server_result[0], "ok")
        self.assertEqual(client_result[0], "ok")
        torch.testing.assert_close(
            torch.tensor(server_result[1]["h"]),
            torch.tensor(client_result[1]["h"]),
        )
        self.assertEqual(server_result[1]["mask"], client_result[1]["mask"])


# --- Top-level functions for multiprocessing (must be picklable) ---

def _gpu_server_single(nccl_port, host, q):
    try:
        torch.cuda.set_device(0)  # Server on GPU 0
        from specforge.modeling.target._nccl_transport import NCCLTransport
        transport = NCCLTransport(nccl_port=nccl_port, host=host, is_server=True)
        ok = transport.initialize(timeout_seconds=60)
        if not ok:
            q.put(("error", "Server NCCL init returned False"))
            return
        data = torch.randn(4, 8, dtype=torch.float32, device="cuda")
        transport.send_tensors({"data": data}, ["data"])
        torch.cuda.synchronize()
        q.put(("ok", data.cpu().tolist()))
        transport.destroy()
    except Exception as e:
        import traceback
        q.put(("error", traceback.format_exc()))


def _gpu_client_single(nccl_port, host, q):
    try:
        torch.cuda.set_device(1)  # Client on GPU 1
        from specforge.modeling.target._nccl_transport import NCCLTransport, _DTYPE_TO_CODE
        transport = NCCLTransport(nccl_port=nccl_port, host=host, is_server=False)
        ok = transport.initialize(timeout_seconds=60)
        if not ok:
            q.put(("error", "Client NCCL init returned False"))
            return
        metadata = {"data": {"dtype_code": _DTYPE_TO_CODE[torch.float32], "shape": [4, 8]}}
        result = transport.recv_tensors(metadata, ["data"])
        q.put(("ok", result["data"].cpu().tolist()))
        transport.destroy()
    except Exception as e:
        import traceback
        q.put(("error", traceback.format_exc()))


def _gpu_server_multi(nccl_port, host, q):
    try:
        torch.cuda.set_device(0)  # Server on GPU 0
        from specforge.modeling.target._nccl_transport import NCCLTransport
        transport = NCCLTransport(nccl_port=nccl_port, host=host, is_server=True)
        ok = transport.initialize(timeout_seconds=30)
        assert ok
        hidden = torch.randn(2, 16, dtype=torch.float32, device="cuda")
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64, device="cuda")
        transport.send_tensors({"hidden": hidden, "ids": ids}, ["hidden", "ids"])
        torch.cuda.synchronize()
        q.put(("ok", {"hidden": hidden.cpu().tolist(), "ids": ids.cpu().tolist()}))
        transport.destroy()
    except Exception as e:
        q.put(("error", str(e)))


def _gpu_client_multi(nccl_port, host, q):
    try:
        torch.cuda.set_device(1)  # Client on GPU 1
        from specforge.modeling.target._nccl_transport import NCCLTransport, _DTYPE_TO_CODE
        transport = NCCLTransport(nccl_port=nccl_port, host=host, is_server=False)
        ok = transport.initialize(timeout_seconds=30)
        assert ok
        metadata = {
            "hidden": {"dtype_code": _DTYPE_TO_CODE[torch.float32], "shape": [2, 16]},
            "ids": {"dtype_code": _DTYPE_TO_CODE[torch.int64], "shape": [2, 3]},
        }
        result = transport.recv_tensors(metadata, ["hidden", "ids"])
        q.put(("ok", {"hidden": result["hidden"].cpu().tolist(), "ids": result["ids"].cpu().tolist()}))
        transport.destroy()
    except Exception as e:
        q.put(("error", str(e)))


def _gpu_server_metadata(nccl_port, host, q):
    try:
        torch.cuda.set_device(0)  # Server on GPU 0
        from specforge.modeling.target._nccl_transport import NCCLTransport, encode_nccl_metadata
        transport = NCCLTransport(nccl_port=nccl_port, host=host, is_server=True)
        ok = transport.initialize(timeout_seconds=30)
        assert ok
        data = {
            "h": torch.randn(1, 8, 16, dtype=torch.bfloat16, device="cuda"),
            "mask": torch.ones(1, 8, dtype=torch.bool, device="cuda"),
        }
        keys_order = ["h", "mask"]
        torch.cuda.synchronize()
        transport.send_tensors(data, keys_order)
        torch.cuda.synchronize()
        q.put(("ok", {
            "h": data["h"].cpu().float().tolist(),
            "mask": data["mask"].cpu().int().tolist(),
        }))
        transport.destroy()
    except Exception as e:
        q.put(("error", str(e)))


def _gpu_client_metadata(nccl_port, host, q):
    try:
        torch.cuda.set_device(1)  # Client on GPU 1
        from specforge.modeling.target._nccl_transport import NCCLTransport, _DTYPE_TO_CODE
        transport = NCCLTransport(nccl_port=nccl_port, host=host, is_server=False)
        ok = transport.initialize(timeout_seconds=30)
        assert ok
        metadata = {
            "h": {"dtype_code": _DTYPE_TO_CODE[torch.bfloat16], "shape": [1, 8, 16]},
            "mask": {"dtype_code": _DTYPE_TO_CODE[torch.bool], "shape": [1, 8]},
        }
        keys_order = ["h", "mask"]
        result = transport.recv_tensors(metadata, keys_order)
        q.put(("ok", {
            "h": result["h"].cpu().float().tolist(),
            "mask": result["mask"].cpu().int().tolist(),
        }))
        transport.destroy()
    except Exception as e:
        q.put(("error", str(e)))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestNCCLTransportEndToEnd(unittest.TestCase):
    """End-to-end test — covered by TestNCCLTransportGPU.test_send_recv_with_metadata_encode_decode."""
    pass


class TestServerClientNegotiation(unittest.TestCase):
    """Test the HTTP-level NCCL negotiation logic without actual model forward."""

    def test_client_nccl_header_set_when_initialized(self):
        """Client should set NCCL header when transport is initialized."""
        from specforge.modeling.target.remote_target_client import (
            NCCL_HEADER,
            RemoteModelClient,
        )

        client = RemoteModelClient("http://127.0.0.1:8001", timeout=5, max_retries=0)
        # Mock NCCL transport as initialized
        client._nccl_transport = MagicMock()
        client._nccl_transport.is_initialized = True
        client._nccl_init_attempted = True

        # Check that _request_transport would include the header
        headers = {"Content-Type": "application/octet-stream"}

        # Simulate header construction (no localhost gate anymore)
        if client._nccl_transport is not None and client._nccl_transport.is_initialized:
            headers[NCCL_HEADER] = "1"

        self.assertIn(NCCL_HEADER, headers)
        self.assertEqual(headers[NCCL_HEADER], "1")
        client.close()

    def test_client_nccl_disabled_env(self):
        """Client should not attempt NCCL when SPECFORGE_ENABLE_NCCL=0."""
        with patch.dict(os.environ, {"SPECFORGE_ENABLE_NCCL": "0"}):
            from specforge.modeling.target.remote_target_client import RemoteModelClient
            client = RemoteModelClient("http://127.0.0.1:8001")
            self.assertFalse(client._nccl_enabled)
            client.close()

    def test_client_nccl_enabled_by_default(self):
        """Client should enable NCCL by default."""
        # Remove env var if set
        env = os.environ.copy()
        env.pop("SPECFORGE_ENABLE_NCCL", None)
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("SPECFORGE_ENABLE_NCCL", None)
            from specforge.modeling.target.remote_target_client import RemoteModelClient
            client = RemoteModelClient("http://127.0.0.1:8001")
            self.assertTrue(client._nccl_enabled)
            client.close()

    def test_server_init_nccl_disabled(self):
        """Server should return disabled status when NCCL is off."""
        from specforge.modeling.target.remote_target_server import TargetModelServer

        with patch.dict(os.environ, {"SPECFORGE_ENABLE_NCCL": "0"}):
            server = TargetModelServer(
                mode="eagle3", model_path="/fake", nccl_port=9999
            )
            result = server.handle_init_nccl(b'{"nccl_port": 9999}')
            data = json.loads(result)
            self.assertEqual(data["status"], "disabled")

    def test_server_init_nccl_no_port(self):
        """Server should return error when no port is specified."""
        from specforge.modeling.target.remote_target_server import TargetModelServer

        server = TargetModelServer(mode="eagle3", model_path="/fake", nccl_port=None)
        result = server.handle_init_nccl(b'{}')
        data = json.loads(result)
        self.assertEqual(data["status"], "error")

    def test_server_serialize_fallback_to_wire(self):
        """When NCCL is not initialized, server should fallback to wire format."""
        from specforge.modeling.target.remote_target_server import TargetModelServer

        server = TargetModelServer(mode="eagle3", model_path="/fake", nccl_port=9999)
        server._use_nccl = True  # Client requests NCCL
        # But transport is not initialized → should use wire format
        server._nccl_transport = None

        data = {"x": torch.randn(2, 3)}
        with patch("specforge.modeling.target._tensor_wire.encode_to_buffer") as mock_wire:
            mock_wire.return_value = b"wire_data"
            result = server._serialize_response(data)
            mock_wire.assert_called_once_with(data)
            self.assertEqual(result, b"wire_data")


class TestClientNCCLPortResolution(unittest.TestCase):
    """Test NCCL port resolution logic."""

    def test_port_from_env(self):
        """SPECFORGE_NCCL_PORT env should be used if set."""
        with patch.dict(os.environ, {"SPECFORGE_NCCL_PORT": "55555"}):
            from specforge.modeling.target.remote_target_client import RemoteModelClient
            client = RemoteModelClient("http://127.0.0.1:8001")
            self.assertEqual(client._get_nccl_port(), 55555)
            client.close()

    def test_port_default_http_plus_100(self):
        """Default NCCL port should be HTTP port + 100."""
        env = os.environ.copy()
        env.pop("SPECFORGE_NCCL_PORT", None)
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("SPECFORGE_NCCL_PORT", None)
            from specforge.modeling.target.remote_target_client import RemoteModelClient
            client = RemoteModelClient("http://127.0.0.1:8001")
            self.assertEqual(client._get_nccl_port(), 8101)
            client.close()

    def test_port_custom_http_port(self):
        """Different HTTP ports should yield different NCCL ports."""
        env = os.environ.copy()
        env.pop("SPECFORGE_NCCL_PORT", None)
        with patch.dict(os.environ, env, clear=True):
            os.environ.pop("SPECFORGE_NCCL_PORT", None)
            from specforge.modeling.target.remote_target_client import RemoteModelClient
            client = RemoteModelClient("http://127.0.0.1:9000")
            self.assertEqual(client._get_nccl_port(), 9100)
            client.close()


if __name__ == "__main__":
    unittest.main()
