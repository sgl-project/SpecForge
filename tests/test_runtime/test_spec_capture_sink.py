"""Capture snapshots must survive scheduler reuse and writer-thread handoff."""

from __future__ import annotations

import ctypes
import sys
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

PATCH = (
    Path(__file__).resolve().parents[2] / "patches/sglang/v0.5.18/spec-capture.patch"
)


def _added_source(path):
    section = next(
        part
        for part in PATCH.read_text().split("diff --git ")
        if part.startswith(f"a/{path} b/{path}\n")
    )
    return "\n".join(
        line[1:]
        for line in section.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    )


def _load_sink():
    module = types.ModuleType("sglang.srt.spec_capture_sink")
    exec(_added_source("python/sglang/srt/spec_capture_sink.py"), module.__dict__)
    return module


def _append_capture():
    source = _added_source(
        "python/sglang/srt/managers/scheduler_components/batch_result_processor.py"
    )
    start = source.index("    def _append_spec_capture_states(")
    end = source.index("    def _sink_spec_capture(", start)
    namespace = {}
    exec(
        "from __future__ import annotations\n" + textwrap.dedent(source[start:end]),
        namespace,
    )
    return namespace["_append_spec_capture_states"]


class _BufferStore:
    def __init__(self, *, device=False, register_status=0):
        self.device = device
        self.register_status = register_status
        self.registrations = []
        self.unregistered = []
        self.values = {}

    def register_buffer(self, ptr, size):
        self.registrations.append((ptr, size))
        return self.register_status

    def unregister_buffer(self, ptr):
        self.unregistered.append(ptr)
        return 0

    def batch_put_from(self, keys, pointers, sizes, config):
        for key, ptr, size in zip(keys, pointers, sizes):
            if self.device:
                pointer = types.SimpleNamespace(
                    __cuda_array_interface__={
                        "shape": (size,),
                        "typestr": "|u1",
                        "data": (ptr, False),
                        "version": 3,
                    }
                )
                self.values[key] = (
                    torch.as_tensor(pointer, device="cuda").cpu().numpy().tobytes()
                )
            else:
                self.values[key] = ctypes.string_at(ptr, size)
        return [0] * len(keys)


class SpecCaptureSinkTest(unittest.TestCase):
    def setUp(self):
        self.module = _load_sink()
        self.sink = self.module.SpecCaptureSink()
        self.addCleanup(self.sink._executor.shutdown)
        self.env = mock.patch.dict(
            "os.environ",
            {"SGLANG_SPEC_CAPTURE_GPU_PUT": "0", "MOONCAKE_PROTOCOL": "rdma"},
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def _request(self, sample_id):
        return types.SimpleNamespace(
            spec_capture={
                "store_id": "test",
                "sample_id": sample_id,
                "features": {"aux": "hidden_states"},
            },
            spec_capture_aux=[],
            spec_capture_last_hidden=[],
        )

    def _append(self, request, logits, start, length):
        processor = types.SimpleNamespace(
            output_streamer=types.SimpleNamespace(
                ps=types.SimpleNamespace(attn_tp_rank=0)
            )
        )
        with mock.patch.dict(
            sys.modules, {"sglang.srt.spec_capture_sink": self.module}
        ):
            return _append_capture()(
                processor,
                req=request,
                logits_output=logits,
                hidden_state_offset=start,
                extend_input_len=length,
            )

    def test_batch_views_register_their_shared_storage_once(self):
        backing = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        requests = [self._request("a"), self._request("b")]
        logits = types.SimpleNamespace(hidden_states=backing, last_hidden_states=None)
        self._append(requests[0], logits, 0, 2)
        self._append(requests[1], logits, 2, 4)
        store = self.sink._store = _BufferStore()
        self.sink.submit_samples(
            [(req.spec_capture, req.spec_capture_aux[0], None) for req in requests]
        ).result(timeout=5)
        self.assertEqual(
            store.registrations, [(backing.data_ptr(), backing.numel() * 4)]
        )
        self.assertEqual(store.unregistered, [backing.data_ptr()])
        self.assertEqual(
            store.values["test/a/g1/hidden_states"], backing[:2].numpy().tobytes()
        )
        self.assertEqual(
            store.values["test/b/g1/hidden_states"], backing[2:].numpy().tobytes()
        )

    def test_gpu_capture_rejects_premature_host_staging(self):
        with mock.patch.dict("os.environ", {"SGLANG_SPEC_CAPTURE_GPU_PUT": "1"}):
            with self.assertRaisesRegex(RuntimeError, "CPU staging"):
                self._append(
                    self._request("a"),
                    types.SimpleNamespace(hidden_states=torch.ones(2, 4)),
                    0,
                    2,
                )

    def test_external_gpu_put_rejects_tcp(self):
        with mock.patch.dict(
            "os.environ",
            {"SGLANG_SPEC_CAPTURE_GPU_PUT": "1", "MOONCAKE_PROTOCOL": "tcp"},
        ):
            with self.assertRaisesRegex(ValueError, "requires MOONCAKE_PROTOCOL=rdma"):
                self.module.gpu_put_enabled()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_gpu_snapshot_survives_overwrite_and_stream_handoff(self):
        store = self.sink._store = _BufferStore(device=True)
        request = self._request("gpu")
        stream = torch.cuda.Stream()
        with mock.patch.dict("os.environ", {"SGLANG_SPEC_CAPTURE_GPU_PUT": "1"}):
            with torch.cuda.stream(stream):
                torch.cuda._sleep(10_000_000)
                source = torch.full((4096, 8), 7.0, device="cuda")
                logits = types.SimpleNamespace(hidden_states=source)
                self._append(request, logits, 0, source.shape[0])
                source.zero_()
                future = self.sink.submit_samples(
                    [(request.spec_capture, request.spec_capture_aux[0], None)]
                )
            future.result(timeout=20)
        expected = torch.full((4096, 8), 7.0).numpy().tobytes()
        self.assertEqual(store.values["test/gpu/g1/hidden_states"], expected)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_snapshot_finishes_before_forward_can_overwrite_graph_storage(self):
        source = _added_source("python/sglang/srt/managers/utils.py")
        start = source.index("        if capture_stream is not None:")
        last = "            capture_stream.wait_stream(torch.cuda.current_stream())"
        end = source.index(last, start) + len(last)
        namespace = {"torch": torch}
        exec("def snapshot(self, capture_stream):\n" + source[start:end], namespace)
        forward = torch.cuda.Stream()
        copier = torch.cuda.Stream()
        with torch.cuda.stream(forward):
            backing = torch.full((4096, 8), 7.0, device="cuda")
        logits = types.SimpleNamespace(hidden_states=backing, last_hidden_states=None)
        result = types.SimpleNamespace(logits_output=logits)
        copier.wait_stream(forward)
        with torch.cuda.stream(copier):
            torch.cuda._sleep(10_000_000)
            namespace["snapshot"](result, forward)
        with torch.cuda.stream(forward):
            backing.zero_()
        forward.synchronize()
        self.assertTrue(torch.all(logits.spec_capture_aux_gpu == 7).item())

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_gpu_registration_failure_prevents_publication(self):
        store = self.sink._store = _BufferStore(device=True, register_status=-600)
        with mock.patch.dict("os.environ", {"SGLANG_SPEC_CAPTURE_GPU_PUT": "1"}):
            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                self.sink.put_samples(
                    [
                        (
                            self._request("gpu").spec_capture,
                            torch.ones(2, 4, device="cuda"),
                            None,
                        )
                    ]
                )
        self.assertEqual(store.values, {})
        self.assertEqual(store.unregistered, [])
