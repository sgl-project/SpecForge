"""Device receive copies must complete before their slots are reused."""

import types
import unittest
from unittest import mock

import torch

from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore, _meta


class _DeviceReads(_FakeMooncakeStore):
    def get_into(self, key, ptr, size):
        pointer = types.SimpleNamespace(
            __cuda_array_interface__={
                "shape": (size,),
                "typestr": "|u1",
                "data": (ptr, False),
                "version": 3,
            }
        )
        destination = torch.as_tensor(pointer, device="cuda")
        destination.copy_(torch.frombuffer(bytearray(self._d[key]), dtype=torch.uint8))
        torch.cuda.current_stream().synchronize()
        return size


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class MooncakeReceiveCudaTest(unittest.TestCase):
    def test_pinned_and_cuda_outputs_survive_immediate_slot_reuse(self):
        for kind in ("pinned", "cuda"):
            with self.subTest(kind=kind):
                backend = _DeviceReads() if kind == "cuda" else _FakeMooncakeStore()
                producer = MooncakeFeatureStore(store=backend, store_id="test")
                consumer = MooncakeFeatureStore(
                    store=backend, store_id="test", receive_buffers=kind
                )
                first = producer.put(
                    {"hidden_state": torch.full((8192, 128), 7.0)},
                    sample_id="first",
                    metadata=_meta(),
                )
                second = producer.put(
                    {"hidden_state": torch.full((8192, 128), 11.0)},
                    sample_id="second",
                    metadata=_meta(),
                )
                pool = consumer._receive_pool
                stream = pool.copy_stream(torch.device("cuda"))
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(10_000_000)
                output, _ = consumer.get(first, device="cuda")
                self.assertTrue(stream.query(), "slot released before copy completion")
                next_output, _ = consumer.get(second, device="cuda")
                self.assertTrue(torch.all(output["hidden_state"] == 7).item())
                self.assertTrue(torch.all(next_output["hidden_state"] == 11).item())
                self.assertEqual(pool.health()["receive_pool_hits"], 1)

    def test_cuda_staging_fallback_reports_registration_failure_once(self):
        backend = _DeviceReads()
        producer = MooncakeFeatureStore(store=backend, store_id="test")
        consumer = MooncakeFeatureStore(
            store=backend, store_id="test", receive_buffers="cuda", receive_pool_bytes=1
        )
        ref = producer.put(
            {"hidden_state": torch.ones(8, 16)}, sample_id="sample", metadata=_meta()
        )
        with mock.patch.object(
            backend, "register_buffer", return_value=-600
        ) as register:
            with self.assertLogs(
                "specforge.runtime.data_plane.mooncake_store", "WARNING"
            ):
                for _ in range(2):
                    tensors, _ = consumer.get(ref, device="cuda")
                    self.assertTrue(torch.all(tensors["hidden_state"] == 1).item())
        self.assertEqual(register.call_count, 1)
