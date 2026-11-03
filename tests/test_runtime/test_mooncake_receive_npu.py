"""Ascend NPU receive path: device dispatch (mocked) and real-backend copies.

The dispatch tests run on any host with device stand-ins and localized
mocks; the copy/lifecycle tests require torch_npu and mirror
``test_mooncake_receive_cuda.py``.
"""

import os
import types
import unittest
from unittest import mock

import torch

from specforge.runtime.data_plane import mooncake_store
from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.mooncake_store import (
    MooncakeFeatureStore,
    ReceiveBufferPool,
)
from tests.test_runtime.test_mooncake_store import _FakeMooncakeStore, _meta


def _make_device(type_, index=None):
    return types.SimpleNamespace(type=type_, index=index)


def _npu_consumer(kind="pinned", device_type="npu", ascend=True):
    backend = _FakeMooncakeStore()
    store = MooncakeFeatureStore(store=backend, receive_buffers=kind)
    patches = (
        mock.patch("specforge.utils.get_device_type", return_value=device_type),
        mock.patch.object(
            mooncake_store, "_ascend_runtime_available", return_value=ascend
        ),
        mock.patch.object(torch, "npu", mock.Mock(), create=True),
        mock.patch.object(mooncake_store.torch, "device", side_effect=_make_device),
    )
    for patch in patches:
        patch.start()
    return store, patches


class NpuDispatchTest(unittest.TestCase):
    def test_pinned_consumer_device_follows_the_npu_local_rank(self):
        store, patches = _npu_consumer()
        self.addCleanup(store.close)
        for patch in patches:
            self.addCleanup(patch.stop)
        with mock.patch.dict(os.environ, {"LOCAL_RANK": "3"}):
            device = store.consumer_device()
        self.assertEqual((device.type, device.index), ("npu", 3))

    def test_pinned_consumer_device_stays_host_without_an_accelerator(self):
        for device_type, ascend in (("cpu", False), ("cpu", True), ("npu", False)):
            with self.subTest(device_type=device_type, ascend=ascend):
                store, patches = _npu_consumer(device_type=device_type, ascend=ascend)
                self.addCleanup(store.close)
                for patch in patches:
                    self.addCleanup(patch.stop)
                self.assertIsNone(store.consumer_device())

    def test_copy_stream_uses_the_npu_stream_class(self):
        pool = ReceiveBufferPool(_FakeMooncakeStore(), kind="pinned")
        self.addCleanup(pool.close)
        npu_module = mock.Mock()
        device = types.SimpleNamespace(type="npu", index=None)

        def make_device(type_, index=None):
            return types.SimpleNamespace(type=type_, index=index)

        with (
            mock.patch.object(torch, "npu", npu_module, create=True),
            mock.patch.object(mooncake_store.torch, "device", side_effect=make_device),
        ):
            npu_module.current_device.return_value = 1
            pool.copy_stream(device)
        called = npu_module.Stream.call_args.kwargs["device"]
        self.assertEqual((called.type, called.index), ("npu", 1))

    def test_pinned_slots_pin_host_memory_when_ascend_is_available(self):
        pool = ReceiveBufferPool(_FakeMooncakeStore(), kind="pinned")
        self.addCleanup(pool.close)
        calls = []
        real_empty = torch.empty

        def spy_empty(*args, **kwargs):
            calls.append(dict(kwargs))
            kwargs["pin_memory"] = False  # plain CPU torch cannot pin here
            return real_empty(*args, **kwargs)

        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(
                mooncake_store, "_ascend_runtime_available", return_value=True
            ),
            mock.patch.object(mooncake_store.torch, "empty", side_effect=spy_empty),
        ):
            pool._new_slot(64, torch.device("cpu"))
        self.assertEqual(calls[0]["pin_memory"], True)


@unittest.skipUnless(
    hasattr(torch, "npu") and torch.npu.is_available(), "requires torch_npu"
)
class MooncakeReceiveNpuTest(unittest.TestCase):
    def test_prefetched_batch_handoff_waits_for_collate_on_consumer_stream(self):
        consumer_stream = torch.npu.Stream()
        gate_stream = torch.npu.Stream()
        gate = torch.npu.Event()
        with torch.npu.stream(consumer_stream):
            torch.ones(32, device="npu").sum().item()
        with torch.npu.stream(gate_stream):
            left = torch.ones(4096, 4096, device="npu", dtype=torch.float16)
            right = torch.ones(4096, 4096, device="npu", dtype=torch.float16)
            for _ in range(128):
                left = left @ right
            gate.record()

        def delayed_collate(features):
            source = features[0]["hidden_state"]
            output = torch.zeros_like(source)
            torch.npu.current_stream().wait_event(gate)
            output.copy_(source)
            return {"hidden_state": output}

        store = MooncakeFeatureStore(
            store=_FakeMooncakeStore(), receive_buffers="pinned", retain_on_release=True
        )
        ref = store.put(
            {"hidden_state": torch.full((32,), 7.0)},
            sample_id="sample",
            metadata=_meta(),
        )
        loader = FeatureDataLoader(
            store,
            refs=[ref],
            device="npu:0",
            num_workers=1,
            collate_fn=delayed_collate,
        )
        iterator = iter(loader)
        try:
            with torch.npu.stream(consumer_stream):
                batch = next(iterator)
                result = batch.tensors["hidden_state"].sum()
            self.assertEqual(result.item(), 224)
        finally:
            iterator.close()
            loader.close()
            store.close()

    def test_pinned_outputs_survive_immediate_slot_reuse(self):
        backend = _FakeMooncakeStore()
        producer = MooncakeFeatureStore(store=backend, store_id="test")
        consumer = MooncakeFeatureStore(
            store=backend, store_id="test", receive_buffers="pinned"
        )
        self.addCleanup(consumer.close)
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
        stream = pool.copy_stream(torch.device("npu"))
        with torch.npu.stream(stream):
            if hasattr(torch.npu, "_sleep"):
                torch.npu._sleep(10_000_000)
        output, _ = consumer.get(first, device="npu")
        self.assertTrue(stream.query(), "slot released before copy completion")
        next_output, _ = consumer.get(second, device="npu")
        self.assertTrue(torch.all(output["hidden_state"] == 7).item())
        self.assertTrue(torch.all(next_output["hidden_state"] == 11).item())
        self.assertEqual(pool.health()["receive_pool_hits"], 1)

    def test_device_reads_keep_integer_features_on_host(self):
        features = {
            "hidden_states": torch.randn(1, 64, 32).to(torch.bfloat16),
            "input_ids": torch.arange(64).unsqueeze(0),
            "loss_mask": (torch.arange(64) % 3 > 0).long().unsqueeze(0),
        }
        backend = _FakeMooncakeStore()
        producer = MooncakeFeatureStore(store=backend, store_id="test")
        consumer = MooncakeFeatureStore(
            store=backend, store_id="test", receive_buffers="pinned"
        )
        ref = producer.put(features, sample_id="sample", metadata=_meta())
        try:
            got, _ = consumer.get(ref, device="npu")
            torch.npu.synchronize()
        finally:
            consumer.close()
        self.assertEqual(got["hidden_states"].device.type, "npu")
        self.assertEqual(got["input_ids"].device.type, "cpu")
        self.assertEqual(got["loss_mask"].device.type, "cpu")
        for name, expected in features.items():
            self.assertTrue(torch.equal(got[name].cpu(), expected), name)

    def test_consumer_device_reports_the_real_npu(self):
        store = MooncakeFeatureStore(
            store=_FakeMooncakeStore(), receive_buffers="pinned"
        )
        self.addCleanup(store.close)
        with mock.patch.dict(os.environ, {"LOCAL_RANK": "0"}):
            device = store.consumer_device()
        self.assertIsNotNone(device)
        self.assertEqual(device, torch.device("npu", 0))


if __name__ == "__main__":
    unittest.main()
