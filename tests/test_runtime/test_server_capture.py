# coding=utf-8
"""Unit tests for the server-side spec-capture transport (no GPU, no server).

A stub ``post_fn`` stands in for the patched SGLang server: it writes raw
tensor bytes into the shared fake Mooncake backend at the sink's key layout
(``{store_id}/{sample_id}/g{gen}/{name}``) and returns the
``meta_info["spec_capture"]`` result rows — i.e. it emulates
``patches/sglang/v0.5.14/spec-capture.patch``'s ``SpecCaptureSink`` byte-for-
byte. The tests then drive the REAL client stack over it:
``SGLangServerCaptureAdapter.produce_refs`` -> contract verification from
FeatureSpecs -> ``MooncakeFeatureStore.adopt`` -> ``RolloutWorker`` commit ->
zero-copy ``MooncakeFeatureStore.get``. The real-server end-to-end lives in
``test_server_capture_gate.py`` (opt-in, needs GPU + mooncake master).
"""

import ctypes
import os
import tempfile
import unittest
from typing import Any, Dict, List

import torch

from specforge.inference.adapters.server_capture import (
    ServerCaptureFailure,
    SGLangServerCaptureAdapter,
    resolve_server_capture_schema,
)
from specforge.inference.capture import (
    FeatureContract,
    FeatureContractError,
    verify_feature_contract_specs,
)
from specforge.runtime.contracts import FeatureSpec, PromptTask, SampleRef
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

HIDDEN = 8
AUX_LAYERS = (2, 5, 8)


class _FakeMooncakeStore:
    """API-subset fake (mirrors test_mooncake_store's), shared server<->client."""

    def __init__(self) -> None:
        self._d = {}
        self.last_config = None

    def is_exist(self, key):
        return 1 if key in self._d else 0

    def register_buffer(self, ptr, size):
        return 0

    def unregister_buffer(self, ptr):
        return 0

    def put_from(self, key, ptr, size, config=None):
        self.last_config = config
        self._d[key] = ctypes.string_at(ptr, size)
        return 0

    def get_into(self, key, ptr, size):
        data = self._d.get(key)
        if not data:
            return -1
        n = min(size, len(data))
        ctypes.memmove(ptr, data, n)
        return n

    def remove(self, key):
        self._d.pop(key, None)
        return 0

    def get_size(self, key):
        return len(self._d.get(key, b""))


def _dtype_str(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


class _StubCaptureServer:
    """Emulates the patched server: sink writes + meta_info result rows."""

    def __init__(
        self,
        backend: _FakeMooncakeStore,
        *,
        hidden: int = HIDDEN,
        aux_width: int = len(AUX_LAYERS) * HIDDEN,
        aux_layer_ids=AUX_LAYERS,
        error_sample_ids=(),
    ) -> None:
        self.backend = backend
        self.hidden = hidden
        self.aux_width = aux_width
        self.aux_layer_ids = list(aux_layer_ids)
        self.error_sample_ids = set(error_sample_ids)
        self.expected: Dict[str, Dict[str, torch.Tensor]] = {}

    def _put(self, key: str, t: torch.Tensor) -> None:
        t = t.detach().cpu().contiguous()
        self.backend.put_from(key, t.data_ptr(), t.element_size() * t.numel())

    def __call__(self, url: str, json_body: Dict[str, Any], timeout: float):
        assert url.endswith("/generate")
        rows: List[Dict[str, Any]] = []
        for input_ids, spec in zip(json_body["input_ids"], json_body["spec_capture"]):
            sid, gen = spec["sample_id"], int(spec["gen"])
            if sid in self.error_sample_ids:
                rows.append(
                    {
                        "meta_info": {
                            "spec_capture": {
                                "sample_id": sid,
                                "error": "injected sink error",
                            }
                        }
                    }
                )
                continue
            length = len(input_ids)
            torch.manual_seed(length + gen)
            written: Dict[str, torch.Tensor] = {}
            feats: Dict[str, Dict[str, Any]] = {}

            def _write(name: str, t: torch.Tensor) -> None:
                self._put(f"{spec['store_id']}/{sid}/g{gen}/{name}", t)
                written[name] = t
                feats[name] = {"shape": list(t.shape), "dtype": _dtype_str(t)}

            mapping = spec["features"]
            if "aux" in mapping:
                _write(
                    mapping["aux"],
                    torch.randn(1, length, self.aux_width, dtype=torch.bfloat16),
                )
            if "last_hidden" in mapping:
                _write(
                    mapping["last_hidden"],
                    torch.randn(1, length, self.hidden, dtype=torch.bfloat16),
                )
            for item in spec.get("passthrough", []):
                _write(
                    item["name"],
                    torch.tensor(item["data"], dtype=torch.int64).reshape(
                        item["shape"]
                    ),
                )
            self.expected[sid] = written
            rows.append(
                {
                    "meta_info": {
                        "spec_capture": {
                            "sample_id": sid,
                            "store_id": spec["store_id"],
                            "gen": gen,
                            "aux_layer_ids": self.aux_layer_ids,
                            "features": feats,
                        }
                    }
                }
            )
        return rows


class _FakeController:
    def __init__(self):
        self.committed: List[SampleRef] = []
        self.failed: List[Dict[str, Any]] = []
        self.leased: List[PromptTask] = []

    def register_rollout_worker(self, info):
        return info.get("worker_id") or "w0"

    def lease_prompt_tasks(self, worker_id, max_tasks):
        out, self.leased = self.leased[:max_tasks], self.leased[max_tasks:]
        return out

    def commit_samples(self, worker_id, refs):
        self.committed.extend(refs)

    def fail_prompt_tasks(self, worker_id, task_ids, *, reason, retryable):
        self.failed.append(
            {"task_ids": list(task_ids), "reason": reason, "retryable": retryable}
        )


def _task(i: int, length: int) -> PromptTask:
    return PromptTask(
        task_id=f"t{i}",
        run_id="run0",
        source_id="unit",
        payload={
            "input_ids": list(range(1, length + 1)),
            "loss_mask": [0] * (length // 2) + [1] * (length - length // 2),
        },
        max_length=length,
    )


def _eagle3_contract() -> FeatureContract:
    return FeatureContract.from_strategy(
        required_features={
            "input_ids",
            "attention_mask",
            "loss_mask",
            "hidden_state",
            "target",
        },
        aux_hidden_state_layer_ids=AUX_LAYERS,
        target_repr="hidden_state",
        target_hidden_size=HIDDEN,
    )


def _dflash_contract() -> FeatureContract:
    return FeatureContract.from_strategy(
        required_features={"input_ids", "hidden_states", "loss_mask"},
        aux_hidden_state_layer_ids=AUX_LAYERS,
        target_repr="hidden_state",
        target_hidden_size=HIDDEN,
    )


def _mk(strategy="eagle3", server=None, backend=None):
    backend = backend or _FakeMooncakeStore()
    server = server or _StubCaptureServer(backend)
    store = MooncakeFeatureStore(store=backend, store_id="run0")
    adapter = SGLangServerCaptureAdapter(
        "http://server:30000",
        store,
        run_id="run0",
        strategy=strategy,
        post_fn=server,
    )
    return backend, server, store, adapter


class TestServerCaptureAdapter(unittest.TestCase):
    def test_eagle3_refs_and_zero_copy_roundtrip(self):
        backend, server, store, adapter = _mk()
        tasks = [_task(0, 6), _task(1, 9)]
        refs = adapter.produce_refs(tasks, capture=_eagle3_contract())
        self.assertEqual(len(refs), 2)
        for task, ref in zip(tasks, refs):
            self.assertIsInstance(ref, SampleRef)
            self.assertEqual(ref.sample_id, f"run0:{task.task_id}")
            self.assertEqual(ref.strategy, "eagle3")
            self.assertEqual(ref.metadata["generation"], 1)
            self.assertEqual(ref.metadata["transport"], "sglang_server_capture")
            length = len(task.payload["input_ids"])
            self.assertEqual(
                ref.feature_specs["hidden_state"].shape,
                (1, length, len(AUX_LAYERS) * HIDDEN),
            )
            self.assertEqual(ref.feature_specs["target"].shape, (1, length, HIDDEN))
            self.assertEqual(ref.feature_specs["target"].target_repr, "hidden_state")
            # offline convention for the hidden_state train path: (B, L);
            # TargetHead.preprocess adds the trailing mask dim
            self.assertEqual(ref.feature_specs["loss_mask"].shape, (1, length))
            # zero-copy consume: bytes come back bit-exact from the fake store
            out, handle = store.get(ref)
            for name, expected in server.expected[ref.sample_id].items():
                self.assertTrue(
                    torch.equal(out[name], expected),
                    f"{name} mismatch after zero-copy roundtrip",
                )
            store.release(handle)
        # consume-once: released samples are physically freed
        self.assertFalse(any(backend._d))

    def test_dflash_schema_names_and_no_target(self):
        backend, server, store, adapter = _mk(strategy="dflash")
        refs = adapter.produce_refs([_task(0, 5)], capture=_dflash_contract())
        (ref,) = refs
        self.assertIsInstance(ref, SampleRef)
        self.assertEqual(
            sorted(ref.feature_specs), ["hidden_states", "input_ids", "loss_mask"]
        )
        self.assertEqual(
            ref.feature_specs["hidden_states"].shape,
            (1, 5, len(AUX_LAYERS) * HIDDEN),
        )
        self.assertEqual(ref.feature_specs["loss_mask"].shape, (1, 5))
        out, _ = store.get(ref)
        self.assertEqual(out["input_ids"].tolist(), [list(range(1, 6))])

    def test_aux_width_mismatch_fails_loud_and_frees_keys(self):
        backend = _FakeMooncakeStore()
        server = _StubCaptureServer(backend, aux_width=HIDDEN)  # wrong width
        _, _, store, adapter = _mk(server=server, backend=backend)
        (result,) = adapter.produce_refs([_task(0, 4)], capture=_eagle3_contract())
        self.assertIsInstance(result, ServerCaptureFailure)
        self.assertFalse(result.retryable)
        self.assertIn("aux width", result.reason)
        # the mismatched sample's server-written keys were freed via abort
        self.assertFalse(any(backend._d), f"leaked keys: {list(backend._d)}")

    def test_per_task_server_error_becomes_failure_marker(self):
        backend = _FakeMooncakeStore()
        server = _StubCaptureServer(backend, error_sample_ids={"run0:t0"})
        _, _, store, adapter = _mk(server=server, backend=backend)
        results = adapter.produce_refs(
            [_task(0, 4), _task(1, 4)], capture=_eagle3_contract()
        )
        self.assertIsInstance(results[0], ServerCaptureFailure)
        self.assertIn("injected sink error", results[0].reason)
        self.assertIsInstance(results[1], SampleRef)

    def test_rollout_worker_ref_path(self):
        from specforge.inference.rollout_worker import RolloutWorker

        backend = _FakeMooncakeStore()
        server = _StubCaptureServer(backend, error_sample_ids={"run0:t1"})
        _, _, store, adapter = _mk(server=server, backend=backend)
        controller = _FakeController()
        controller.leased = [_task(0, 4), _task(1, 4), _task(2, 7)]
        worker = RolloutWorker(
            controller,
            store,
            adapter,
            _eagle3_contract(),
            run_id="run0",
            worker_id="w0",
        )
        worker.start()
        refs = worker.run_once(max_tasks=8)
        self.assertEqual(len(refs), 2)
        self.assertEqual(len(controller.committed), 2)
        self.assertEqual(len(controller.failed), 1)
        self.assertEqual(controller.failed[0]["task_ids"], ["t1"])
        self.assertIn("injected sink error", controller.failed[0]["reason"])

    def test_adopt_enables_abort_of_server_written_keys(self):
        backend, server, store, adapter = _mk()
        (ref,) = adapter.produce_refs([_task(0, 4)], capture=_eagle3_contract())
        self.assertTrue(any(backend._d))
        store.abort(ref.sample_id, reason="unit")  # adopt() made this effective
        self.assertFalse(any(backend._d))
        with self.assertRaises(KeyError):
            store.get(ref)

    def test_retry_bumps_generation(self):
        backend, server, store, adapter = _mk()
        task = PromptTask(
            task_id="t0",
            run_id="run0",
            source_id="unit",
            payload={"input_ids": [1, 2, 3]},
            max_length=3,
            attempt=2,
        )
        (ref,) = adapter.produce_refs([task], capture=_eagle3_contract())
        self.assertEqual(ref.metadata["generation"], 3)
        self.assertTrue(any("/g3/" in k for k in backend._d))

    def test_verify_specs_matches_tensor_verification_semantics(self):
        contract = _eagle3_contract()
        specs = {
            n: FeatureSpec(name=n, shape=s, dtype="bfloat16")
            for n, s in {
                "input_ids": (1, 4),
                "attention_mask": (1, 4),
                "loss_mask": (1, 4, 1),
                "hidden_state": (1, 4, len(AUX_LAYERS) * HIDDEN),
                "target": (1, 4, HIDDEN),
            }.items()
        }
        verify_feature_contract_specs(
            specs, contract, sample_id="s0", recorded_aux_layer_ids=AUX_LAYERS
        )
        bad = dict(specs)
        bad["hidden_state"] = FeatureSpec(
            name="hidden_state", shape=(1, 4, HIDDEN), dtype="bfloat16"
        )
        with self.assertRaises(FeatureContractError):
            verify_feature_contract_specs(bad, contract, sample_id="s0")
        with self.assertRaises(FeatureContractError):
            verify_feature_contract_specs(
                specs,
                contract,
                sample_id="s0",
                recorded_aux_layer_ids=(1, 2, 3),
            )

    def test_unknown_strategy_raises(self):
        with self.assertRaises(KeyError):
            resolve_server_capture_schema("nonexistent")


class TestServerCaptureProducerWiring(unittest.TestCase):
    """The example's exact path: build_disagg_online_producer(feature_source=...)."""

    def test_producer_streams_refs_via_feature_source(self):
        from specforge.launch import build_disagg_online_producer
        from specforge.runtime.data_plane.streaming_ref_channel import (
            StreamingRefChannel,
        )

        backend = _FakeMooncakeStore()
        server = _StubCaptureServer(backend)
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        adapter = SGLangServerCaptureAdapter(
            "http://server:30000",
            store,
            run_id="run0",
            strategy="dflash",
            post_fn=server,
        )
        prompts = [
            {
                "payload": {
                    "input_ids": list(range(1, 5 + i)),
                    "loss_mask": [1] * (4 + i),
                }
            }
            for i in range(3)
        ]
        channel = StreamingRefChannel(
            os.path.join(tempfile.mkdtemp(prefix="sc_prod_"), "refs.jsonl")
        )
        _workers, drive = build_disagg_online_producer(
            strategy="dflash",
            feature_source=adapter,
            prompts=prompts,
            feature_store=store,
            channel=channel,
            run_id="run0",
            target_hidden_size=HIDDEN,
            target_repr=None,
            aux_hidden_state_layer_ids=AUX_LAYERS,
        )
        produced = drive()
        self.assertEqual(produced, len(prompts))
        self.assertEqual(channel.published, len(prompts))
        self.assertTrue(channel.is_closed())


if __name__ == "__main__":
    unittest.main()
