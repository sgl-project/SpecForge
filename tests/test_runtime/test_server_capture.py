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

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.inference.adapters.server_capture import (
    ServerCaptureFailure,
    ServerCaptureSchema,
    SGLangServerCaptureAdapter,
)
from specforge.inference.capture import (
    CaptureConfig,
    CaptureMismatchError,
    verify_capture_specs,
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
        self.aux_layer_ids = None if aux_layer_ids is None else list(aux_layer_ids)
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


def _eagle3_contract() -> CaptureConfig:
    return CaptureConfig.from_strategy(
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


def _dflash_contract() -> CaptureConfig:
    return CaptureConfig.from_strategy(
        required_features={"input_ids", "hidden_states", "loss_mask"},
        aux_hidden_state_layer_ids=AUX_LAYERS,
        target_repr="hidden_state",
        target_hidden_size=HIDDEN,
    )


def _dspark_contract() -> CaptureConfig:
    return CaptureConfig.from_strategy(
        required_features={
            "input_ids",
            "hidden_states",
            "loss_mask",
            "target_last_hidden_states",
        },
        aux_hidden_state_layer_ids=AUX_LAYERS,
        target_repr="hidden_state",
        target_hidden_size=HIDDEN,
    )


def _capture_schema(algorithm: str) -> ServerCaptureSchema:
    registration = builtin_algorithm_registry().resolve(algorithm)
    layout = registration.providers.server_streaming_for("text").layout
    return ServerCaptureSchema(
        aux_feature=layout.aux_feature,
        last_hidden_feature=layout.last_hidden_feature,
        passthrough=layout.passthrough,
        attention_mask_feature=layout.attention_mask_feature,
    )


class _GenericRequestInputAdapter:
    def __init__(self, request_inputs):
        self.request_inputs = request_inputs
        self.seen_tasks = None

    def load_input_tools(self, config):
        return config

    def prepare_prompts(self, config, input_tools, *, draft_config):
        return []

    def build_request_inputs(self, tasks):
        self.seen_tasks = list(tasks)
        return self.request_inputs


def _mk(
    algorithm="eagle3",
    server=None,
    backend=None,
    request_input_adapter=None,
):
    backend = backend or _FakeMooncakeStore()
    server = server or _StubCaptureServer(backend)
    store = MooncakeFeatureStore(store=backend, store_id="run0")
    adapter = SGLangServerCaptureAdapter(
        "http://server:30000",
        store,
        run_id="run0",
        algorithm=algorithm,
        schema=_capture_schema(algorithm),
        request_input_adapter=request_input_adapter,
        post_fn=server,
    )
    return backend, server, store, adapter


class TestServerCaptureAdapter(unittest.TestCase):
    def test_generic_adapter_inputs_merge_with_runtime_owned_request_fields(self):
        backend = _FakeMooncakeStore()
        server = _StubCaptureServer(backend)
        posted = []

        def recording_server(url, json_body, timeout):
            posted.append(json_body)
            return server(url, json_body, timeout)

        tasks = [_task(0, 4), _task(1, 6)]
        model_inputs = {
            "input_ids": [task.payload["input_ids"] for task in tasks],
            "multi_modal_data": {
                "image": ["image-0", "image-1"],
            },
        }
        input_adapter = _GenericRequestInputAdapter(model_inputs)
        _, _, _, capture_adapter = _mk(
            server=recording_server,
            backend=backend,
            request_input_adapter=input_adapter,
        )

        refs = capture_adapter.produce_refs(tasks, capture=_eagle3_contract())

        self.assertTrue(all(isinstance(ref, SampleRef) for ref in refs))
        self.assertEqual(tasks, input_adapter.seen_tasks)
        self.assertEqual(1, len(posted))
        request = posted[0]
        self.assertEqual(model_inputs["input_ids"], request["input_ids"])
        self.assertEqual(
            model_inputs["multi_modal_data"],
            request["multi_modal_data"],
        )
        self.assertEqual(
            {"temperature": 0.0, "max_new_tokens": 1},
            request["sampling_params"],
        )
        self.assertEqual(2, len(request["spec_capture"]))

    def test_generic_adapter_cannot_override_runtime_owned_request_fields(self):
        task = _task(0, 4)
        for reserved_field in ("sampling_params", "spec_capture"):
            with self.subTest(field=reserved_field):
                input_adapter = _GenericRequestInputAdapter(
                    {
                        "input_ids": [task.payload["input_ids"]],
                        reserved_field: "plugin-owned",
                    }
                )
                _, _, _, capture_adapter = _mk(request_input_adapter=input_adapter)

                with self.assertRaisesRegex(
                    ValueError,
                    f"runtime-owned request fields: \\['{reserved_field}'\\]",
                ):
                    capture_adapter.produce_refs(
                        [task],
                        capture=_eagle3_contract(),
                    )

    def test_generic_adapter_must_return_nonempty_model_input_mapping(self):
        task = _task(0, 4)
        cases = (
            ([], TypeError, "must return a mapping"),
            ({}, ValueError, "returned no model inputs"),
        )
        for request_inputs, exception_type, message in cases:
            with self.subTest(request_inputs=request_inputs):
                _, _, _, capture_adapter = _mk(
                    request_input_adapter=_GenericRequestInputAdapter(request_inputs)
                )
                with self.assertRaisesRegex(exception_type, message):
                    capture_adapter.produce_refs(
                        [task],
                        capture=_eagle3_contract(),
                    )

    def test_required_loss_mask_never_defaults_to_all_ones(self):
        task = PromptTask(
            task_id="t0",
            run_id="run0",
            source_id="unit",
            payload={"input_ids": [1, 2, 3]},
            max_length=3,
        )
        backend, server, _, adapter = _mk()

        with self.assertRaisesRegex(
            ValueError, "required capture payload 'loss_mask' is missing"
        ):
            adapter.produce_refs([task], capture=_eagle3_contract())

        self.assertFalse(server.expected)
        self.assertFalse(backend._d)

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
        backend, server, store, adapter = _mk(algorithm="dflash")
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

    def test_dspark_schema_includes_target_last_hidden(self):
        backend, server, store, adapter = _mk(algorithm="dspark")
        refs = adapter.produce_refs([_task(0, 5)], capture=_dspark_contract())
        (ref,) = refs
        self.assertIsInstance(ref, SampleRef)
        self.assertEqual(ref.strategy, "dspark")
        self.assertEqual(
            sorted(ref.feature_specs),
            [
                "hidden_states",
                "input_ids",
                "loss_mask",
                "target_last_hidden_states",
            ],
        )
        self.assertEqual(
            ref.feature_specs["hidden_states"].shape,
            (1, 5, len(AUX_LAYERS) * HIDDEN),
        )
        self.assertEqual(
            ref.feature_specs["target_last_hidden_states"].shape,
            (1, 5, HIDDEN),
        )
        self.assertEqual(
            ref.feature_specs["target_last_hidden_states"].target_repr,
            "hidden_state",
        )
        out, _ = store.get(ref)
        self.assertIn("target_last_hidden_states", out)

    def test_batch_response_wrappers_are_normalized_by_sample_id(self):
        backend = _FakeMooncakeStore()
        server = _StubCaptureServer(backend)

        def wrapped_server(url, json_body, timeout):
            rows = server(url, json_body, timeout)
            results = [row["meta_info"]["spec_capture"] for row in rows]
            for row in rows:
                row["meta_info"]["spec_capture"] = [results]
            return [[row] for row in rows]

        _, _, _, adapter = _mk(
            algorithm="dflash", server=wrapped_server, backend=backend
        )
        refs = adapter.produce_refs(
            [_task(0, 5), _task(1, 6)], capture=_dflash_contract()
        )
        self.assertTrue(all(isinstance(ref, SampleRef) for ref in refs))

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

    def test_missing_aux_layer_ids_fails_loud_and_frees_keys(self):
        backend = _FakeMooncakeStore()
        server = _StubCaptureServer(backend, aux_layer_ids=None)
        _, _, _, adapter = _mk(server=server, backend=backend)

        (result,) = adapter.produce_refs([_task(0, 4)], capture=_eagle3_contract())

        self.assertIsInstance(result, ServerCaptureFailure)
        self.assertFalse(result.retryable)
        self.assertIn("capture omitted aux-layer ids", result.reason)
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

    def test_retry_reuses_generation_after_a_lost_response(self):
        backend, server, store, adapter = _mk()
        initial = PromptTask(
            task_id="t0",
            run_id="run0",
            source_id="unit",
            payload={"input_ids": [1, 2, 3], "loss_mask": [0, 1, 1]},
            max_length=3,
        )
        original_post = adapter.post_fn
        requests = []

        def lose_first_response(url, json_body, timeout):
            requests.append(json_body["spec_capture"][0])
            response = original_post(url, json_body, timeout)
            if len(requests) == 1:
                raise ConnectionError("response lost after server write")
            return response

        adapter.post_fn = lose_first_response
        with self.assertRaisesRegex(ConnectionError, "response lost"):
            adapter.produce_refs([initial], capture=_eagle3_contract())
        self.assertTrue(backend._d)
        self.assertTrue(all("/g1/" in key for key in backend._d))

        retry = PromptTask(
            task_id=initial.task_id,
            run_id=initial.run_id,
            source_id=initial.source_id,
            payload=initial.payload,
            max_length=initial.max_length,
            attempt=1,
        )
        (ref,) = adapter.produce_refs([retry], capture=_eagle3_contract())

        self.assertEqual(1, ref.metadata["generation"])
        self.assertEqual([False, True], [request["replace"] for request in requests])
        self.assertEqual(0, store.discard_external_attempts())
        self.assertTrue(all("/g1/" in key for key in backend._d))
        self.assertFalse(any("/g2/" in key for key in backend._d))

    def test_terminal_lost_response_is_reclaimed_without_a_ref(self):
        backend, server, store, adapter = _mk()

        def lose_response(url, json_body, timeout):
            server(url, json_body, timeout)
            raise ConnectionError("response lost after server write")

        adapter.post_fn = lose_response
        with self.assertRaisesRegex(ConnectionError, "response lost"):
            adapter.produce_refs([_task(0, 4)], capture=_eagle3_contract())

        self.assertTrue(backend._d)
        self.assertEqual(1, store.health()["provisional_external"])
        self.assertEqual(
            1,
            store.discard_external_attempts(reason="terminal-producer-cleanup"),
        )
        self.assertFalse(backend._d)
        self.assertEqual(0, store.health()["provisional_external"])

    def test_later_invalid_row_does_not_strand_an_adopted_prefix(self):
        backend = _FakeMooncakeStore()
        server = _StubCaptureServer(backend)

        def corrupt_second_identity(url, json_body, timeout):
            rows = server(url, json_body, timeout)
            rows[1]["meta_info"]["spec_capture"]["store_id"] = "wrong-store"
            return rows

        _, _, store, adapter = _mk(
            server=corrupt_second_identity,
            backend=backend,
        )
        with self.assertRaisesRegex(RuntimeError, "wrong object identity"):
            adapter.produce_refs(
                [_task(0, 4), _task(1, 4)],
                capture=_eagle3_contract(),
            )

        self.assertEqual(0, store.health()["resident_samples"])
        self.assertEqual(2, store.health()["provisional_external"])
        self.assertEqual(2, store.discard_external_attempts())
        self.assertFalse(backend._d)

    def test_failed_provisional_cleanup_remains_retryable(self):
        backend = _FakeMooncakeStore()
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        key = "run0/run0:t0/g1/input_ids"
        backend._d[key] = b"captured"
        store.track_external_attempt(
            "run0:t0",
            generation=1,
            feature_names=["input_ids"],
        )

        original_remove = backend.remove
        original_exists = backend.is_exist
        backend.remove = lambda _key: -1

        def unavailable(_key):
            raise RuntimeError("metadata unavailable")

        backend.is_exist = unavailable
        with self.assertRaisesRegex(RuntimeError, "metadata unavailable"):
            store.discard_external_attempts()

        health = store.health()
        self.assertEqual(1, health["resident_samples"])
        self.assertEqual(1, health["provisional_external"])
        self.assertEqual(1, health["release_pending"])

        backend.remove = original_remove
        backend.is_exist = original_exists
        store.drain_pending_removals(retry_interval_s=0)
        self.assertFalse(backend._d)
        self.assertEqual(0, store.health()["provisional_external"])

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
        verify_capture_specs(
            specs, contract, sample_id="s0", recorded_aux_layer_ids=AUX_LAYERS
        )
        bad = dict(specs)
        bad["hidden_state"] = FeatureSpec(
            name="hidden_state", shape=(1, 4, HIDDEN), dtype="bfloat16"
        )
        with self.assertRaises(CaptureMismatchError):
            verify_capture_specs(bad, contract, sample_id="s0")
        with self.assertRaises(CaptureMismatchError):
            verify_capture_specs(
                specs,
                contract,
                sample_id="s0",
                recorded_aux_layer_ids=(1, 2, 3),
            )

    def test_transport_requires_an_injected_capture_schema(self):
        backend = _FakeMooncakeStore()
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        with self.assertRaisesRegex(TypeError, "injected ServerCaptureSchema"):
            SGLangServerCaptureAdapter(
                "http://server:30000",
                store,
                run_id="run0",
                algorithm="eagle3",
                schema=object(),
            )


class TestLoaderGcPump(unittest.TestCase):
    """Consumer-side frees are lease-deferred by Mooncake (remove during the
    get() read-lease fails, e.g. -706); release() parks them and the LOADER
    must pump gc() or every consumed sample leaks until the segment fills."""

    def test_loader_gc_pump_frees_lease_deferred_removes(self):
        import time as _time

        from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader

        TTL = 0.05

        class _LeaseFake(_FakeMooncakeStore):
            """Mooncake lease semantics: is_exist GRANTS a read lease (TTL),
            and a remove during a live lease fails (-706). get() probes
            existence, so the release-time remove always fails; a retry frees
            only if it runs after the TTL and is NOT preceded by another
            exist probe (the gc() ordering bug this pins)."""

            def __init__(self):
                super().__init__()
                self._lease_until = {}

            def is_exist(self, key):
                self._lease_until[key] = _time.monotonic() + TTL
                return super().is_exist(key)

            def remove(self, key):
                if _time.monotonic() < self._lease_until.get(key, 0.0):
                    return -706
                return super().remove(key)

        backend = _LeaseFake()
        store = MooncakeFeatureStore(
            store=backend, store_id="run0", max_release_attempts=100
        )
        refs = [
            store.put(
                {"x": torch.ones(2, 2)},
                sample_id=f"s{i}",
                metadata={"run_id": "run0", "strategy": "eagle3"},
            )
            for i in range(3)
        ]
        loader = FeatureDataLoader(
            store, refs=refs, batch_size=1, strategy="eagle3", gc_interval_s=0.0
        )
        for _ in loader:
            pass
        # release-time frees parked: the get() exist-probe held the lease
        self.assertTrue(any(backend._d))
        self.assertEqual(store.health()["release_pending"], 3)
        _time.sleep(TTL * 3)  # pump cadence must exceed the lease TTL
        loader._maybe_gc()  # retries remove FIRST (no re-leasing pre-check)
        self.assertFalse(any(backend._d), f"leaked: {list(backend._d)}")
        self.assertEqual(store.health()["release_pending"], 0)


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
            algorithm="dflash",
            schema=_capture_schema("dflash"),
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
        channel.publish_consumer_quantum(1)
        _workers, drive = build_disagg_online_producer(
            algorithm=builtin_algorithm_registry().resolve("dflash"),
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
