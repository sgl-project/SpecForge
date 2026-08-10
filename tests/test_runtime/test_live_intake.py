# coding=utf-8
"""Unit tests for the online-live intake: pushed records -> verified refs.

No GPU and no real server: records are the JSON dicts a live-patched SGLang
sink would POST after writing tensors into Mooncake.  These tests cover the
receiving half — the handshake ``config_payload`` document, record validation
in :class:`LiveIntakeRefSource`, and the :class:`CaptureIntakeServer` HTTP
protocol (dedup, shed, reject)."""

import json
import unittest
from typing import Any, Dict
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from specforge.inference.adapters.live_intake import LiveIntakeRefSource
from specforge.inference.adapters.server_capture import ServerCaptureSchema
from specforge.inference.capture import CaptureConfig, CaptureMismatchError
from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.intake_server import CaptureIntakeServer

HIDDEN = 8
AUX_LAYERS = (2, 5, 8)
MAX_TOKENS = 64


class _StubStore:
    store_id = "store0"

    def __init__(self):
        self.adopted = []

    def adopt(self, ref):
        self.adopted.append(ref)


def _dspark_schema() -> ServerCaptureSchema:
    return ServerCaptureSchema(
        aux_feature="hidden_states",
        last_hidden_feature="target_last_hidden_states",
        passthrough=(("input_ids", "input_ids", ()), ("loss_mask", "loss_mask", ())),
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


def _source(store=None) -> LiveIntakeRefSource:
    return LiveIntakeRefSource(
        store or _StubStore(),
        run_id="run0",
        algorithm="dspark",
        schema=_dspark_schema(),
        capture=_dspark_contract(),
        max_num_tokens=MAX_TOKENS,
        target_model_version="target-v1",
    )


def _record(length: int = 5, **overrides) -> Dict[str, Any]:
    aux_width = len(AUX_LAYERS) * HIDDEN
    record = {
        "sample_id": "live-abc123",
        "store_id": "store0",
        "gen": 1,
        "num_tokens": length,
        "aux_layer_ids": list(AUX_LAYERS),
        "features": {
            "hidden_states": {"shape": [1, length, aux_width], "dtype": "bfloat16"},
            "target_last_hidden_states": {
                "shape": [1, length, HIDDEN],
                "dtype": "bfloat16",
            },
            "input_ids": {"shape": [1, length], "dtype": "int64"},
            "loss_mask": {"shape": [1, length], "dtype": "int64"},
        },
    }
    record.update(overrides)
    return record


class LiveIntakeRefSourceTest(unittest.TestCase):
    def test_config_payload_carries_the_capture_document(self):
        payload = _source().config_payload()
        self.assertEqual(payload["store_id"], "store0")
        self.assertEqual(payload["gen"], 1)
        self.assertEqual(
            payload["features"],
            {"aux": "hidden_states", "last_hidden": "target_last_hidden_states"},
        )
        self.assertEqual(
            payload["passthrough"],
            [
                {"name": "input_ids", "source": "tokens"},
                {"name": "loss_mask", "source": "ones"},
            ],
        )
        self.assertEqual(payload["min_num_tokens"], 2)
        self.assertEqual(payload["max_num_tokens"], MAX_TOKENS)

    def test_config_payload_rejects_trailing_passthrough_shapes(self):
        schema = ServerCaptureSchema(
            aux_feature="hidden_states",
            last_hidden_feature=None,
            passthrough=(("depths", "depths", (4,)),),
        )
        source = LiveIntakeRefSource(
            _StubStore(),
            run_id="run0",
            algorithm="peagle",
            schema=schema,
            capture=_dspark_contract(),
            max_num_tokens=MAX_TOKENS,
        )
        with self.assertRaisesRegex(ValueError, "trailing shape"):
            source.config_payload()

    def test_record_becomes_a_committed_ready_ref(self):
        ref = _source().ref_from_record(_record(length=6, origin="10.0.0.9"))
        self.assertIsInstance(ref, SampleRef)
        self.assertEqual(ref.sample_id, "live-abc123")
        self.assertEqual(ref.num_tokens, 6)
        self.assertEqual(ref.feature_store_uri, "mooncake://store0/live-abc123")
        self.assertEqual(ref.feature_keys["input_ids"], "live-abc123/input_ids")
        self.assertEqual(ref.metadata["transport"], "sglang_live_capture")
        self.assertEqual(ref.metadata["server"], "10.0.0.9")
        self.assertEqual(ref.metadata["generation"], 1)
        self.assertEqual(
            ref.feature_specs["target_last_hidden_states"].target_repr,
            "hidden_state",
        )

    def test_identity_and_bounds_violations_are_rejected(self):
        source = _source()
        cases = {
            "store_id": _record(store_id="other-store"),
            "gen": _record(gen=2),
            "num_tokens": _record(num_tokens=0),
            "one_token_warmup": _record(num_tokens=1),
            "cap": _record(num_tokens=MAX_TOKENS + 1),
            "features": _record(features={}),
        }
        for name, record in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                source.ref_from_record(record)

    def test_contract_violations_are_rejected(self):
        source = _source()
        extra = _record()
        extra["features"]["surprise"] = {"shape": [1, 5], "dtype": "int64"}
        with self.assertRaises(CaptureMismatchError):
            source.ref_from_record(extra)

        short = _record()
        short["features"]["input_ids"]["shape"] = [1, 3]
        with self.assertRaisesRegex(CaptureMismatchError, "seq len"):
            source.ref_from_record(short)

        wrong_width = _record()
        wrong_width["features"]["hidden_states"]["shape"] = [1, 5, HIDDEN]
        with self.assertRaisesRegex(CaptureMismatchError, "aux width"):
            source.ref_from_record(wrong_width)

        wrong_layers = _record(aux_layer_ids=[1, 2, 3])
        with self.assertRaisesRegex(CaptureMismatchError, "aux-layer id"):
            source.ref_from_record(wrong_layers)

        missing_layers = _record(aux_layer_ids=None)
        with self.assertRaisesRegex(CaptureMismatchError, "omitted aux-layer"):
            source.ref_from_record(missing_layers)


def _get(url: str):
    with urlopen(url, timeout=5.0) as response:
        return response.status, json.load(response)


def _post(url: str, payload) -> tuple:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urlopen(request, timeout=5.0) as response:
            return response.status, json.load(response)
    except HTTPError as exc:
        return exc.code, json.load(exc)


class CaptureIntakeServerTest(unittest.TestCase):
    def setUp(self):
        self.dispatched = []
        self.reply = ("accepted", "")

        def on_record(record):
            self.dispatched.append(record)
            return self.reply

        self.server = CaptureIntakeServer(
            "127.0.0.1",
            0,
            config_payload={"store_id": "store0", "gen": 1},
            on_record=on_record,
        ).start()
        self.origin = f"http://127.0.0.1:{self.server.port}"
        self.addCleanup(self.server.stop)

    def test_config_endpoint_serves_the_handshake_document(self):
        status, payload = _get(f"{self.origin}/v1/spec-capture/config")
        self.assertEqual(status, 200)
        self.assertEqual(payload, {"store_id": "store0", "gen": 1})

    def test_accepted_record_is_dispatched_once_and_deduped_on_retry(self):
        record = {"sample_id": "s0", "num_tokens": 4}
        status, payload = _post(f"{self.origin}/v1/spec-capture/records", record)
        self.assertEqual((status, payload["status"]), (200, "accepted"))
        status, payload = _post(f"{self.origin}/v1/spec-capture/records", record)
        self.assertEqual((status, payload["status"]), (200, "duplicate"))
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(self.dispatched[0]["origin"], "127.0.0.1")
        self.assertEqual(
            self.server.stats(),
            {"accepted": 1, "shed": 0, "rejected": 0, "duplicate": 1},
        )

    def test_shed_and_rejected_records_return_non_2xx_and_are_not_deduped(self):
        self.reply = ("shed", "watermark")
        status, _ = _post(
            f"{self.origin}/v1/spec-capture/records", {"sample_id": "s1"}
        )
        self.assertEqual(status, 429)

        self.reply = ("rejected", "bad record")
        status, payload = _post(
            f"{self.origin}/v1/spec-capture/records", {"sample_id": "s1"}
        )
        self.assertEqual(status, 422)
        self.assertEqual(payload["detail"], "bad record")
        self.assertEqual(len(self.dispatched), 2)

    def test_malformed_requests_never_reach_the_dispatcher(self):
        status, _ = _post(f"{self.origin}/v1/spec-capture/records", ["not", "a", "dict"])
        self.assertEqual(status, 400)
        status, _ = _post(f"{self.origin}/v1/spec-capture/records", {"sample_id": ""})
        self.assertEqual(status, 400)
        status, _ = _post(f"{self.origin}/v1/other", {"sample_id": "s2"})
        self.assertEqual(status, 404)
        self.assertEqual(self.dispatched, [])

    def test_dispatcher_crash_maps_to_500(self):
        self.server.on_record = lambda record: 1 / 0
        status, payload = _post(
            f"{self.origin}/v1/spec-capture/records", {"sample_id": "s3"}
        )
        self.assertEqual(status, 500)
        self.assertIn("ZeroDivisionError", payload["error"])


if __name__ == "__main__":
    unittest.main()
