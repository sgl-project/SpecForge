# coding=utf-8
"""End-to-end unit tests for the online-live producer (no GPU, no SGLang).

A stub "sink" plays the live-patched serving engine: it POSTs capture records
over real HTTP to the producer-hosted :class:`CaptureIntakeServer` while the
REAL live stack runs underneath — ``LiveIntakeRefSource`` validation,
``adopt``, watermark shedding, byte accounting, ``StreamingRefChannel``
publication, and terminal cleanup via ``_finalize_online_producer``."""

import json
import os
import tempfile
import threading
import time
import unittest
from typing import Any, Dict
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from specforge.inference.adapters.live_intake import LiveIntakeRefSource
from specforge.inference.adapters.server_capture import ServerCaptureSchema
from specforge.inference.capture import CaptureConfig
from specforge.launch import build_disagg_live_producer
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel

HIDDEN = 8
AUX_LAYERS = (2, 5, 8)


class _FakeStore:
    store_id = "store0"

    def __init__(self):
        self.adopted = []
        self.aborted = []
        self.gc_calls = 0

    def adopt(self, ref):
        self.adopted.append(ref.sample_id)

    def abort(self, sample_id, *, reason="aborted"):
        self.aborted.append((sample_id, reason))

    def gc(self):
        self.gc_calls += 1
        return {}


def _source(store) -> LiveIntakeRefSource:
    return LiveIntakeRefSource(
        store,
        run_id="run0",
        algorithm="dspark",
        schema=ServerCaptureSchema(
            aux_feature="hidden_states",
            last_hidden_feature="target_last_hidden_states",
            passthrough=(
                ("input_ids", "input_ids", ()),
                ("loss_mask", "loss_mask", ()),
            ),
        ),
        capture=CaptureConfig.from_strategy(
            required_features={
                "input_ids",
                "hidden_states",
                "loss_mask",
                "target_last_hidden_states",
            },
            aux_hidden_state_layer_ids=AUX_LAYERS,
            target_repr="hidden_state",
            target_hidden_size=HIDDEN,
        ),
        max_num_tokens=64,
    )


def _record(sample_id: str, length: int = 5) -> Dict[str, Any]:
    aux_width = len(AUX_LAYERS) * HIDDEN
    return {
        "sample_id": sample_id,
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


class LiveProducerTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="live_producer_")
        self.channel_path = os.path.join(self.work, "refs.jsonl")
        self.channel = StreamingRefChannel(self.channel_path)
        self.store = _FakeStore()

    def _build(self, **overrides):
        kwargs = dict(
            feature_store=self.store,
            channel=self.channel,
            ref_source=_source(self.store),
            intake_host="127.0.0.1",
            intake_port=0,
            in_flight_high_watermark=4,
            in_flight_low_watermark=2,
            backpressure_poll_s=0.01,
            gc_interval_s=0.05,
        )
        kwargs.update(overrides)
        return build_disagg_live_producer(**kwargs)

    def _drive_async(self, drive):
        result: Dict[str, Any] = {}

        def run():
            try:
                result["produced"] = drive(should_stop=self.channel.consumer_stopped)
            except BaseException as exc:  # noqa: BLE001 — surfaced by the test
                result["error"] = exc

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, result

    def _post(self, server, record):
        body = json.dumps(record).encode("utf-8")
        request = Request(
            f"http://127.0.0.1:{server.port}/v1/spec-capture/records",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=5.0) as response:
                return response.status, json.load(response)
        except HTTPError as exc:
            return exc.code, json.load(exc)

    def _await(self, predicate, timeout_s: float = 5.0):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError("condition not met before timeout")

    def test_pushed_records_are_adopted_published_and_deduped(self):
        server, drive = self._build()
        self.channel.publish_consumer_quantum(2)
        thread, result = self._drive_async(drive)
        self._await(lambda: server.port != 0 and drive.stats()["quantum"] == 2)

        status, payload = self._post(server, _record("s0"))
        self.assertEqual((status, payload["status"]), (200, "accepted"))
        status, payload = self._post(server, _record("s0"))
        self.assertEqual((status, payload["status"]), (200, "duplicate"))
        status, payload = self._post(server, _record("s1", length=7))

        self.assertEqual(self.store.adopted, ["s0", "s1"])
        refs = self.channel.poll()
        self.assertEqual([ref.sample_id for ref in refs], ["s0", "s1"])
        self.assertEqual(refs[0].metadata["transport"], "sglang_live_capture")
        self.assertEqual(refs[1].num_tokens, 7)

        self._await(lambda: self.store.gc_calls > 0)
        self.channel.mark_consumer_done()
        thread.join(timeout=5.0)
        self.assertEqual(result.get("produced"), 2)
        self.assertTrue(self.channel.is_closed())

    def test_invalid_records_are_rejected_without_adoption(self):
        server, drive = self._build()
        self.channel.publish_consumer_quantum(1)
        thread, _result = self._drive_async(drive)
        self._await(lambda: drive.stats()["quantum"] == 1)

        wrong_store = _record("bad0")
        wrong_store["store_id"] = "other"
        status, payload = self._post(server, wrong_store)
        self.assertEqual(status, 422)
        self.assertIn("store_id", payload["detail"])

        too_long = _record("bad1", length=65)
        status, _ = self._post(server, too_long)
        self.assertEqual(status, 422)

        self.assertEqual(self.store.adopted, [])
        self.assertEqual(self.channel.poll(), [])
        self.channel.mark_consumer_done()
        thread.join(timeout=5.0)

    def test_watermark_shed_and_resume_hysteresis(self):
        server, drive = self._build(
            in_flight_high_watermark=2, in_flight_low_watermark=1
        )
        self.channel.publish_consumer_quantum(1)
        thread, _result = self._drive_async(drive)
        self._await(lambda: drive.stats()["quantum"] == 1)

        self.assertEqual(self._post(server, _record("s0"))[0], 200)
        self.assertEqual(self._post(server, _record("s1"))[0], 200)
        # in_flight == high watermark == 2 -> shed until consumption drops it.
        status, payload = self._post(server, _record("s2"))
        self.assertEqual((status, payload["status"]), (429, "shed"))

        self.channel.mark_consumed(1)  # in_flight 1 == low watermark -> resume
        self.assertEqual(self._post(server, _record("s3"))[0], 200)
        self.assertEqual(drive.stats()["shed"], 1)

        self.channel.mark_consumer_done()
        thread.join(timeout=5.0)

    def test_resident_byte_hard_cap_sheds_instead_of_failing(self):
        record_bytes = 5 * (len(AUX_LAYERS) * HIDDEN + HIDDEN) * 2 + 2 * 5 * 8
        server, drive = self._build(
            feature_store_max_resident_bytes=record_bytes + 1
        )
        self.channel.publish_consumer_quantum(1)
        thread, _result = self._drive_async(drive)
        self._await(lambda: drive.stats()["quantum"] == 1)

        self.assertEqual(self._post(server, _record("s0"))[0], 200)
        status, payload = self._post(server, _record("s1"))
        self.assertEqual((status, payload["status"]), (429, "shed"))
        self.assertIn("hard cap", payload["detail"])

        self.channel.mark_consumed(1)  # frees the resident bytes
        self.assertEqual(self._post(server, _record("s2"))[0], 200)

        self.channel.mark_consumer_done()
        thread.join(timeout=5.0)
        self.assertEqual(self.store.aborted, [])

    def test_setup_failure_publishes_the_failure_sentinel(self):
        _server, drive = self._build(peer_wait_timeout_s=0.05)
        with self.assertRaises(TimeoutError):
            drive()
        self.assertIsNotNone(self.channel.failure())

    def test_undersized_watermarks_fail_against_the_consumer_quantum(self):
        _server, drive = self._build(
            in_flight_high_watermark=4, in_flight_low_watermark=2
        )
        self.channel.publish_consumer_quantum(3)
        with self.assertRaisesRegex(ValueError, "low watermark"):
            drive()


if __name__ == "__main__":
    unittest.main()
