"""Private-network inbox relay contracts."""

from __future__ import annotations

import json
import socket
import tempfile
import threading
import unittest
from unittest import mock
from urllib.request import Request, urlopen

from specforge.runtime.contracts import FeatureSpec, SampleRef
from specforge.runtime.data_plane.http_inbox import InboxHTTPServer, RemoteInboxChannel
from specforge.runtime.data_plane.ref_distributor import RefDistributor
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel


def _ref(sample_id: str) -> SampleRef:
    return SampleRef(
        sample_id=sample_id,
        run_id="run0",
        source_task_id=f"task-{sample_id}",
        feature_store_uri=f"mooncake://run0/{sample_id}",
        feature_keys={"hidden_state": f"{sample_id}/hidden_state"},
        feature_specs={
            "hidden_state": FeatureSpec(
                name="hidden_state", shape=(2, 4), dtype="float32"
            )
        },
        strategy="dspark",
        metadata={"target_repr": "hidden_state"},
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class TestHTTPInbox(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="http-inbox-")
        self.path = RefDistributor.inbox_path(self.work, 1)
        self.local = StreamingRefChannel(self.path)
        self.origin = f"http://127.0.0.1:{_free_port()}"
        self.server = InboxHTTPServer(
            self.work, 2, self.origin, bind_host="127.0.0.1"
        ).start()
        self.remote = RemoteInboxChannel(self.origin, 1)

    def tearDown(self):
        self.server.stop()

    def test_tail_read_close_and_consumed_counter(self):
        self.local.publish_batch([_ref("s0"), _ref("s1")])
        self.assertEqual([ref.sample_id for ref in self.remote.poll()], ["s0", "s1"])
        self.assertEqual(self.remote.poll(), [])

        self.remote.mark_consumed(2)
        self.assertEqual(self.local.consumed_remote(), 2)

        self.remote.mark_consumed(1)
        self.assertEqual(self.local.consumed_remote(), 3)

        self.local.close()
        self.assertTrue(self.remote.is_closed())

    def test_status_probe_does_not_discard_unpolled_refs(self):
        self.local.publish(_ref("s0"))
        self.assertFalse(self.remote.is_closed())
        self.assertIsNone(self.remote.failure())
        self.assertEqual([ref.sample_id for ref in self.remote.poll()], ["s0"])

    def test_failure_is_forwarded(self):
        failure = self.path + ".failed"
        with open(failure, "w", encoding="utf-8") as stream:
            stream.write("capture failed")
        self.assertIn("capture failed", self.remote.failure())

    def test_consumed_target_is_idempotent_under_concurrent_retries(self):
        body = json.dumps({"target": 5}).encode("utf-8")

        def post_target():
            request = Request(
                f"{self.origin}/v1/inboxes/1/consumed",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=2.0) as response:
                self.assertEqual(json.load(response)["consumed"], 5)

        threads = [threading.Thread(target=post_target) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(self.local.consumed_remote(), 5)

    def test_concurrent_client_acks_accumulate_exactly(self):
        # The background durable ack and loader failure settlement can both
        # advance one client's target; neither update may be lost.
        errors = []

        def ack_many():
            try:
                for _ in range(25):
                    self.remote.mark_consumed(1)
            except BaseException as exc:  # surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=ack_many) for _ in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.local.consumed_remote(), 150)

    def test_pull_treats_connection_reset_as_transient(self):
        with mock.patch(
            "specforge.runtime.data_plane.http_inbox.urlopen",
            side_effect=ConnectionResetError("peer reset"),
        ):
            self.assertEqual(self.remote.poll(), [])

    def test_invalid_origin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "http://host:port"):
            RemoteInboxChannel("https://trainer.example:35900/path", 1)

    def test_split_multibyte_utf8_across_pulls_does_not_corrupt_stream(self):
        from specforge.runtime.data_plane.ref_serialization import ref_to_dict

        payload = (
            json.dumps(ref_to_dict(_ref("样本")), ensure_ascii=False) + "\n"
        ).encode("utf-8")
        split = payload.index("样".encode("utf-8")) + 1
        with open(self.path, "wb") as stream:
            stream.write(payload[:split])

        self.assertEqual(self.remote.poll(), [])

        with open(self.path, "ab") as stream:
            stream.write(payload[split:])

        got = self.remote.poll()
        self.assertEqual([ref.sample_id for ref in got], ["样本"])
        self.assertEqual(self.remote.poll(), [])

    def test_corrupt_payload_does_not_advance_read_offset(self):
        with open(self.path, "wb") as stream:
            stream.write(b"\xff\n")

        with self.assertRaises(UnicodeDecodeError):
            self.remote.poll()
        self.assertEqual(self.remote._read_offset, 0)

    def test_pull_surfaces_http_errors_loudly(self):
        from urllib.error import HTTPError

        bad_rank = RemoteInboxChannel(self.origin, 99)
        with (
            self.assertLogs("specforge.runtime.data_plane.http_inbox", level="ERROR"),
            self.assertRaises(HTTPError) as ctx_404,
        ):
            bad_rank.poll()
        self.assertEqual(ctx_404.exception.code, 404)

        err_500 = HTTPError(self.remote.path, 500, "Internal Server Error", {}, None)
        with (
            mock.patch(
                "specforge.runtime.data_plane.http_inbox.urlopen", side_effect=err_500
            ),
            self.assertLogs("specforge.runtime.data_plane.http_inbox", level="ERROR"),
            self.assertRaises(HTTPError) as ctx_500,
        ):
            self.remote.poll()
        self.assertEqual(ctx_500.exception.code, 500)

    def test_inbox_server_stop_closes_socket_even_without_start(self):
        origin = f"http://127.0.0.1:{_free_port()}"
        unstarted = InboxHTTPServer(self.work, 2, origin, bind_host="127.0.0.1")
        self.assertNotEqual(unstarted._httpd.socket.fileno(), -1)
        unstarted.stop()
        self.assertEqual(unstarted._httpd.socket.fileno(), -1)

    def test_concurrent_poll_returns_each_ref_once(self):
        total = 40
        collected = []
        collected_lock = threading.Lock()
        errors = []
        start_barrier = threading.Barrier(4)

        def reader():
            try:
                start_barrier.wait(timeout=5.0)
                while True:
                    batch = self.remote.poll()
                    if batch:
                        with collected_lock:
                            collected.extend(ref.sample_id for ref in batch)
                        continue
                    if self.remote.is_closed():
                        tail = self.remote.poll()
                        if tail:
                            with collected_lock:
                                collected.extend(ref.sample_id for ref in tail)
                        return
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
        for thread in threads:
            thread.start()
        for start in range(0, total, 4):
            self.local.publish_batch([_ref(f"s{i}") for i in range(start, start + 4)])
        self.local.close()
        for thread in threads:
            thread.join(timeout=5.0)
            self.assertFalse(thread.is_alive())

        self.assertEqual(errors, [])
        self.assertEqual(len(collected), total)
        self.assertEqual(sorted(collected), sorted(f"s{i}" for i in range(total)))

    def test_pull_retries_transient_gateway_errors_and_recovers(self):
        from urllib.error import HTTPError
        from urllib.request import urlopen as real_urlopen

        self.local.publish(_ref("s0"))
        attempts = {"n": 0}

        def flaky_urlopen(request, timeout=None):
            attempts["n"] += 1
            if attempts["n"] <= 2:
                raise HTTPError(request.full_url, 503, "Service Unavailable", {}, None)
            return real_urlopen(request, timeout=timeout)

        with (
            mock.patch(
                "specforge.runtime.data_plane.http_inbox.urlopen",
                side_effect=flaky_urlopen,
            ),
            mock.patch("specforge.runtime.data_plane.http_inbox.time.sleep"),
        ):
            got = self.remote.poll()
        self.assertEqual([ref.sample_id for ref in got], ["s0"])
        self.assertEqual(attempts["n"], 3)

        # Exhausted transient retries still raise HTTPError loudly.
        err_504 = HTTPError(self.remote.path, 504, "Gateway Timeout", {}, None)
        with (
            mock.patch(
                "specforge.runtime.data_plane.http_inbox.urlopen", side_effect=err_504
            ),
            mock.patch("specforge.runtime.data_plane.http_inbox.time.sleep"),
            self.assertLogs("specforge.runtime.data_plane.http_inbox", level="ERROR"),
            self.assertRaises(HTTPError) as ctx,
        ):
            self.remote.poll()
        self.assertEqual(ctx.exception.code, 504)

    def test_closed_inbox_rejects_trailing_incomplete_utf8_and_partial_jsonl(self):
        from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefQueue

        for case, payload in (
            ("mid_utf8", "样本".encode("utf-8")[:2]),
            ("partial_jsonl", b'{"sample_id": "s0"'),
        ):
            with self.subTest(case=case):
                work = tempfile.mkdtemp(prefix=f"http-inbox-{case}-")
                path = RefDistributor.inbox_path(work, 1)
                local = StreamingRefChannel(path)
                origin = f"http://127.0.0.1:{_free_port()}"
                server = InboxHTTPServer(work, 2, origin, bind_host="127.0.0.1").start()
                try:
                    remote = RemoteInboxChannel(origin, 1)
                    with open(path, "wb") as stream:
                        stream.write(payload)
                    # Open inbox buffers incomplete bytes without raising.
                    self.assertEqual(remote.poll(), [])
                    local.close()
                    queue = StreamingRefQueue(remote, poll_s=0.0)
                    with self.assertRaisesRegex(ValueError, "closed inbox"):
                        queue.get(1)
                finally:
                    server.stop()


if __name__ == "__main__":
    unittest.main()
