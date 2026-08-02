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

    def test_pull_treats_connection_reset_as_transient(self):
        with mock.patch(
            "specforge.runtime.data_plane.http_inbox.urlopen",
            side_effect=ConnectionResetError("peer reset"),
        ):
            self.assertEqual(self.remote.poll(), [])

    def test_invalid_origin_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "http://host:port"):
            RemoteInboxChannel("https://trainer.example:35900/path", 1)


if __name__ == "__main__":
    unittest.main()
