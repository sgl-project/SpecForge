# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Private-network HTTP transport for rank-local online reference inboxes.

Online feature tensors remain in Mooncake.  This module relays only the small,
tensor-free ``SampleRef`` JSONL records that :class:`RefDistributor` already
writes.  It lets multi-node consumers run on container platforms where trainer
nodes cannot mount a shared control filesystem.

Rank 0 owns the server and its local inbox files.  Every other rank tail-reads
its private stream by byte offset and posts only its durable consumed-count
target.  Requests are idempotent with respect to a client retry: a read offset
is advanced only after the response arrives, and an absolute consumed target is
applied under a per-rank server lock.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.ref_distributor import RefDistributor
from specforge.runtime.data_plane.ref_serialization import ref_from_dict
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel

_CLOSED_SUFFIX = ".closed"
_FAILED_SUFFIX = ".failed"


class _InboxThreadingHTTPServer(ThreadingHTTPServer):
    # socketserver.TCPServer defaults to a backlog of five.  A DP consumer can
    # have dozens of ranks polling in lockstep, so that default turns a healthy
    # rank-0 relay into intermittent connection resets at every step boundary.
    request_queue_size = 256
    daemon_threads = True


def _validated_origin(origin: str):
    parsed = urlparse(origin)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.port is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "inbox HTTP origin must be http://host:port without credentials, "
            "path, query, or fragment"
        )
    return parsed


class InboxHTTPServer:
    """Serve rank-0 inbox files and consumed counters on a private origin."""

    def __init__(
        self,
        inbox_dir: str,
        dp_size: int,
        origin: str,
        *,
        bind_host: str = "0.0.0.0",
    ) -> None:
        parsed = _validated_origin(origin)
        if dp_size < 2:
            raise ValueError("inbox HTTP server requires dp_size >= 2")
        self.inbox_dir = os.path.abspath(inbox_dir)
        self.dp_size = dp_size
        self.origin = origin.rstrip("/")
        self._ack_channels = [
            StreamingRefChannel(RefDistributor.inbox_path(self.inbox_dir, rank))
            for rank in range(dp_size)
        ]
        # A target ack may be retried after an ambiguous connection reset.  The
        # per-rank lock makes the read/advance/respond sequence atomic across
        # the original request and its retry.
        self._ack_locks = [threading.Lock() for _ in range(dp_size)]
        self._httpd = _InboxThreadingHTTPServer(
            (bind_host, parsed.port), self._handler_type()
        )
        self._thread: threading.Thread | None = None

    def _handler_type(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "SpecForgeInbox/1"

            def log_message(self, _format, *_args):
                return

            def _rank(self, *, consumed: bool = False) -> int | None:
                suffix = "/consumed" if consumed else ""
                path = urlparse(self.path).path
                prefix = "/v1/inboxes/"
                if not path.startswith(prefix) or not path.endswith(suffix):
                    return None
                token = path[len(prefix) :]
                if suffix:
                    token = token[: -len(suffix)]
                try:
                    rank = int(token)
                except ValueError:
                    return None
                return rank if 0 <= rank < owner.dp_size else None

            def _json(self, status: HTTPStatus, payload) -> None:
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                rank = self._rank()
                if rank is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "unknown inbox"})
                    return
                query = parse_qs(urlparse(self.path).query)
                try:
                    offset = int(query.get("offset", ["0"])[0])
                except ValueError:
                    offset = -1
                if offset < 0:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": "invalid offset"})
                    return
                path = RefDistributor.inbox_path(owner.inbox_dir, rank)
                data = b""
                next_offset = offset
                try:
                    with open(path, "rb") as stream:
                        stream.seek(offset)
                        data = stream.read()
                        next_offset = stream.tell()
                except FileNotFoundError:
                    pass
                failure = None
                try:
                    with open(path + _FAILED_SUFFIX, encoding="utf-8") as stream:
                        failure = stream.read()
                except FileNotFoundError:
                    pass
                self._json(
                    HTTPStatus.OK,
                    {
                        "data": base64.b64encode(data).decode("ascii"),
                        "next_offset": next_offset,
                        "closed": os.path.exists(path + _CLOSED_SUFFIX),
                        "failure": failure,
                    },
                )

            def do_POST(self):
                rank = self._rank(consumed=True)
                if rank is None:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "unknown inbox"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 1 or length > 1024:
                        raise ValueError("invalid body size")
                    payload = json.loads(self.rfile.read(length))
                    target = int(payload["target"])
                    if target < 1:
                        raise ValueError("target must be positive")
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                with owner._ack_locks[rank]:
                    channel = owner._ack_channels[rank]
                    current = channel.consumed_remote()
                    if target < current:
                        self._json(
                            HTTPStatus.CONFLICT,
                            {
                                "error": "consumed target moved backwards",
                                "consumed": current,
                            },
                        )
                        return
                    if target > current:
                        channel.mark_consumed(target - current)
                    self._json(HTTPStatus.OK, {"consumed": target})

        return Handler

    def start(self) -> InboxHTTPServer:
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="specforge-inbox-http",
            daemon=True,
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5.0)
        self._thread = None


class RemoteInboxChannel:
    """Read one rank inbox and acknowledge it through ``InboxHTTPServer``."""

    def __init__(self, origin: str, dp_rank: int, *, timeout_s: float = 10.0) -> None:
        _validated_origin(origin)
        if dp_rank < 0:
            raise ValueError("dp_rank must be non-negative")
        self.path = f"{origin.rstrip('/')}/v1/inboxes/{dp_rank}"
        self.timeout_s = timeout_s
        self._read_offset = 0
        self._buf = ""
        self._closed = False
        self._failure: str | None = None
        self._pending: list[SampleRef] = []
        self._pull_lock = threading.Lock()
        self._consumed_target = 0

    def _pull(self) -> None:
        with self._pull_lock:
            request = Request(f"{self.path}?offset={self._read_offset}")
            try:
                with urlopen(request, timeout=self.timeout_s) as response:
                    payload = json.load(response)
            except (URLError, OSError, TimeoutError):
                # Let StreamingRefQueue's configured idle timeout distinguish a
                # transient network outage from a dead rank-0 service.
                return
            data = base64.b64decode(payload["data"])
            next_offset = int(payload["next_offset"])
            if (
                next_offset < self._read_offset
                or next_offset - self._read_offset != len(data)
            ):
                raise RuntimeError("inbox HTTP server returned an invalid byte range")
            self._read_offset = next_offset
            self._closed = bool(payload["closed"])
            self._failure = payload.get("failure")
            self._buf += data.decode("utf-8")
            lines = self._buf.split("\n")
            self._buf = lines.pop()
            self._pending.extend(
                ref_from_dict(json.loads(line)) for line in lines if line
            )

    def poll(self) -> list[SampleRef]:
        self._pull()
        refs, self._pending = self._pending, []
        return refs

    def is_closed(self) -> bool:
        self._pull()
        return self._closed

    def failure(self) -> str | None:
        self._pull()
        if self._failure is None:
            return None
        return f"ref-distributor died:\n{self._failure}"

    def mark_consumed(self, n: int) -> None:
        if n < 1:
            return
        target = self._consumed_target + n
        body = json.dumps({"target": target}, separators=(",", ":")).encode("utf-8")
        transient_error = None
        for attempt in range(4):
            request = Request(
                f"{self.path}/consumed",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=self.timeout_s) as response:
                    payload = json.load(response)
            except HTTPError:
                raise
            except (URLError, OSError, TimeoutError) as exc:
                transient_error = exc
                if attempt < 3:
                    time.sleep(0.05 * (attempt + 1))
                continue
            if int(payload.get("consumed", -1)) != target:
                raise RuntimeError("inbox HTTP server did not confirm consumed target")
            self._consumed_target = target
            return
        raise RuntimeError(
            "inbox HTTP consumed acknowledgement failed after retries"
        ) from transient_error


__all__ = ["InboxHTTPServer", "RemoteInboxChannel"]
