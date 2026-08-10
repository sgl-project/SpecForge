# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Producer-hosted HTTP intake for online-live capture records.

Live capture servers write feature tensors straight into Mooncake and push
only tensor-free capture records here.  The endpoint pair is the whole live
protocol: ``GET /v1/spec-capture/config`` hands the server its capture
document (store id, feature names, passthrough synthesis rules, token cap) and
``POST /v1/spec-capture/records`` submits one record per captured request.

Responses drive the server's cleanup contract: any non-2xx status tells the
sink to remove the keys it just wrote.  A duplicate ``sample_id`` (a sink
retry after a lost response) is acknowledged with 200 without re-dispatching.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, Tuple
from urllib.parse import urlparse

_CONFIG_PATH = "/v1/spec-capture/config"
_RECORDS_PATH = "/v1/spec-capture/records"
_MAX_RECORD_BYTES = 1 << 20

_STATUS_CODES = {
    "accepted": HTTPStatus.OK,
    "shed": HTTPStatus.TOO_MANY_REQUESTS,
    "rejected": HTTPStatus.UNPROCESSABLE_ENTITY,
}


class _IntakeThreadingHTTPServer(ThreadingHTTPServer):
    request_queue_size = 256
    daemon_threads = True


class CaptureIntakeServer:
    """Serve the live capture config and accept pushed capture records.

    ``on_record`` is called with each fresh record under one server-wide lock
    and returns ``(status, detail)`` with status in ``accepted``/``shed``/
    ``rejected``.  Only accepted sample ids enter the dedup window.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        config_payload: Dict,
        on_record: Callable[[Dict], Tuple[str, str]],
        dedup_capacity: int = 65536,
    ) -> None:
        self.config_payload = dict(config_payload)
        self.on_record = on_record
        self._dedup: OrderedDict[str, None] = OrderedDict()
        self._dedup_capacity = int(dedup_capacity)
        self._lock = threading.Lock()
        self._stats = {"accepted": 0, "shed": 0, "rejected": 0, "duplicate": 0}
        self._httpd = _IntakeThreadingHTTPServer((host, port), self._handler_type())
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def _dispatch(self, record: Dict) -> Tuple[HTTPStatus, Dict]:
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            return HTTPStatus.BAD_REQUEST, {"error": "record requires a sample_id"}
        with self._lock:
            if sample_id in self._dedup:
                self._stats["duplicate"] += 1
                return HTTPStatus.OK, {"status": "duplicate"}
            status, detail = self.on_record(record)
            code = _STATUS_CODES.get(status)
            if code is None:
                raise RuntimeError(f"on_record returned unknown status {status!r}")
            self._stats[status] += 1
            if status == "accepted":
                self._dedup[sample_id] = None
                while len(self._dedup) > self._dedup_capacity:
                    self._dedup.popitem(last=False)
            return code, {"status": status, "detail": detail}

    def _handler_type(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "SpecForgeIntake/1"

            def log_message(self, _format, *_args):
                return

            def _json(self, status: HTTPStatus, payload) -> None:
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if urlparse(self.path).path != _CONFIG_PATH:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "unknown path"})
                    return
                self._json(HTTPStatus.OK, owner.config_payload)

            def do_POST(self):
                if urlparse(self.path).path != _RECORDS_PATH:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "unknown path"})
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length < 1 or length > _MAX_RECORD_BYTES:
                        raise ValueError("invalid body size")
                    record = json.loads(self.rfile.read(length))
                    if not isinstance(record, dict):
                        raise ValueError("record must be a JSON object")
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                record.setdefault("origin", self.client_address[0])
                try:
                    status, payload = owner._dispatch(record)
                except Exception as exc:  # noqa: BLE001 — sink must see a 5xx
                    self._json(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        {"error": f"{type(exc).__name__}: {exc}"},
                    )
                    return
                self._json(status, payload)

        return Handler

    def start(self) -> "CaptureIntakeServer":
        if self._thread is not None:
            return self
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="specforge-capture-intake",
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


__all__ = ["CaptureIntakeServer"]
