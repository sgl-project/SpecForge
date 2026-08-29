#!/usr/bin/env python3
"""Measure GPU-direct release-control round trips without tensor payloads."""

from __future__ import annotations

import argparse
import json
import socketserver
import sys
import time
from pathlib import Path


class _ControlHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        while True:
            payload = self.rfile.readline(1 << 20)
            if not payload:
                return
            request = json.loads(payload)
            op = request.get("op")
            if op in {"release", "abort"}:
                released = 1
            elif op in {"release_batch", "abort_batch"}:
                released = len(request.get("items", []))
            else:
                response = {"ok": False, "error": f"unsupported operation {op!r}"}
                self.wfile.write(json.dumps(response).encode() + b"\n")
                self.wfile.flush()
                continue
            self.server.request_count += 1  # type: ignore[attr-defined]
            self.server.item_count += released  # type: ignore[attr-defined]
            self.wfile.write(
                json.dumps({"ok": True, "released": released}).encode() + b"\n"
            )
            self.wfile.flush()


class _ControlServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address: tuple[str, int]):
        self.request_count = 0
        self.item_count = 0
        super().__init__(address, _ControlHandler)


def _serve(host: str, port: int) -> None:
    with _ControlServer((host, port)) as server:
        print(json.dumps({"event": "ready", "host": host, "port": port}), flush=True)
        server.serve_forever()


def _client(
    specforge_root: str,
    endpoint: str,
    *,
    mode: str,
    batch_size: int,
    iterations: int,
) -> None:
    sys.path.insert(0, str(Path(specforge_root).resolve()))
    from specforge.runtime.data_plane.gpu_direct_store import _control_request

    items = [
        {"sample_id": f"sample-{index}", "generation": 1}
        for index in range(batch_size)
    ]
    request_count = 0
    started = time.perf_counter()
    for _ in range(iterations):
        if mode == "legacy":
            for item in items:
                _control_request(
                    endpoint,
                    {
                        "op": "release",
                        "token": "benchmark",
                        **item,
                        "reason": "benchmark",
                    },
                )
                request_count += 1
        else:
            _control_request(
                endpoint,
                {
                    "op": "release_batch",
                    "token": "benchmark",
                    "items": items,
                    "reason": "benchmark",
                },
            )
            request_count += 1
    elapsed = time.perf_counter() - started
    item_count = batch_size * iterations
    print(
        json.dumps(
            {
                "batch_size": batch_size,
                "elapsed_s": elapsed,
                "item_count": item_count,
                "items_per_s": item_count / elapsed,
                "iterations": iterations,
                "mode": mode,
                "request_count": request_count,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    server = subparsers.add_parser("server")
    server.add_argument("--host", default="0.0.0.0")
    server.add_argument("--port", type=int, default=39100)
    client = subparsers.add_parser("client")
    client.add_argument("--specforge-root", required=True)
    client.add_argument("--endpoint", required=True)
    client.add_argument("--mode", choices=("legacy", "batch"), required=True)
    client.add_argument("--batch-size", type=int, default=32)
    client.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.command == "server":
        _serve(args.host, args.port)
    else:
        _client(
            args.specforge_root,
            args.endpoint,
            mode=args.mode,
            batch_size=args.batch_size,
            iterations=args.iterations,
        )


if __name__ == "__main__":
    main()
