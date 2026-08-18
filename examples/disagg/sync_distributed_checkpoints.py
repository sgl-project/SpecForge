#!/usr/bin/env python3
"""Mirror rank-local SpecForge checkpoints between two trainer nodes.

SpecForge writes ``training_state.pt`` on rank 0 and one
``training_state_rankN.pt`` file per rank. On platforms without a shared output
filesystem, each trainer node therefore owns only part of a multi-node
checkpoint. Run one relay on each node to package the files written locally,
serve them over the private trainer network, fetch the peer package, and
atomically assemble a complete checkpoint directory on both nodes.

The HTTP server intentionally has no authentication or TLS. Bind it only to a
trusted private network interface and do not expose its port publicly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import tarfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

CHUNK_BYTES = 8 * 1024 * 1024
STATE_FILE = "training_state.pt"


class _PrivateHTTPServer(ThreadingHTTPServer):
    request_queue_size = 64
    daemon_threads = True


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, _format, *_args) -> None:
        return


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _atomic_text(path: Path, value: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def _rank_range(value: str) -> tuple[int, ...]:
    match = re.fullmatch(r"(\d+)-(\d+)", value)
    if match is None:
        raise argparse.ArgumentTypeError("rank range must look like 0-7")
    first, last = (int(item) for item in match.groups())
    if first < 0 or last < first:
        raise argparse.ArgumentTypeError("rank range must be increasing")
    return tuple(range(first, last + 1))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


class CheckpointRelay:
    def __init__(self, args: argparse.Namespace) -> None:
        self.run_root = Path(args.run_root).resolve()
        self.output_dir = self.run_root / "output"
        self.relay_dir = self.run_root / "checkpoint-relay"
        self.run_id = args.run_id
        self.local_ranks = tuple(args.local_ranks)
        self.peer_ranks = tuple(args.peer_ranks)
        self.peer_url = args.peer_url.rstrip("/")
        self.poll_s = float(args.poll_s)
        self.max_archives = int(args.max_archives)
        self.relay_dir.mkdir(parents=True, exist_ok=True)
        self._stop = threading.Event()
        handler = partial(_QuietHandler, directory=str(self.relay_dir))
        self._httpd = _PrivateHTTPServer(
            (args.serve_host, int(args.serve_port)), handler
        )
        self._server_thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="checkpoint-relay-http",
            daemon=True,
        )
        self._step_pattern = re.compile(rf"^{re.escape(self.run_id)}-step(\d+)$")
        self._local_archive_pattern = self._archive_pattern(self.local_ranks)
        self._peer_archive_pattern = self._archive_pattern(self.peer_ranks)

    def _archive_pattern(self, ranks: tuple[int, ...]) -> re.Pattern[str]:
        return re.compile(
            rf"^{re.escape(self.run_id)}-step(\d+)-"
            rf"ranks{ranks[0]}-{ranks[-1]}\.tar$"
        )

    def _local_names(self) -> tuple[str, ...]:
        names = [f"training_state_rank{rank}.pt" for rank in self.local_ranks]
        if 0 in self.local_ranks:
            names.insert(0, STATE_FILE)
        return tuple(names)

    def _peer_names(self) -> tuple[str, ...]:
        names = [f"training_state_rank{rank}.pt" for rank in self.peer_ranks]
        if 0 in self.peer_ranks:
            names.insert(0, STATE_FILE)
        return tuple(names)

    def _checkpoint_dirs(self) -> list[tuple[int, Path]]:
        found = []
        try:
            children = list(self.output_dir.iterdir())
        except FileNotFoundError:
            return []
        for child in children:
            match = self._step_pattern.fullmatch(child.name)
            if match is not None and child.is_dir():
                found.append((int(match.group(1)), child))
        return sorted(found)

    def _archive_name(self, step: int) -> str:
        return (
            f"{self.run_id}-step{step}-"
            f"ranks{self.local_ranks[0]}-{self.local_ranks[-1]}.tar"
        )

    def _local_archives(self) -> list[tuple[int, Path]]:
        found = []
        for path in self.relay_dir.iterdir():
            match = self._local_archive_pattern.fullmatch(path.name)
            if match is not None and path.is_file():
                found.append((int(match.group(1)), path))
        return sorted(found)

    def _prune_local_archives(self, keep: set[str]) -> None:
        for path in self.relay_dir.iterdir():
            if self._local_archive_pattern.fullmatch(path.name) is None:
                continue
            if path.name in keep:
                continue
            path.unlink(missing_ok=True)
            path.with_name(path.name + ".sha256").unlink(missing_ok=True)

    def _prune_peer_archives(self, keep: set[str]) -> None:
        for path in self.relay_dir.iterdir():
            if not path.name.startswith("peer-"):
                continue
            archive_name = path.name.removeprefix("peer-")
            if archive_name.endswith(".partial"):
                archive_name = archive_name.removesuffix(".partial")
            if self._peer_archive_pattern.fullmatch(archive_name) is None:
                continue
            if archive_name not in keep:
                path.unlink(missing_ok=True)

    def _publish_local(self) -> None:
        local_names = self._local_names()
        checkpoint_dirs = self._checkpoint_dirs()[-self.max_archives :]
        for step, checkpoint_dir in checkpoint_dirs:
            sources = [checkpoint_dir / name for name in local_names]
            if not all(path.is_file() and path.stat().st_size > 0 for path in sources):
                continue
            archive = self.relay_dir / self._archive_name(step)
            sha_path = archive.with_name(archive.name + ".sha256")
            if not archive.is_file():
                tmp = archive.with_name(archive.name + ".tmp")
                with tarfile.open(tmp, mode="w") as bundle:
                    for source in sources:
                        bundle.add(source, arcname=source.name, recursive=False)
                os.replace(tmp, archive)
                _atomic_text(sha_path, _sha256(archive))
                print(
                    f"PACKED step={step} bytes={archive.stat().st_size} "
                    f"archive={archive.name}",
                    flush=True,
                )
        archives = self._local_archives()[-self.max_archives :]
        self._prune_local_archives({archive.name for _, archive in archives})
        entries = []
        for step, archive in archives:
            sha_path = archive.with_name(archive.name + ".sha256")
            try:
                archive_sha = sha_path.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                archive_sha = _sha256(archive)
                _atomic_text(sha_path, archive_sha)
            entries.append(
                {
                    "step": step,
                    "archive": archive.name,
                    "sha256": archive_sha,
                    "files": list(local_names),
                }
            )
        _atomic_json(
            self.relay_dir / "manifest.json",
            {"run_id": self.run_id, "entries": entries},
        )

    def _peer_manifest(self) -> dict | None:
        try:
            with urlopen(f"{self.peer_url}/manifest.json", timeout=5.0) as response:
                payload = json.load(response)
        except (HTTPError, URLError, OSError, TimeoutError, ValueError):
            return None
        if payload.get("run_id") != self.run_id:
            return None
        return payload

    def _download(self, name: str, expected_sha: str) -> Path:
        match = self._peer_archive_pattern.fullmatch(name)
        if match is None or Path(name).name != name:
            raise ValueError(f"unexpected peer archive name {name!r}")
        destination = self.relay_dir / f"peer-{name}"
        if destination.is_file() and _sha256(destination) == expected_sha:
            return destination
        partial_path = destination.with_name(destination.name + ".partial")
        digest = hashlib.sha256()
        with (
            urlopen(f"{self.peer_url}/{name}", timeout=300.0) as response,
            partial_path.open("wb") as stream,
        ):
            while chunk := response.read(CHUNK_BYTES):
                stream.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected_sha:
            partial_path.unlink(missing_ok=True)
            raise ValueError(f"SHA-256 mismatch for peer archive {name}")
        os.replace(partial_path, destination)
        return destination

    def _install_peer_archive(self, entry: dict) -> None:
        step = int(entry["step"])
        archive_name = str(entry["archive"])
        archive_match = self._peer_archive_pattern.fullmatch(archive_name)
        if archive_match is None or int(archive_match.group(1)) != step:
            raise ValueError(
                f"unexpected peer archive for step {step}: {archive_name!r}"
            )
        expected_names = set(self._peer_names())
        if set(entry.get("files", ())) != expected_names:
            raise ValueError(f"unexpected peer file set at step {step}")
        checkpoint_dir = self.output_dir / f"{self.run_id}-step{step}"
        if not checkpoint_dir.is_dir():
            return
        marker = checkpoint_dir / (
            f".checkpoint-relay-ranks{self.peer_ranks[0]}-"
            f"{self.peer_ranks[-1]}.json"
        )
        expected_sha = str(entry["sha256"])
        try:
            with marker.open(encoding="utf-8") as stream:
                if json.load(stream).get("sha256") == expected_sha:
                    return
        except (FileNotFoundError, OSError, ValueError):
            pass
        archive = self._download(archive_name, expected_sha)
        with tarfile.open(archive, mode="r") as bundle:
            members = bundle.getmembers()
            names = {member.name for member in members}
            if names != expected_names or any(
                not member.isfile() for member in members
            ):
                raise ValueError(f"unsafe or incomplete peer archive {archive.name}")
            for member in members:
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read {member.name} from {archive.name}")
                destination = checkpoint_dir / member.name
                tmp = destination.with_name(destination.name + ".relay-tmp")
                with source, tmp.open("wb") as stream:
                    while chunk := source.read(CHUNK_BYTES):
                        stream.write(chunk)
                os.replace(tmp, destination)
        _atomic_json(
            marker,
            {"sha256": expected_sha, "archive": entry["archive"], "step": step},
        )
        print(
            f"INSTALLED step={step} peer_files={len(expected_names)} "
            f"sha256={expected_sha}",
            flush=True,
        )

    def _pull_peer(self) -> None:
        manifest = self._peer_manifest()
        if manifest is None:
            return
        entries = sorted(
            manifest.get("entries", ()), key=lambda item: int(item["step"])
        )[-self.max_archives :]
        self._prune_peer_archives({str(entry["archive"]) for entry in entries})
        for entry in entries:
            self._install_peer_archive(entry)

    def stop(self, *_args) -> None:
        self._stop.set()

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        self._server_thread.start()
        print(
            f"STARTED run={self.run_id} local_ranks={self.local_ranks[0]}-"
            f"{self.local_ranks[-1]} peer={self.peer_url} "
            f"max_archives={self.max_archives}",
            flush=True,
        )
        try:
            while not self._stop.is_set():
                try:
                    self._publish_local()
                    self._pull_peer()
                except Exception as exc:  # noqa: BLE001 - retry loop boundary
                    print(f"RETRY {type(exc).__name__}: {exc}", flush=True)
                self._stop.wait(self.poll_s)
        finally:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._server_thread.join(timeout=5.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--local-ranks", required=True, type=_rank_range)
    parser.add_argument("--peer-ranks", required=True, type=_rank_range)
    parser.add_argument("--serve-host", required=True)
    parser.add_argument("--serve-port", required=True, type=int)
    parser.add_argument("--peer-url", required=True)
    parser.add_argument("--poll-s", type=float, default=15.0)
    parser.add_argument("--max-archives", type=_positive_int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    CheckpointRelay(_parse_args()).run()
