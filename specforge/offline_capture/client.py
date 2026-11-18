"""Offline preparation through the SGLang plugin and temporary Mooncake features."""

from __future__ import annotations

import os
import uuid
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager

from specforge.inference.adapters.server_capture import (
    ServerCaptureFailure,
    ServerCaptureSchema,
    SGLangServerCaptureAdapter,
    detect_server_capture,
)
from specforge.inference.capture import CaptureConfig
from specforge.runtime.contracts import PromptTask


def _server_info(url, timeout):
    import requests

    response = requests.get(f"{url}/server_info", timeout=timeout)
    response.raise_for_status()
    return response.json()


class OfflineCaptureClient:
    """A bounded-batch CPU client; independent workers shard dataset samples."""

    def __init__(
        self,
        server_urls,
        *,
        target_model_path,
        capture_layers,
        layout,
        target_hidden_size,
        timeout_s=300.0,
        store=None,
        server_info_fn=None,
        post_fn=None,
    ):
        self.urls = tuple(url.rstrip("/") for url in server_urls)
        if not self.urls or any(not url for url in self.urls):
            raise ValueError("offline capture requires at least one server URL")
        if timeout_s <= 0:
            raise ValueError("capture timeout must be positive")
        self.run_id = f"offline-{uuid.uuid4().hex}"
        self.layout = layout
        self.layers = tuple(capture_layers)
        info_fn = server_info_fn or _server_info
        # Validate every endpoint before opening a store or publishing requests.
        for url in self.urls:
            info = info_fn(url, timeout_s)
            detect_server_capture(info, server_url=url)
            if info.get("model_path") != target_model_path:
                raise ValueError(
                    f"{url}: model_path={info.get('model_path')!r}, "
                    f"expected {target_model_path!r}"
                )
            for state in info.get("internal_states") or []:
                if (
                    not isinstance(state, Mapping)
                    or state.get("spec_capture_backend") != "plugin"
                    or state.get("spec_capture_method") != layout.capture_method
                    or tuple(state.get("spec_capture_aux_layer_ids") or ())
                    != self.layers
                ):
                    raise ValueError(
                        f"{url}: capture contract differs from "
                        f"{layout.capture_method}/{self.layers}; configure the server "
                        "accordingly and install the current capture plugin"
                    )
        aux = "aux_hidden_states" if layout.aux_feature is not None else None
        last = "last_hidden_states" if layout.last_hidden_feature is not None else None
        self.contract = CaptureConfig.from_strategy(
            required_features={name for name in (aux, last) if name is not None},
            aux_hidden_state_layer_ids=self.layers,
            target_repr="hidden_state" if last else None,
            target_hidden_size=target_hidden_size,
        )
        self.store = store if store is not None else self._connect_store()
        self.adapters = [
            SGLangServerCaptureAdapter(
                url,
                self.store,
                run_id=self.run_id,
                algorithm=layout.capture_method,
                schema=ServerCaptureSchema(aux, last, ()),
                timeout_s=timeout_s,
                target_model_version=target_model_path,
                server_capture="plugin",
                post_fn=post_fn,
            )
            for url in self.urls
        ]
        self._next_server = 0
        self._closed = False

    def _connect_store(self):
        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

        protocol = os.environ.get("MOONCAKE_PROTOCOL", "tcp")
        if protocol not in ("tcp", "rdma"):
            raise ValueError("offline capture requires MOONCAKE_PROTOCOL=tcp or rdma")
        required = ("MOONCAKE_METADATA_SERVER", "MOONCAKE_MASTER_SERVER_ADDR")
        for key in required:
            if not os.environ.get(key):
                raise ValueError(f"offline capture requires {key}")
        return MooncakeFeatureStore(
            store_id=self.run_id,
            setup_kwargs={
                "local_hostname": os.environ.get(
                    "MOONCAKE_LOCAL_HOSTNAME", "127.0.0.1"
                ),
                "metadata_server": os.environ[required[0]],
                "master_server_addr": os.environ[required[1]],
                "protocol": protocol,
                "rdma_devices": os.environ.get("MOONCAKE_RDMA_DEVICES", ""),
                # Servers contribute storage; these clients only read features.
                "global_segment_size": 0,
                "local_buffer_size": 0,
            },
            receive_buffers="pageable",
            retain_on_release=False,
        )

    @contextmanager
    def capture_rows(self, input_ids, *, sample_ids):
        """Hold captured CPU tensors and store objects until files are written."""
        if self._closed:
            raise RuntimeError("offline capture client is closed")
        if len(input_ids) != len(sample_ids) or len(set(sample_ids)) != len(sample_ids):
            raise ValueError("capture rows require distinct, aligned sample IDs")
        if not input_ids:
            yield []
            return
        if any(not row for row in input_ids):
            raise ValueError("capture rows must contain at least one token")
        # Keep retry attempts isolated; the caller's file IDs remain stable.
        attempt = uuid.uuid4().hex
        tasks = [
            PromptTask(
                task_id=f"{sample_id}-{attempt}",
                run_id=self.run_id,
                source_id="offline-preparation",
                payload={"input_ids": list(row)},
                max_length=len(row),
            )
            for sample_id, row in zip(sample_ids, input_ids)
        ]
        adapter = self.adapters[self._next_server % len(self.adapters)]
        self._next_server += 1
        with ExitStack() as cleanup:
            # Register cleanup before HTTP to cover lost responses and partial
            # adoption when validating a batched response raises midway.
            cleanup.callback(self.store.discard_external_attempts)
            for task in tasks:
                cleanup.callback(
                    self.store.abort,
                    f"{self.run_id}:{task.task_id}",
                    reason="offline-finished",
                )
            refs = adapter.produce_refs(tasks, capture=self.contract)
            failures = [r for r in refs if isinstance(r, ServerCaptureFailure)]
            if failures:
                raise RuntimeError(f"offline server capture failed: {failures}")
            rows = []
            for ref in refs:
                tensors, handle = self.store.get(ref, device="cpu")
                cleanup.callback(self.store.release, handle, reason="offline-finished")
                rows.append(tensors)
            yield rows
        self.store.gc()

    def close(self):
        if self._closed:
            return
        try:
            self.store.discard_external_attempts()
            self.store.drain_pending_removals()
        finally:
            self.store.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
