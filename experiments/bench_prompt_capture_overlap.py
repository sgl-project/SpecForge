#!/usr/bin/env python3
"""Measure prompt materialization overlap with a deterministic capture stub."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path


class _DelayedPrompts:
    def __init__(self, count: int, delay_s: float) -> None:
        self.count = count
        self.delay_s = delay_s

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int):
        if index < 0 or index >= self.count:
            raise IndexError(index)
        time.sleep(self.delay_s)
        length = 16 + index % 8
        return {
            "task_id": f"prompt-{index}",
            "payload": {
                "input_ids": list(range(1, length + 1)),
                "loss_mask": [1] * length,
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specforge-root", required=True)
    parser.add_argument("--prefetch-batches", type=int, required=True)
    parser.add_argument("--prompts", type=int, default=256)
    parser.add_argument("--ingest-batch-size", type=int, default=32)
    parser.add_argument("--materialize-delay-ms", type=float, default=1.0)
    parser.add_argument("--capture-delay-ms", type=float, default=30.0)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.specforge_root).resolve()))
    from specforge.algorithms.builtin import builtin_algorithm_registry
    from specforge.inference.adapters.server_capture import (
        SGLangServerCaptureAdapter,
        ServerCaptureSchema,
    )
    from specforge.launch import build_disagg_online_producer
    from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
    from specforge.runtime.data_plane.streaming_ref_channel import (
        StreamingRefChannel,
    )
    from tests.test_runtime.test_server_capture import (
        AUX_LAYERS,
        HIDDEN,
        _FakeMooncakeStore,
        _StubCaptureServer,
    )

    algorithm = builtin_algorithm_registry().resolve("dflash")
    layout = algorithm.providers.server_streaming_for("text").layout
    backend = _FakeMooncakeStore()
    sink = _StubCaptureServer(backend)

    def delayed_post(url, json_body, timeout):
        time.sleep(args.capture_delay_ms / 1000.0)
        return sink(url, json_body, timeout)

    store = MooncakeFeatureStore(store=backend, store_id="overlap-benchmark")
    adapter = SGLangServerCaptureAdapter(
        "http://capture-benchmark:30000",
        store,
        run_id="overlap-benchmark",
        algorithm=algorithm.name,
        schema=ServerCaptureSchema(
            aux_feature=layout.aux_feature,
            last_hidden_feature=layout.last_hidden_feature,
            passthrough=layout.passthrough,
            attention_mask_feature=layout.attention_mask_feature,
        ),
        post_fn=delayed_post,
    )
    workdir = tempfile.mkdtemp(prefix="specforge-overlap-")
    channel = StreamingRefChannel(os.path.join(workdir, "refs.jsonl"))
    channel.publish_consumer_quantum(1)
    kwargs = {}
    if args.prefetch_batches:
        kwargs.update(
            producer_prompt_prefetch_batches=args.prefetch_batches,
            producer_reorder_buffer=args.concurrency,
        )
    prompts = _DelayedPrompts(
        args.prompts, args.materialize_delay_ms / 1000.0
    )
    _workers, drive = build_disagg_online_producer(
        algorithm=algorithm,
        feature_source=adapter,
        prompts=prompts,
        feature_store=store,
        channel=channel,
        run_id="overlap-benchmark",
        target_hidden_size=HIDDEN,
        target_repr=None,
        aux_hidden_state_layer_ids=AUX_LAYERS,
        lease=8,
        producer_concurrency=args.concurrency,
        producer_ordered_publish=True,
        prompt_ingest_batch_size=args.ingest_batch_size,
        in_flight_high_watermark=args.prompts + 32,
        in_flight_low_watermark=args.prompts + 16,
        backpressure_poll_s=0.001,
        sleep=lambda delay: time.sleep(min(delay, 0.001)),
        **kwargs,
    )
    started = time.perf_counter()
    produced = drive()
    elapsed = time.perf_counter() - started
    sample_ids = [ref.sample_id for ref in channel.poll()]
    print(
        json.dumps(
            {
                "elapsed_s": elapsed,
                "id_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
                "prefetch_batches": args.prefetch_batches,
                "produced": produced,
                "refs_per_s": produced / elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
