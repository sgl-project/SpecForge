#!/usr/bin/env python3
"""Run a finite DSpark capture, RDMA readback, and durable cleanup pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specforge-root", required=True)
    parser.add_argument("--server-urls", required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--local-hostname", required=True)
    parser.add_argument("--rdma-devices", required=True)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--prefetch-batches", type=int, default=0)
    parser.add_argument("--ingest-batch-size", type=int, default=32)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.specforge_root).resolve()))
    from datasets import Dataset

    from specforge.algorithms.builtin import builtin_algorithm_registry
    from specforge.data.prompt_builder import _ProcessedPromptSequence
    from specforge.inference.adapters.server_capture import (
        SGLangServerCaptureAdapter,
        ServerCaptureSchema,
    )
    from specforge.launch import build_disagg_online_producer
    from specforge.runtime.data_plane.gpu_direct_store import (
        MooncakeGpuDirectFeatureStore,
    )
    from specforge.runtime.data_plane.streaming_ref_channel import (
        StreamingRefChannel,
    )

    if args.samples < 1:
        raise ValueError("samples must be positive")
    dataset = Dataset.from_file(args.dataset_file)
    if args.samples > len(dataset):
        raise ValueError(f"requested {args.samples} rows from {len(dataset)}")
    prompts = _ProcessedPromptSequence(
        dataset.select(range(args.samples)),
        max_length=8192,
        min_loss_tokens=0,
        loss_mask_filter=None,
    )
    algorithm = builtin_algorithm_registry().resolve("dspark")
    layout = algorithm.providers.server_streaming_for("text").layout
    run_id = f"q38-live-capture-{uuid.uuid4().hex[:10]}"
    store = MooncakeGpuDirectFeatureStore(
        store_id=run_id,
        local_hostname=args.local_hostname,
        transport="rdma",
        rdma_devices=args.rdma_devices,
        retain_on_release=True,
    )
    schema = ServerCaptureSchema(
        aux_feature=layout.aux_feature,
        last_hidden_feature=layout.last_hidden_feature,
        passthrough=layout.passthrough,
        attention_mask_feature=layout.attention_mask_feature,
    )
    server_urls = [url.strip().rstrip("/") for url in args.server_urls.split(",")]
    server_urls = [url for url in server_urls if url]
    adapters = [
        SGLangServerCaptureAdapter(
            url,
            store,
            run_id=run_id,
            algorithm=algorithm.name,
            schema=schema,
            timeout_s=300,
        )
        for url in server_urls
    ]
    workdir = tempfile.mkdtemp(prefix="q38-live-capture-")
    channel = StreamingRefChannel(os.path.join(workdir, "refs.jsonl"))
    channel.publish_consumer_quantum(1)
    optional = {}
    if args.prefetch_batches:
        optional.update(
            producer_prompt_prefetch_batches=args.prefetch_batches,
            producer_reorder_buffer=args.concurrency,
        )
    _workers, drive = build_disagg_online_producer(
        algorithm=algorithm,
        feature_source=adapters,
        prompts=prompts,
        feature_store=store,
        channel=channel,
        run_id=run_id,
        target_hidden_size=5120,
        target_repr=None,
        aux_hidden_state_layer_ids=[5, 19, 33, 47, 61],
        lease=1,
        producer_concurrency=args.concurrency,
        producer_ordered_publish=True,
        prompt_ingest_batch_size=args.ingest_batch_size,
        in_flight_high_watermark=args.samples + 32,
        in_flight_low_watermark=args.samples + 16,
        backpressure_poll_s=0.01,
        peer_wait_timeout_s=300,
        prompt_epochs=1,
        prompt_seed=42,
        **optional,
    )

    capture_started = time.perf_counter()
    produced = drive()
    capture_elapsed = time.perf_counter() - capture_started
    refs = channel.poll()
    if produced != args.samples or len(refs) != args.samples:
        raise RuntimeError(
            f"capture count mismatch produced={produced} refs={len(refs)}"
        )

    materialized_bytes = 0
    materialize_elapsed = 0.0
    if args.materialize:
        torch = __import__("torch")
        materialize_started = time.perf_counter()
        for ref in refs:
            tensors, handle = store.get(ref, device="cuda:0")
            materialized_bytes += sum(
                tensor.numel() * tensor.element_size()
                for tensor in tensors.values()
            )
            store.release(handle, reason="live-pipeline-readback")
            del tensors
        torch.cuda.synchronize()
        materialize_elapsed = time.perf_counter() - materialize_started

    cleanup_started = time.perf_counter()
    sample_ids = [ref.sample_id for ref in refs]
    abort_many = getattr(store, "abort_many", None)
    if args.prefetch_batches and callable(abort_many):
        removed = abort_many(sample_ids, reason="live-pipeline-durable-ack")
    else:
        for sample_id in sample_ids:
            store.abort(sample_id, reason="live-pipeline-durable-ack")
        removed = len(sample_ids)
    store.drain_pending_removals(max_attempts=8, retry_interval_s=0.25)
    cleanup_elapsed = time.perf_counter() - cleanup_started
    health = store.health()
    print(
        json.dumps(
            {
                "capture_elapsed_s": capture_elapsed,
                "capture_samples_per_s": produced / capture_elapsed,
                "cleanup_elapsed_s": cleanup_elapsed,
                "consumer_health": health,
                "id_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
                "materialize_bytes": materialized_bytes,
                "materialize_elapsed_s": materialize_elapsed,
                "prefetch_batches": args.prefetch_batches,
                "produced": produced,
                "removed": removed,
                "run_id": run_id,
                "server_urls": server_urls,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
