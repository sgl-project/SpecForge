#!/usr/bin/env python3
"""Measure one deterministic online prompt-materialization batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specforge-root", required=True)
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(args.specforge_root).resolve()))
    from datasets import Dataset

    from specforge.data.prompt_builder import _ProcessedPromptSequence
    from specforge.launch import _iter_epoch_online_prompt_batches

    dataset = Dataset.from_file(args.dataset_file)
    prompts = _ProcessedPromptSequence(
        dataset,
        max_length=args.max_length,
        min_loss_tokens=1,
        loss_mask_filter=None,
    )
    started = time.perf_counter()
    batch = next(
        _iter_epoch_online_prompt_batches(
            prompts,
            0,
            3,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    )
    elapsed = time.perf_counter() - started
    encoded = pickle.dumps(batch, protocol=5)
    payload_tokens = sum(len(item["payload"]["input_ids"]) for item in batch)
    loss_tokens = sum(sum(item["payload"]["loss_mask"]) for item in batch)
    print(
        json.dumps(
            {
                "batch_size": len(batch),
                "elapsed_s": elapsed,
                "payload_tokens": payload_tokens,
                "loss_tokens": loss_tokens,
                "sha256": hashlib.sha256(encoded).hexdigest(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
