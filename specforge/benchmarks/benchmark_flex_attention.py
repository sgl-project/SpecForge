"""Compare SDPA and Flex for the canonical EAGLE3 recurrent decoder.

Times include forward/backward through fusion, decoder and its differentiable
KV cache. Model construction and compilation warmup are outside the measurement.
"""

import argparse
import json
import time
from dataclasses import replace

import torch
import yaml

from specforge.config.draft.codec import parse_draft
from specforge.modeling import build_draft_model


def benchmark(model, *, backend, batch_size, sequence_length, depth, warmup, steps):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    spec = model.config.draft.spec_config
    ids = torch.zeros(batch_size, sequence_length, dtype=torch.long, device=device)
    features = torch.randn(
        batch_size,
        sequence_length,
        spec.target_features.input_width,
        device=device,
        dtype=dtype,
    )
    padding = torch.ones_like(ids, dtype=torch.bool)
    positions = torch.arange(sequence_length, device=device)[None].expand_as(ids)

    def step():
        model.zero_grad(set_to_none=True)
        rollout = model.new_rollout(backend)
        hidden = model.fusion(features)
        losses = []
        for _ in range(depth):
            hidden = rollout.step(ids, hidden, padding, positions)
            losses.append(hidden.float().square().mean())
        (sum(losses) / depth).backward()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    return dict(
        backend=backend,
        sequence_length=sequence_length,
        step_seconds=(time.perf_counter() - start) / steps,
        peak_memory_bytes=torch.cuda.max_memory_allocated(),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", required=True, help="Canonical EAGLE3 YAML")
    parser.add_argument(
        "--sequence-lengths", type=int, nargs="+", default=[512, 2048, 8192]
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if (
        min(args.sequence_lengths + [args.batch_size, args.depth, args.steps]) < 1
        or args.warmup < 0
    ):
        parser.error(
            "lengths, batch size, depth and steps must be positive; warmup must be non-negative"
        )
    with open(args.draft) as stream:
        config = parse_draft(yaml.safe_load(stream), target_config=None)
    if config.architecture != "Eagle3DraftModel":
        parser.error("--draft must describe EAGLE3")
    # Only recurrent decoder execution is measured; a small token table avoids
    # allocating target-vocabulary weights that this benchmark never projects.
    config = replace(
        config,
        vocab_size=32,
        draft_vocab_size=32,
        target_bindings=(),
        target_logit_transform=None,
    )
    model = build_draft_model(config, device="cuda", dtype=torch.bfloat16)
    for length in args.sequence_lengths:
        for backend in ("sdpa", "flex_attention"):
            try:
                result = benchmark(
                    model,
                    backend=backend,
                    batch_size=args.batch_size,
                    sequence_length=length,
                    depth=args.depth,
                    warmup=args.warmup,
                    steps=args.steps,
                )
            except torch.OutOfMemoryError:
                model.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                result = dict(
                    backend=backend, sequence_length=length, error="out_of_memory"
                )
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
