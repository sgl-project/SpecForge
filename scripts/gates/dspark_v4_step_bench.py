#!/usr/bin/env python3
"""Gate 0c: DSparkV4 trainer step-time / memory benchmark on synthetic data.

Runs the real training math (DSparkV4DraftModel under FSDP FULL_SHARD inside
OnlineDSparkModel with BF16Optimizer) on synthetic features, without any
capture servers. Measures per-optimizer-step wall time and peak memory so the
2-node recipe's feasibility numbers are grounded before the real run.

Launch on a free 8-GPU node:
    torchrun --standalone --nproc_per_node 8 \
        scripts/gates/dspark_v4_step_bench.py --steps 3 --accum 16
Or a quick 2-GPU sanity pass:
    torchrun --standalone --nproc_per_node 2 \
        scripts/gates/dspark_v4_step_bench.py --steps 1 --accum 2 --seq-len 2048 --num-anchors 128
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
import time

import torch
import torch.distributed as dist

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--accum", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--num-anchors", type=int, default=512)
    parser.add_argument("--chunk-blocks", type=int, default=128)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument(
        "--draft-config",
        default=os.path.join(REPO_ROOT, "configs/deepseek-v4-flash-dspark-official.json"),
    )
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(1234 + rank)

    def log(msg):
        if rank == 0:
            print(f"[bench] {msg}", flush=True)

    from specforge.modeling.auto import AutoDraftModel
    from specforge.modeling.auto import AutoDraftModelConfig
    from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
    from specforge.optimizer import BF16Optimizer

    cfg = AutoDraftModelConfig.from_file(args.draft_config)
    log("building model ...")
    draft = AutoDraftModel.from_config(cfg, torch_dtype=torch.bfloat16).to(device)
    n_params = sum(p.numel() for p in draft.parameters())
    log(f"draft params: {n_params/1e9:.2f}B")

    vocab, hidden = cfg.vocab_size, cfg.hidden_size
    lm_head = torch.nn.Linear(hidden, vocab, bias=False, dtype=torch.bfloat16).to(device)
    embed = torch.nn.Embedding(vocab, hidden, dtype=torch.bfloat16).to(device)
    for p in lm_head.parameters():
        p.requires_grad_(False)
    for p in embed.parameters():
        p.requires_grad_(False)

    online = OnlineDSparkModel(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=cfg.dflash_config["mask_token_id"],
        block_size=cfg.block_size,
        attention_backend="native",
        num_anchors=args.num_anchors,
        loss_decay_gamma=4.0,
        objective_chunk_blocks=args.chunk_blocks,
    )

    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    block_classes = {
        type(m) for m in online.modules()
        if type(m).__name__ in (draft._no_split_modules or [])
    }
    model = FSDP(
        online,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16, buffer_dtype=torch.float32
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        ignored_modules=[lm_head, embed],
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=block_classes
        ),
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        device_id=device,
    )
    optimizer = BF16Optimizer(
        model,
        lr=args.lr,
        max_grad_norm=1.0,
        total_steps=2000,
        warmup_ratio=0.02,
        lr_scheduler="constant",
    )
    log(f"FSDP wrapped ({len(block_classes)} block classes)")

    S = args.seq_len
    n_layers = len(cfg.dflash_config["target_layer_ids"])

    def synthetic_batch():
        input_ids = torch.randint(0, vocab, (1, S), device=device)
        hidden_states = (
            torch.randn(1, S, n_layers * hidden, device=device) * 2.0
        ).to(torch.bfloat16)
        last_hidden = (torch.randn(1, S, hidden, device=device)).to(torch.bfloat16)
        loss_mask = torch.ones(1, S, device=device)
        return input_ids, hidden_states, last_hidden, loss_mask

    torch.cuda.reset_peak_memory_stats(device)
    for step in range(args.steps):
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for micro in range(args.accum):
            input_ids, hidden_states, last_hidden, loss_mask = synthetic_batch()
            # Reduce-scatter every micro-step: no_sync accumulation would
            # hold ~40 GiB of unsharded grads for this drafter.
            loss, accuracy, _metrics = model(
                input_ids=input_ids,
                hidden_states=hidden_states,
                loss_mask=loss_mask,
                target_last_hidden_states=last_hidden,
            )
            (loss / args.accum).backward()
        grad_norm = optimizer.step()  # BF16Optimizer clears grads itself
        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0
        peak = torch.cuda.max_memory_allocated(device) / 2**30
        log(
            f"step {step}: {dt:.2f}s  loss={loss.item():.4f} "
            f"acc={accuracy.item():.4f} grad_norm={float(grad_norm):.3f} "
            f"peak_mem={peak:.1f}GiB"
        )
    dist.barrier()
    dist.destroy_process_group()
    log("DONE")


if __name__ == "__main__":
    main()
