"""End-to-end offline training benchmark through the real trainer lifecycle.

Unlike ``bench_fsdp_backends.py`` (which times ``TrainerCore`` steps on one
resident batch), this drives ``specforge.launch.build_offline_runtime`` ->
``Trainer.fit()``: the offline reader, ``FeatureDataLoader`` with workers,
``TrainerController`` (acks, logging, interval and final checkpoints) and the
backend, on production-shaped drafts and synthetic feature files on disk. It
reports steady-state throughput from the controller's own log callbacks and
the total wall time of ``fit``.

    torchrun --nproc_per_node=4 benchmarks/fsdp_backend/e2e_offline.py \
        --algo dflash2 --backend fsdp2 --label fsdp2 --data-root /workspace/e2e_data --out-dir /workspace/e2e
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
import tempfile
import time
import traceback

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_fsdp_backends import build_dflash_family, build_eagle3  # noqa: E402

ALGORITHM_NAME = {"eagle3": "eagle3", "dflash2": "dflash", "dspark": "dspark"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=["eagle3", "dflash2", "dspark"])
    p.add_argument("--backend", required=True, choices=["fsdp", "fsdp2"])
    p.add_argument("--label", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--draft-config", default=None)
    p.add_argument("--attention-backend", default="flex_attention")
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--num-epochs", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--log-interval", type=int, default=2)
    p.add_argument("--measure-last", type=int, default=24, help="steady-state window (optimizer steps)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--ttt-length", type=int, default=7)
    p.add_argument("--num-anchors", type=int, default=512)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--objective-chunk-blocks", type=int, default=128)
    p.add_argument("--teacher-metrics", action="store_true")
    p.add_argument("--compile-blocks", action="store_true")
    p.add_argument("--fp8-linear", action="store_true")
    p.add_argument("--shard-frozen-tables", action="store_true")
    p.add_argument("--checkpoint-async", action="store_true")
    p.add_argument("--save-interval", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_draft_config(algo):
    root = _repo_root()
    return {
        "eagle3": os.path.join(root, "configs", "qwen3-8b-eagle3.json"),
        "dflash2": os.path.join(root, "configs", "qwen3-8b-dflash.json"),
        "dspark": os.path.join(root, "configs", "qwen3-8b-dspark.json"),
    }[algo]


def write_features(algo, feat_dir, n, seq, shapes, rank, world):
    """Rank-sharded generation of synthetic offline feature files."""
    os.makedirs(feat_dir, exist_ok=True)
    done_marker = os.path.join(feat_dir, "DONE")
    if os.path.exists(done_marker):
        return
    g = torch.Generator().manual_seed(4321 + rank)
    for i in range(rank, n, world):
        path = os.path.join(feat_dir, f"{i:05d}.ckpt")
        if os.path.exists(path):
            continue
        input_ids = torch.randint(0, shapes["vocab"], (seq,), generator=g)
        loss_mask = torch.ones(seq, dtype=torch.long)
        if algo == "eagle3":
            sample = {
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                "hidden_state": torch.randn(1, seq, shapes["hidden"], generator=g).to(torch.bfloat16),
                "aux_hidden_state": torch.randn(1, seq, 3 * shapes["hidden"], generator=g).to(torch.bfloat16),
            }
        else:
            sample = {
                "input_ids": input_ids,
                "loss_mask": loss_mask,
                "hidden_states": torch.randn(1, seq, shapes["width"], generator=g).to(torch.bfloat16),
            }
            if algo == "dspark":
                sample["target_last_hidden_states"] = torch.randn(1, seq, shapes["hidden"], generator=g).to(torch.bfloat16)
        torch.save(sample, path + ".tmp")
        os.replace(path + ".tmp", path)
    dist.barrier()
    if rank == 0:
        open(done_marker, "w").close()
    dist.barrier()


def main():
    args = parse_args()
    if args.draft_config is None:
        args.draft_config = _default_draft_config(args.algo)
    os.environ.setdefault("SPECFORGE_DEVICE", "cuda")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from specforge.distributed import init_distributed

    init_distributed(timeout=60)
    rank, world = dist.get_rank(), dist.get_world_size()
    label = args.label or args.backend
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.algo}-{label}-rank{rank}.json")
    result = {
        "algo": args.algo, "backend": args.backend, "label": label, "world_size": world, "rank": rank,
        "args": vars(args), "torch": torch.__version__, "device_name": torch.cuda.get_device_name(local_rank),
    }
    workdir = tempfile.mkdtemp(prefix=f"e2e_{args.algo}_{label}_")
    try:
        from specforge.algorithms.builtin import builtin_algorithm_registry
        from specforge.launch import build_offline_runtime
        from specforge.optimizer import BF16Optimizer

        torch.cuda.reset_peak_memory_stats()
        t_build0 = time.perf_counter()
        if args.algo == "eagle3":
            model, head, shapes = build_eagle3(args, device, workdir, rank)
        else:
            model, shapes = build_dflash_family(args, device, args.algo)
            head = None
        feat_dir = os.path.join(args.data_root, f"{args.algo}-seq{args.seq_len}-n{args.samples}")
        write_features(args.algo, feat_dir, args.samples, args.seq_len, shapes, rank, world)

        def optimizer_factory(module):
            return BF16Optimizer(module, lr=1e-4, max_grad_norm=0.5, warmup_ratio=0.0, total_steps=10_000)

        log_points = []

        def recorder(metrics, step):
            log_points.append((time.perf_counter(), int(step)))
            if rank == 0:
                loss = metrics.get("loss") if isinstance(metrics, dict) else None
                try:
                    loss = float(loss) if loss is not None else None
                except (TypeError, ValueError):
                    loss = None
                print(f"[e2e] step {step} loss={loss}", flush=True)

        algorithm = builtin_algorithm_registry().resolve(ALGORITHM_NAME[args.algo])
        kwargs = dict(
            algorithm=algorithm,
            hidden_states_path=feat_dir,
            draft_model=model,
            target_head=head,
            optimizer_factory=optimizer_factory,
            training_backend=args.backend,
            run_id=f"e2e-{args.algo}-{label}",
            output_dir=os.path.join(args.out_dir, f"ckpt-{args.algo}-{label}"),
            ttt_length=args.ttt_length,
            max_len=args.seq_len,
            batch_size=args.batch,
            accumulation_steps=args.accum,
            num_epochs=args.num_epochs,
            max_steps=args.max_steps,
            save_interval=args.save_interval,
            logger=recorder,
            log_interval=args.log_interval,
            dataloader_num_workers=args.num_workers,
            seed=args.seed,
        )
        sig = inspect.signature(build_offline_runtime).parameters
        requested = {
            "compile_blocks": bool(args.compile_blocks),
            "fp8_linear": bool(args.fp8_linear),
            "shard_frozen_tables": bool(args.shard_frozen_tables),
        }
        if any(requested.values()):
            from specforge.training.backend import BackendOptions

            fields = BackendOptions.__dataclass_fields__
            missing = [k for k, v in requested.items() if v and k not in fields]
            if missing or "backend_options" not in sig:
                raise SystemExit(f"this checkout cannot run options {requested}: missing {missing}")
            kwargs["backend_options"] = BackendOptions(**{k: v for k, v in requested.items() if k in fields})
        if args.checkpoint_async:
            if "checkpoint_async" not in sig:
                raise SystemExit("this checkout has no checkpoint_async")
            kwargs["checkpoint_async"] = True

        trainer = build_offline_runtime(**kwargs)
        torch.cuda.synchronize()
        dist.barrier()
        t_fit0 = time.perf_counter()
        result["build_and_wrap_s"] = round(t_fit0 - t_build0, 2)
        steps = trainer.fit()
        torch.cuda.synchronize()
        t_fit1 = time.perf_counter()
        result["steps"] = int(steps)
        result["fit_wall_s"] = round(t_fit1 - t_fit0, 2)
        samples_per_step = args.batch * args.accum * world
        result["samples_total"] = int(steps) * samples_per_step
        result["e2e_samples_per_s_total"] = round(result["samples_total"] / (t_fit1 - t_fit0), 3)
        # steady state from the controller's log callbacks (last `measure_last` steps)
        pts = sorted(log_points, key=lambda x: x[1])
        result["log_points"] = [(round(t - t_fit0, 3), s) for t, s in pts]
        if len(pts) >= 2:
            last_t, last_s = pts[-1]
            # steady state = the LAST `measure_last` steps (skip compile warm-up)
            start = next((p for p in reversed(pts) if last_s - p[1] >= args.measure_last), pts[0])
            if start[1] < last_s:
                dt = last_t - start[0]
                ds = last_s - start[1]
                result["steady_window_steps"] = ds
                result["steady_step_s"] = round(dt / ds, 4)
                result["steady_samples_per_s_total"] = round(ds * samples_per_step / dt, 3)
                result["steady_samples_per_s_per_gpu"] = round(ds * samples_per_step / dt / world, 3)
        result["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        result["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 1024**2, 1)
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        if rank == 0:
            print(json.dumps({k: v for k, v in result.items() if k not in ("traceback", "args")}, indent=2))
            if not result.get("ok"):
                print(result.get("traceback"))
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
