"""Aggregate bench_fsdp_backends.py JSON files into a markdown table."""

import glob
import json
import os
import sys


def load(out_dir):
    runs = {}
    for path in glob.glob(os.path.join(out_dir, "*-rank*.json")):
        r = json.load(open(path))
        runs.setdefault((r["algo"], r.get("label") or r["backend"]), []).append(r)
    return runs


def agg(ranks, key, fn=max):
    vals = [r[key] for r in ranks if key in r and r[key] is not None]
    return fn(vals) if vals else None


def fmt(v, nd=2):
    return "–" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main(out_dir):
    runs = load(out_dir)
    algos = sorted({a for a, _ in runs})
    lines = []
    for algo in algos:
        f1 = runs.get((algo, "fsdp"), [])
        f2 = runs.get((algo, "fsdp2"), [])
        if not f1 or not f2:
            lines.append(f"\n**{algo}**: missing runs (fsdp={len(f1)} ranks, fsdp2={len(f2)} ranks)\n")
            continue
        bad = [r for r in f1 + f2 if not r.get("ok")]
        if bad:
            lines.append(f"\n**{algo}**: {len(bad)} rank(s) failed: {bad[0].get('error')}\n")
            continue
        a0 = f1[0]["args"]
        hdr = (
            f"\n**{algo}** — {f1[0]['world_size']}×{f1[0]['device_name']}, torch {f1[0]['torch']}, "
            f"batch {a0['batch']} × accum {a0['accum']}, seq {a0['seq_len']}, "
            f"attention {a0['attention_backend']}, sharding {a0['sharding']}"
        )
        if algo == "eagle3":
            hdr += f", ttt {a0['ttt_length']}"
        else:
            hdr += f", anchors {a0['num_anchors']}, chunk_blocks {a0['objective_chunk_blocks']}"
        hdr += f"; params {f1[0]['params_total_m']}M total / {f1[0]['params_trainable_m']}M trainable"
        lines.append(hdr + "\n")
        lines.append("| metric | FSDP1 | FSDP2 | Δ |")
        lines.append("|---|---|---|---|")

        def row(label, key, fn=max, nd=2, lower_better=True, unit=""):
            v1, v2 = agg(f1, key, fn), agg(f2, key, fn)
            delta = "–"
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v1:
                pct = (v2 - v1) / v1 * 100
                delta = f"{pct:+.1f}%"
            lines.append(f"| {label} | {fmt(v1, nd)}{unit} | {fmt(v2, nd)}{unit} | {delta} |")

        row("optimizer step time (s, mean)", "optimizer_step_s_mean", fn=max, nd=3)
        row("micro-step time (ms, median)", "microstep_ms_median", fn=max, nd=1)
        row("samples/s per GPU", "samples_per_s_per_gpu", fn=min, nd=3)
        row("peak allocated (MB, max rank)", "peak_alloc_mb", fn=max, nd=0)
        row("peak reserved (MB, max rank)", "peak_reserved_mb", fn=max, nd=0)
        row("allocated after wrap (MB, max rank)", "mem_after_wrap_alloc_mb", fn=max, nd=0)
        row("allocated after build (MB)", "mem_model_built_alloc_mb", fn=max, nd=0)
        prof1 = f1[0].get("profile") or {}
        prof2 = f2[0].get("profile") or {}
        if prof1 and prof2:
            def prow(label, get):
                v1, v2 = get(prof1), get(prof2)
                d = "–"
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)) and v1:
                    d = f"{(v2 - v1) / v1 * 100:+.1f}%"
                lines.append(f"| {label} | {fmt(v1)} | {fmt(v2)} | {d} |")
            prow("host syncs per optimizer step (rank0)", lambda p: p["host_syncs_total"])
            for n in ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy"):
                prow(f"  {n}", lambda p, n=n: p["host_syncs"].get(n))
            for k in ("allgather", "reducescatter", "allreduce"):
                prow(f"NCCL {k} kernels / step", lambda p, k=k: p["nccl_kernels"][k]["count"])
                prow(f"NCCL {k} device ms / step", lambda p, k=k: p["nccl_kernels"][k]["ms"])
            prow("device kernel ms / step (rank0)", lambda p: p["device_kernel_ms_total"])
        row("loss after warmup (rank0)", "first_loss", fn=lambda v: v[0], nd=5)
        row("grad norm after warmup (rank0)", "first_grad_norm", fn=lambda v: v[0], nd=5)
    print("\n".join(lines))
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fsdp_backend_bench")
