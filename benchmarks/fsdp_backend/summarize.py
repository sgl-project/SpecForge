"""Aggregate bench_fsdp_backends.py JSON files into markdown tables.

One table per algorithm; one column per run label (``fsdp``, ``fsdp2``,
``fsdp2-compile``, ...). Deltas are relative to ``fsdp2`` when present,
otherwise to ``fsdp``.
"""

import glob
import json
import os
import sys

LABEL_ORDER = [
    "fsdp",
    "fsdp2",
    "fsdp2-compile",
    "fsdp2-fp8",
    "fsdp2-compile-fp8",
    "fsdp2-shardfrozen",
    "fsdp-ckpt",
    "fsdp2-ckpt",
]


def load(out_dir):
    runs = {}
    for path in glob.glob(os.path.join(out_dir, "*-rank*.json")):
        r = json.load(open(path))
        runs.setdefault((r["algo"], r.get("label") or r["backend"]), []).append(r)
    return runs


def rank0(ranks):
    return next((r for r in ranks if r.get("rank") == 0), ranks[0])


def agg(ranks, key, fn=max):
    vals = [r[key] for r in ranks if r.get(key) is not None]
    return fn(vals) if vals else None


def fmt(v, nd=2):
    if v is None:
        return "–"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def cell(v, base, nd):
    if v is None:
        return "–"
    s = fmt(v, nd)
    if isinstance(v, (int, float)) and isinstance(base, (int, float)) and base and v is not base:
        pct = (v - base) / base * 100
        if abs(pct) >= 0.05:
            s += f" ({pct:+.1f}%)"
    return s


def main(out_dir):
    runs = load(out_dir)
    algos = sorted({a for a, _ in runs})
    lines = []
    for algo in algos:
        labels = [l for l in LABEL_ORDER if (algo, l) in runs]
        labels += sorted(l for (a, l) in runs if a == algo and l not in labels)
        base_label = "fsdp2" if "fsdp2" in labels else labels[0]
        cols = {l: runs[(algo, l)] for l in labels}
        bad = {l: [r for r in rs if not r.get("ok")] for l, rs in cols.items()}
        for l, b in bad.items():
            if b:
                lines.append(f"\n**{algo} / {l}**: {len(b)} rank(s) failed: {b[0].get('error')}\n")
        labels = [l for l in labels if not bad[l]]
        if not labels:
            continue
        ref = cols[labels[0]][0]
        a0 = ref["args"]
        hdr = (
            f"\n**{algo}** — {ref['world_size']}×{ref['device_name']}, torch {ref['torch']}, "
            f"batch {a0['batch']} × accum {a0['accum']}, seq {a0['seq_len']}, "
            f"attention {a0['attention_backend']}, sharding {a0['sharding']}"
        )
        hdr += f", ttt {a0['ttt_length']}" if algo == "eagle3" else f", anchors {a0['num_anchors']}, chunk_blocks {a0['objective_chunk_blocks']}"
        hdr += f"; params {ref['params_total_m']}M total / {ref['params_trainable_m']}M trainable; Δ vs `{base_label}`"
        lines.append(hdr + "\n")
        lines.append("| metric | " + " | ".join(labels) + " |")
        lines.append("|---|" + "---|" * len(labels))

        def row(label, key, fn=max, nd=2):
            vals = {l: agg(cols[l], key, fn) for l in labels}
            base = vals.get(base_label)
            if all(v is None for v in vals.values()):
                return
            lines.append(f"| {label} | " + " | ".join(cell(vals[l], base, nd) for l in labels) + " |")

        def prow(label, get):
            vals = {}
            for l in labels:
                prof = rank0(cols[l]).get("profile") or {}
                try:
                    vals[l] = get(prof) if prof else None
                except (KeyError, TypeError):
                    vals[l] = None
            if all(v is None for v in vals.values()):
                return
            base = vals.get(base_label)
            lines.append(f"| {label} | " + " | ".join(cell(vals[l], base, 2) for l in labels) + " |")

        row("optimizer step time (s, mean)", "optimizer_step_s_mean", fn=max, nd=3)
        row("micro-step time (ms, median)", "microstep_ms_median", fn=max, nd=1)
        row("samples/s per GPU", "samples_per_s_per_gpu", fn=min, nd=3)
        row("peak allocated (MB, max rank)", "peak_alloc_mb", fn=max, nd=0)
        row("peak reserved (MB, max rank)", "peak_reserved_mb", fn=max, nd=0)
        row("allocated after wrap (MB, max rank)", "mem_after_wrap_alloc_mb", fn=max, nd=0)
        row("warm-up incl. compile (s)", "warmup_s", fn=max, nd=1)
        prow("host syncs / optimizer step (rank0)", lambda p: p["host_syncs_total"])
        prow("  cudaEventSynchronize", lambda p: p["host_syncs"].get("cudaEventSynchronize"))
        prow("  cudaStreamSynchronize", lambda p: p["host_syncs"].get("cudaStreamSynchronize"))
        prow("NCCL all-gather kernels / step", lambda p: p["nccl_kernels"]["allgather"]["count"])
        prow("NCCL reduce-scatter kernels / step", lambda p: p["nccl_kernels"]["reducescatter"]["count"])
        for variant in ("sync", "async"):
            for key, label in (
                ("state_dict_s", "state_dict gather (s)"),
                ("save_blocking_s", "save, step-loop blocking (s)"),
                ("save_total_s", "save, total until complete (s)"),
                ("bytes_on_disk_mb", "bytes written by rank0 (MB)"),
            ):
                vals = {l: ((rank0(cols[l]).get("checkpoint") or {}).get(variant) or {}).get(key) for l in labels}
                if all(v is None for v in vals.values()):
                    continue
                base = vals.get(base_label)
                lines.append(f"| checkpoint {variant}: {label} | " + " | ".join(cell(vals[l], base, 3) for l in labels) + " |")
        def r0row(label, key, nd):
            vals = {l: rank0(cols[l]).get(key) for l in labels}
            if all(v is None for v in vals.values()):
                return
            base = vals.get(base_label)
            lines.append(f"| {label} | " + " | ".join(cell(vals[l], base, nd) for l in labels) + " |")

        r0row("loss after warm-up (rank0)", "first_loss", 5)
        r0row("grad norm after warm-up (rank0)", "first_grad_norm", 5)
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fsdp_backend_bench")
