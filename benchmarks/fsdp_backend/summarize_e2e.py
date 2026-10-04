"""Markdown table for e2e_offline.py results: one column per label, Δ vs fsdp2."""

import glob
import json
import os
import sys

from summarize import LABEL_ORDER, cell, fmt  # noqa: F401


def main(out_dir):
    runs = {}
    for path in glob.glob(os.path.join(out_dir, "*-rank*.json")):
        r = json.load(open(path))
        runs.setdefault((r["algo"], r["label"]), []).append(r)
    lines = []
    for algo in sorted({a for a, _ in runs}):
        labels = [l for l in LABEL_ORDER if (algo, l) in runs] + sorted(l for (a, l) in runs if a == algo and l not in LABEL_ORDER)
        cols = {l: runs[(algo, l)] for l in labels}
        for l in list(labels):
            bad = [r for r in cols[l] if not r.get("ok")]
            if bad:
                lines.append(f"\n**{algo} / {l}**: failed: {bad[0].get('error')}\n")
                labels.remove(l)
        if not labels:
            continue
        base = "fsdp2" if "fsdp2" in labels else labels[0]
        ref = cols[labels[0]][0]
        a = ref["args"]
        lines.append(
            f"\n**{algo} e2e** — {ref['world_size']}×{ref['device_name']}, torch {ref['torch']}, "
            f"offline runtime, {a['samples']} samples × {a['num_epochs']} epochs, seq {a['seq_len']}, "
            f"batch {a['batch']} × accum {a['accum']}, {a['num_workers']} loader workers, log every {a['log_interval']} steps; "
            f"Δ vs `{base}`\n"
        )
        lines.append("| metric | " + " | ".join(labels) + " |")
        lines.append("|---|" + "---|" * len(labels))

        def row(label, key, fn=max, nd=2):
            vals = {}
            for l in labels:
                v = [r[key] for r in cols[l] if r.get(key) is not None]
                vals[l] = fn(v) if v else None
            if all(v is None for v in vals.values()):
                return
            b = vals.get(base)
            lines.append(f"| {label} | " + " | ".join(cell(vals[l], b, nd) for l in labels) + " |")

        row("optimizer steps run", "steps", fn=max, nd=0)
        row("steady-state step time (s)", "steady_step_s", fn=max, nd=3)
        row("steady-state samples/s (all GPUs)", "steady_samples_per_s_total", fn=min, nd=2)
        row("steady-state samples/s per GPU", "steady_samples_per_s_per_gpu", fn=min, nd=3)
        row("fit wall time incl. warm-up + final checkpoint (s)", "fit_wall_s", fn=max, nd=1)
        row("e2e samples/s over whole fit (all GPUs)", "e2e_samples_per_s_total", fn=min, nd=2)
        row("build + wrap (s)", "build_and_wrap_s", fn=max, nd=1)
        row("peak allocated (MB, max rank)", "peak_alloc_mb", fn=max, nd=0)
        row("peak reserved (MB, max rank)", "peak_reserved_mb", fn=max, nd=0)
    text = "\n".join(lines)
    print(text)
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
