#!/usr/bin/env python3
"""Summarize a disaggregated tuning run: trainer step timing vs producer rate.

Usage: python3 scripts/analyze_tune_run.py outputs/<run>/logs/train.log
"""
import re
import sys


def main(path: str) -> None:
    step_time, data_wait, compute = [], [], []
    published = []  # (epoch_seconds, produced_total, worker)
    backpressure = 0
    for line in open(path, errors="replace"):
        m = re.search(r"perf/optimizer_step_time_s['\"]?[:=]\s*([0-9.]+)", line)
        if m:
            step_time.append(float(m.group(1)))
        m = re.search(r"perf/data_wait_time_s['\"]?[:=]\s*([0-9.]+)", line)
        if m:
            data_wait.append(float(m.group(1)))
        m = re.search(r"perf/train_compute_time_s['\"]?[:=]\s*([0-9.]+)", line)
        if m:
            compute.append(float(m.group(1)))
        if "backpressure wait" in line:
            backpressure += 1
        m = re.search(
            r"published refs worker=(\S+) batch=\d+ produced=(\d+).*elapsed=([0-9.]+)",
            line,
        )
        if m:
            published.append((m.group(1), int(m.group(2)), float(m.group(3))))

    def stats(name, xs):
        if not xs:
            print(f"{name}: none")
            return
        xs2 = xs[max(0, len(xs) // 3):]  # drop warmup third
        print(
            f"{name}: n={len(xs)} last={xs[-1]:.2f}s "
            f"steady-mean={sum(xs2)/len(xs2):.2f}s min={min(xs):.2f} max={max(xs):.2f}"
        )

    stats("optimizer_step_time", step_time)
    stats("data_wait_time", data_wait)
    stats("train_compute_time", compute)
    print(f"producer backpressure-wait log lines: {backpressure}")
    per_worker = {}
    for w, produced, elapsed in published:
        per_worker[w] = (produced, elapsed)
    total = sum(p for p, _ in per_worker.values())
    if per_worker:
        rates = {w: p / e for w, (p, e) in per_worker.items() if e > 0}
        agg = sum(rates.values())
        print(f"produced total={total} samples, aggregate rate={agg*60:.1f} samples/min")
        for w in sorted(per_worker):
            p, e = per_worker[w]
            print(f"  worker {w}: produced={p} over {e:.0f}s -> {p/e*60:.2f}/min")
    if step_time and data_wait:
        sw = step_time[max(0, len(step_time) // 3):]
        dw = data_wait[max(0, len(data_wait) // 3):]
        ratio = (sum(dw) / len(dw)) / max(sum(sw) / len(sw), 1e-9)
        verdict = (
            "PRODUCER-BOUND (trainer starved)" if ratio > 0.15
            else "TRAINER-BOUND (producers keep up)"
        )
        print(f"steady data_wait/step_time = {ratio:.1%} -> {verdict}")


if __name__ == "__main__":
    main(sys.argv[1])
