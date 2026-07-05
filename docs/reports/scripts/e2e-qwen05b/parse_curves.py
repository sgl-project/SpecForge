"""Parse per-step loss curves from the experiment logs into one JSON file."""

import json
import os
import re
import sys

LOGS = sys.argv[1] if len(sys.argv) > 1 else "/root/exp/logs"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/root/exp/curves.json"

CURVE = re.compile(r"CURVE step=(\d+) loss=([0-9.eE+-]+)")

curves = {}
for fname in sorted(os.listdir(LOGS)):
    if not fname.endswith(".log"):
        continue
    name = fname[: -len(".log")]
    points = {}
    with open(os.path.join(LOGS, fname), errors="replace") as f:
        for line in f:
            m = CURVE.search(line)
            if m:
                points[int(m.group(1))] = float(m.group(2))  # last wins
    if points:
        curves[name] = sorted(points.items())

with open(OUT, "w") as f:
    json.dump(curves, f)

for name, pts in curves.items():
    steps = [s for s, _ in pts]
    losses = [v for _, v in pts]
    first = losses[0]
    last10 = losses[-10:]
    print(
        f"{name}: n={len(pts)} steps[{steps[0]}..{steps[-1]}] "
        f"first={first:.4f} mean_last10={sum(last10)/len(last10):.4f}"
    )
