"""Experiment-only: disable TRAIN dataloader shuffling in a SpecForge tree.

The shuffled sampler draws from the global torch RNG at iteration time, so two
scripts that consume RNG differently before the first batch see different
permutations — which breaks step-aligned curve comparison. Applied identically
to BOTH trees (main and ours), so the change cancels out.

Usage: python patch_noshuffle.py <tree_root>
"""

import re
import sys

root = sys.argv[1]

for rel in ("scripts/train_eagle3.py", "scripts/train_dflash.py"):
    p = f"{root}/{rel}"
    src = open(p).read()
    # only the TRAIN loader: the first prepare_dp_dataloaders( ... shuffle=True
    m = re.search(
        r"train_dataloader = prepare_dp_dataloaders\((?:[^()]|\([^()]*\))*?shuffle=True",
        src,
        re.S,
    )
    assert m, f"{rel}: train loader shuffle anchor not found"
    patched = (
        src[: m.start()]
        + m.group(0)[: -len("shuffle=True")]
        + "shuffle=False"
        + src[m.end() :]
    )
    open(p, "w").write(patched)
    print(f"patched {rel}: train shuffle -> False")
