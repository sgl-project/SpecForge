#!/usr/bin/env python3
"""Load / dequantize the official DeepSeek-V4 DSpark ``mtp.*`` weights.

The official checkpoints store the drafter quantized: attention, shared
expert and main_proj linears as FP8 E4M3 with 128x128-block ue8m0 scales,
routed experts as packed FP4 e2m1 with one ue8m0 scale per 32 inputs. The dequantizers live with the DeepSeek MoE
family (``specforge.modeling.draft.moe.families.deepseek``) and follow the
reference ``inference/convert.py``; this module produces the official-named
state dict ``DSparkV4DraftModel`` loads.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, Optional, Tuple

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from specforge.modeling.draft.moe.families.deepseek import (  # noqa: E402
    dequantize_tensors,
    iter_checkpoint_tensors,
)

# The mtp.* drafter bundled with deepseek-ai/DeepSeek-V4-Flash-0731. NOTE:
# the standalone DeepSeek-V4-Flash-DSpark repo is a DIFFERENT model release
# (both its target shards and its drafter weights differ from 0731, despite a
# byte-identical config.json) — do not mix the two.
DEFAULT_SNAPSHOT = os.path.join(
    os.environ.get("HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")),
    "models--deepseek-ai--DeepSeek-V4-Flash-0731",
    "snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062",
)


def resolve_snapshot(snapshot_dir: Optional[str] = None) -> str:
    return snapshot_dir or os.environ.get("DSPARK_OFFICIAL_SNAPSHOT", DEFAULT_SNAPSHOT)


def iter_official_tensors(
    snapshot_dir: str, prefix: str = "mtp."
) -> Iterable[Tuple[str, torch.Tensor]]:
    return iter_checkpoint_tensors(snapshot_dir, prefix)


def load_official_mtp_state(
    snapshot_dir: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, torch.Tensor]:
    """Dequantized ``mtp.*`` state dict in the official tensor naming.

    Quantized and bf16 tensors come back in ``dtype``; fp32 tensors (selection
    biases, hyper-connection parameters, attention sinks) stay fp32.
    """
    snapshot_dir = resolve_snapshot(snapshot_dir)
    raw = dict(iter_official_tensors(snapshot_dir))
    return dequantize_tensors(raw, dtype)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    state = load_official_mtp_state(args.snapshot)
    total = sum(v.numel() for v in state.values())
    print(f"{len(state)} tensors, {total/1e9:.2f}B params")
    if args.summary:
        for name in sorted(state)[:40]:
            print(name, tuple(state[name].shape), state[name].dtype)
