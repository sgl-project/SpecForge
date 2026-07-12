#!/usr/bin/env python3
"""Gate a Domino overfit run on its final loss, accuracy, and checkpoint."""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
from typing import Dict, Tuple


METRIC_LINE = re.compile(r"\[consumer\] step (\d+) (\{.*\})\s*$")


def final_metrics(log_path: str) -> Tuple[int, Dict[str, float]]:
    matches = []
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = METRIC_LINE.search(line)
            if match:
                metrics = ast.literal_eval(match.group(2))
                matches.append((int(match.group(1)), metrics))
    if not matches:
        raise ValueError(f"no consumer metric lines found in {log_path}")
    return matches[-1]


def checkpoint_paths(checkpoint_root: str):
    pattern = os.path.join(checkpoint_root, "*-step*", "training_state.pt")
    return sorted(glob.glob(pattern))


def check_overfit(
    log_path: str,
    checkpoint_root: str,
    *,
    expected_step: int,
    max_loss: float,
    min_accuracy: float,
) -> Dict[str, object]:
    step, metrics = final_metrics(log_path)
    loss = float(metrics["loss"])
    accuracy = float(metrics["accuracy"])
    checkpoints = checkpoint_paths(checkpoint_root)
    errors = []
    if step != expected_step:
        errors.append(f"final logged step {step} != expected {expected_step}")
    if loss > max_loss:
        errors.append(f"final loss {loss} > {max_loss}")
    if accuracy < min_accuracy:
        errors.append(f"final token accuracy {accuracy} < {min_accuracy}")
    if not checkpoints:
        errors.append(f"no training_state.pt checkpoint under {checkpoint_root}")
    result = {
        "step": step,
        "loss": loss,
        "token_accuracy": accuracy,
        "checkpoint": checkpoints[-1] if checkpoints else None,
        "passed": not errors,
    }
    if errors:
        raise ValueError("; ".join(errors))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--max-loss", type=float, default=1e-4)
    parser.add_argument("--min-accuracy", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = check_overfit(
        args.log_path,
        args.checkpoint_root,
        expected_step=args.expected_step,
        max_loss=args.max_loss,
        min_accuracy=args.min_accuracy,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
