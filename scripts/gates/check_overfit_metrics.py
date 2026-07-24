"""Gate an overfit run on its final loss, accuracy, and checkpoint.

Assumptions about the trainer log and output structure:

- Metric lines match either the historical disaggregated consumer form
  ``[consumer] step <N> {<dict>}`` or the unified trainer form
  ``step <N>: {<dict>}``.
- A committed checkpoint contains both ``training_state.pt`` and ``_SUCCESS``
  under ``<checkpoint_root>/<run-id>-step<N>/``.
  If a new method saves checkpoints under a different layout, adjust
  ``checkpoint_paths()`` to match.
- The gate thresholds ``--max-loss`` and ``--min-accuracy`` are method-agnostic
  CLI arguments; every future overfit gate can pass the values appropriate for that
  method without modifying this script.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import re
from typing import Dict, Tuple

METRIC_LINES = (
    re.compile(r"\[consumer\] step (\d+) (\{.*\})\s*$"),
    re.compile(r"(?:^|\s)step (\d+):\s*(\{.*\})\s*$"),
)


def final_metrics(log_path: str) -> Tuple[int, Dict[str, float]]:
    matches = []
    with open(log_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for pattern in METRIC_LINES:
                match = pattern.search(line)
                if match:
                    metrics = ast.literal_eval(match.group(2))
                    matches.append((int(match.group(1)), metrics))
                    break
    if not matches:
        raise ValueError(f"no consumer metric lines found in {log_path}")
    return matches[-1]


def checkpoint_paths(checkpoint_root: str):
    pattern = os.path.join(checkpoint_root, "*-step*", "training_state.pt")
    paths = [
        path
        for path in glob.glob(pattern)
        if os.path.isfile(os.path.join(os.path.dirname(path), "_SUCCESS"))
    ]

    def get_step(path: str) -> int:
        match = re.search(r"-step(\d+)", path)
        return int(match.group(1)) if match else 0

    return sorted(paths, key=get_step)


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
        errors.append(f"no committed checkpoint under {checkpoint_root}")
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
