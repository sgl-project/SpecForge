"""Replay a DFlash checkpoint on a fixed cache of teacher features."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_index(path: Path, limit: int | None = None) -> list[dict]:
    """Resolve ordered single-example files and stable anchor-sampling seeds."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    manifest = json.loads(path.read_text())
    rows = manifest["rows"]
    if not rows:
        raise ValueError("feature index must contain at least one example")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("feature index contains duplicate example IDs")
    resolved = []
    for row in rows[:limit]:
        feature_path = Path(row["path"])
        if not feature_path.is_absolute():
            feature_path = path.parent / feature_path
        if not feature_path.is_file():
            raise FileNotFoundError(feature_path)
        seed = row.get("seed")
        if seed is None:
            # group_hash preserves the seeds used by existing frozen dev caches.
            identity = (
                row.get("group_hash")
                or hashlib.sha256(str(row["id"]).encode()).hexdigest()
            )
            seed = int(identity[:8], 16)
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError(f"invalid uint32 seed for example {row['id']!r}: {seed}")
        resolved.append({**row, "path": str(feature_path.resolve()), "seed": seed})
    return resolved


def _scalar(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite evaluation metric: {number}")
    return number


def _features(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    tensors = torch.load(path, map_location="cpu", weights_only=True)
    required = {"input_ids", "hidden_states", "loss_mask"}
    if not isinstance(tensors, dict) or not required <= tensors.keys():
        raise ValueError(f"{path} must contain {sorted(required)}")
    selected = {key: tensors[key] for key in required}
    if "target_last_hidden_states" in tensors:
        selected["target_last_hidden_states"] = tensors["target_last_hidden_states"]
    ids = selected["input_ids"]
    if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError(f"{path}: input_ids must have shape [1, sequence_length]")
    for name, value in selected.items():
        expected_ndim = 2 if name in {"input_ids", "loss_mask"} else 3
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != expected_ndim
            or value.shape[:2] != ids.shape
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"{path}: invalid shape or non-finite values in {name}")
    if not (selected["loss_mask"] > 0).any():
        raise ValueError(f"{path}: loss_mask contains no supervised tokens")
    return {key: value.to(device) for key, value in selected.items()}


def evaluate_cached_features(model: torch.nn.Module, rows: list[dict]) -> dict:
    """Preserve sample means and pool additive metrics before taking ratios."""
    if not rows:
        raise ValueError("evaluation requires at least one example")
    device = next(model.parameters()).device
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("cached checkpoint evaluation currently supports CPU and CUDA")
    cuda_devices = [device.index] if device.type == "cuda" else []
    records = []
    ratio_totals: dict[str, list[float]] = {}
    sum_totals: dict[str, float] = {}
    was_training = model.training
    started = time.monotonic()
    model.eval()
    try:
        with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
            for row in rows:
                torch.random.default_generator.manual_seed(row["seed"])
                if cuda_devices:
                    with torch.cuda.device(device):
                        torch.cuda.manual_seed(row["seed"])
                path = Path(row["path"])
                digest = _sha256(path)
                if row.get("sha256") is not None and row["sha256"] != digest:
                    raise ValueError(
                        f"feature digest mismatch for {row['id']!r}: {path}"
                    )
                loss, accuracy, metrics = model(
                    **_features(path, device), collect_detailed_metrics=True
                )
                scalars = {"loss": _scalar(loss), "accuracy": _scalar(accuracy)}
                scalars.update(
                    (key, _scalar(value))
                    for key, value in metrics.items()
                    if isinstance(value, (int, float))
                    or isinstance(value, torch.Tensor)
                    and value.numel() == 1
                )
                ratios = {}
                for key, pair in metrics.get("ratio_metrics", {}).items():
                    numerator, denominator = map(_scalar, pair)
                    if denominator < 0 or denominator == 0 and numerator != 0:
                        raise ValueError(f"invalid ratio {key}: {pair}")
                    ratios[key] = [numerator, denominator]
                    total = ratio_totals.setdefault(key, [0.0, 0.0])
                    total[0] += numerator
                    total[1] += denominator
                sums = {
                    key: _scalar(value)
                    for key, value in metrics.get("sum_metrics", {}).items()
                }
                for key, value in sums.items():
                    sum_totals[key] = sum_totals.get(key, 0.0) + value
                if records:
                    for field, values in (
                        ("metrics", scalars),
                        ("ratio_metrics", ratios),
                        ("sum_metrics", sums),
                    ):
                        if values.keys() != records[0][field].keys():
                            raise ValueError(
                                f"{field} names changed between examples; use a uniform feature cache"
                            )
                records.append(
                    {
                        "id": row["id"],
                        "seed": row["seed"],
                        "sha256": digest,
                        "metrics": scalars,
                        "ratio_metrics": ratios,
                        "sum_metrics": sums,
                    }
                )
    finally:
        model.train(was_training)
    means = {
        key: sum(record["metrics"][key] for record in records) / len(records)
        for key in records[0]["metrics"]
    }
    means.update(
        {
            key: numerator / denominator if denominator else 0.0
            for key, (numerator, denominator) in ratio_totals.items()
        }
    )
    return {
        "schema_version": 1,
        "success": True,
        "examples": len(records),
        "seconds": time.monotonic() - started,
        "optimizer_updates": 0,
        "metric_scope": "teacher-forced diagnostics; not serving acceptance, speed, or answer accuracy",
        "mean": means,
        "ratio_totals": ratio_totals,
        "sum_totals": sum_totals,
        "rows": records,
    }


def run_checkpoint_evaluation(
    config, checkpoint: str, index: Path, output: Path, limit: int | None = None
) -> dict:
    """Build only draft/target-head modules; no optimizer or capture service."""
    from specforge.algorithms.builtin import builtin_algorithm_registry
    from specforge.training.assembly import build_model_bundle
    from specforge.training.model_loading import warm_start_draft_model

    if output.exists():
        raise FileExistsError(f"evaluation output already exists: {output}")
    if config.training.strategy != "dflash" or config.model.input_modality != "text":
        raise ValueError("cached checkpoint evaluation requires text strategy=dflash")
    if (
        torch.distributed.is_initialized()
        or int(os.environ.get("WORLD_SIZE", "1")) != 1
    ):
        raise ValueError("run checkpoint evaluation in a standalone single process")
    rows = load_feature_index(index, limit)
    config = config.model_copy(deep=True)
    # The evaluation checkpoint is weights-only even when the recipe was resumed.
    config.model.draft_checkpoint_path = None
    config.training.resume_from = None
    torch.manual_seed(config.training.seed)
    bundle = build_model_bundle(
        config, algorithm=builtin_algorithm_registry().resolve("dflash")
    )
    loaded = warm_start_draft_model(
        bundle.draft_model,
        checkpoint,
        draft_config=bundle.draft_config,
        strategy="dflash",
        allow_missing_embedding=True,
        cache_dir=config.model.cache_dir,
        trust_remote_code=config.model.trust_remote_code,
    )
    report = evaluate_cached_features(bundle.model, rows)
    report.update(
        {
            "checkpoint": checkpoint,
            "checkpoint_format": loaded.checkpoint_format,
            "feature_index": str(index.resolve()),
            "feature_index_sha256": _sha256(index),
            "recipe": config.model_dump(mode="json"),
            "draft_config": bundle.draft_config.to_dict(),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return report
