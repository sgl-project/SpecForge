"""Compare dense vs truncated-SVD efficient-head TTT acceptance length.

Loads a stage-1 dense EAGLE-4 checkpoint and offline hidden-state features,
then reports SpecForge ``eval/simulated_acc_len``, per-position accuracy, and
(for SVD) candidate recall. This is teacher-forced TTT, not SGLang serving.

Example (after capturing held-out hidden states):

    python scripts/eval_eagle4_acc.py \\
      --checkpoint /path/to/qwen3-4b-eagle4-online-step16947 \\
      --draft-config configs/qwen3-4b-eagle4.json \\
      --target-model-path Qwen/Qwen3-4B \\
      --hidden-states-path ./cache/hidden_states/qwen3-4b-eagle4-heldout \\
      --output-json ./outputs/qwen3-4b-eagle4-heldout-acc.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from specforge.runtime.contracts import TrainBatch

EVAL_SCALAR_KEYS = (
    "eval/simulated_acc_len",
    "eval/avg_acc",
    "eval/avg_loss",
)


def jsonable(value: Any) -> Any:
    item = getattr(value, "item", None)
    ndim = getattr(value, "ndim", None)
    if callable(item) and ndim == 0:
        return float(item())
    tolist = getattr(value, "tolist", None)
    if callable(tolist) and ndim is not None and ndim > 0:
        return jsonable(tolist())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float):
        return float(value)
    return value


def relative_drop(baseline: float, other: float) -> float:
    if baseline == 0.0:
        return 0.0
    return (other - baseline) / baseline


def summarize_comparison(
    dense: dict[str, Any], svd_runs: list[dict[str, Any]]
) -> dict[str, Any]:
    payload = {"dense": jsonable(dense), "svd": []}
    dense_len = float(dense["eval/simulated_acc_len"])
    dense_acc = float(dense["eval/avg_acc"])
    for run in svd_runs:
        acc_len = float(run["eval/simulated_acc_len"])
        avg_acc = float(run["eval/avg_acc"])
        payload["svd"].append(
            {
                **jsonable(run),
                "delta_simulated_acc_len": acc_len - dense_len,
                "relative_simulated_acc_len": relative_drop(dense_len, acc_len),
                "delta_avg_acc": avg_acc - dense_acc,
                "relative_avg_acc": relative_drop(dense_acc, avg_acc),
            }
        )
    return payload


def _to_batch(collated: dict[str, Any], sample_ids: list[str]) -> TrainBatch:
    tensors = {
        key: value
        for key, value in collated.items()
        if value is not None and hasattr(value, "shape")
    }
    return TrainBatch(
        sample_ids=sample_ids,
        strategy="eagle3",
        tensors=tensors,
        metadata={"target_repr": "hidden_state"},
    )


def iter_feature_batches(
    hidden_states_path: str,
    *,
    max_len: int,
    batch_size: int,
    max_samples: Optional[int],
) -> Iterator[TrainBatch]:
    from specforge.algorithms.eagle3.data import normalize_offline_sample
    from specforge.data.utils import DataCollatorWithPadding
    from specforge.runtime.data_plane.feature_store import load_feature_file
    from specforge.runtime.data_plane.offline_reader import list_feature_files

    files = list_feature_files(hidden_states_path)
    if not files:
        raise FileNotFoundError(f"no offline feature files under {hidden_states_path}")
    if max_samples is not None:
        files = files[:max_samples]
    collator = DataCollatorWithPadding()
    chunk: list[dict[str, Any]] = []
    ids: list[str] = []
    for path in files:
        sample = normalize_offline_sample(load_feature_file(path), max_len)
        chunk.append(sample)
        ids.append(path)
        if len(chunk) == batch_size:
            yield _to_batch(collator(chunk), ids)
            chunk = []
            ids = []
    if chunk:
        yield _to_batch(collator(chunk), ids)


def _efficient_missing_keys(model: Any) -> set[str]:
    return {name for name in model.state_dict() if name.startswith("efficient_lm_head")}


def load_dense_draft(
    *,
    checkpoint: str,
    draft_config_path: str,
    target_model_path: str,
    attention_backend: str,
    dtype: Any,
    embedding_key: str,
    cache_dir: Optional[str],
):
    from specforge.modeling.auto import AutoDraftModel
    from specforge.training.model_loading import (
        load_draft_config_source,
        warm_start_draft_model,
    )
    from specforge.utils import get_local_device

    draft_config = load_draft_config_source(draft_config_path, cache_dir=cache_dir)
    from specforge.modeling.draft.eagle4_structure import candidate_training_enabled

    if candidate_training_enabled(draft_config):
        raise ValueError("dense eval expects a stage-1 config")
    model = AutoDraftModel.from_config(
        draft_config, attention_backend=attention_backend, torch_dtype=dtype
    )
    warm_start_draft_model(
        model,
        checkpoint,
        draft_config=draft_config,
        strategy="eagle4",
        allow_missing_embedding=True,
    )
    model.load_embedding(
        target_model_path, embedding_key=embedding_key, cache_dir=cache_dir
    )
    model.freeze_embedding()
    return model.to(device=get_local_device(), dtype=dtype).eval()


def load_svd_draft(
    dense_model: Any,
    *,
    rank: int,
    num_candidates: int,
    attention_backend: str,
    dtype: Any,
):
    from specforge.modeling.auto import AutoDraftModel
    from specforge.utils import get_local_device

    config = copy.deepcopy(dense_model.config)
    config.eagle4_training_stage = "stage2"
    config.efficient_lm_head_config = {
        "type": "low_rank",
        "rank": rank,
        "num_candidates": num_candidates,
    }
    model = AutoDraftModel.from_config(
        config, attention_backend=attention_backend, torch_dtype=dtype
    )
    state = dense_model.state_dict()
    missing = _efficient_missing_keys(model)
    result = model.load_state_dict(
        {key: value for key, value in state.items() if key not in missing},
        strict=False,
    )
    unexpected = [key for key in result.unexpected_keys if key not in {"t2d", "d2t"}]
    if unexpected:
        raise ValueError(f"SVD draft rejected unexpected dense keys: {unexpected}")
    required_missing = sorted(set(result.missing_keys) - missing)
    if required_missing:
        raise ValueError(f"SVD draft is missing dense weights: {required_missing}")
    model.init_efficient_head_from_dense()
    model.freeze_embedding()
    return model.to(device=get_local_device(), dtype=dtype).eval()


def _eval_metrics(
    draft_model: Any,
    *,
    target_head: Any,
    batches: Iterable[TrainBatch],
    ttt_length: int,
    attention_backend: str,
) -> dict[str, Any]:
    from specforge.algorithms.eagle3.model import OnlineEagle3Model
    from specforge.eval.evaluator import Evaluator
    from specforge.training.strategies.base import Eagle3TrainStrategy
    from specforge.utils import get_local_device

    wrapper = OnlineEagle3Model(
        draft_model=draft_model,
        length=ttt_length,
        attention_backend=attention_backend,
    ).to(device=get_local_device())
    wrapper.eval()
    strategy = Eagle3TrainStrategy(wrapper, target_head=target_head)
    recall_sum = 0.0
    recall_n = 0

    def forward(batch: TrainBatch):
        nonlocal recall_sum, recall_n
        output = strategy.forward_loss(batch)
        recall = output.metrics.get("candidate_recall")
        if recall is not None:
            item = getattr(recall, "item", None)
            recall_sum += float(item() if callable(item) else recall)
            recall_n += 1
        return output

    metrics = Evaluator().run(forward, batches)
    if recall_n:
        metrics["eval/candidate_recall"] = recall_sum / recall_n
    return metrics


def load_trained_draft(
    *,
    checkpoint: str,
    draft_config_path: str,
    target_model_path: str,
    attention_backend: str,
    dtype: Any,
    embedding_key: str,
    cache_dir: Optional[str],
):
    """Load a stage-2 checkpoint, including its trained low-rank factors."""

    from specforge.modeling.auto import AutoDraftModel
    from specforge.modeling.draft.eagle4_structure import candidate_training_enabled
    from specforge.training.model_loading import (
        load_draft_config_source,
        warm_start_draft_model,
    )
    from specforge.utils import get_local_device

    draft_config = load_draft_config_source(draft_config_path, cache_dir=cache_dir)
    if not candidate_training_enabled(draft_config):
        raise ValueError("trained-head eval expects eagle4_training_stage='stage2'")
    model = AutoDraftModel.from_config(
        draft_config, attention_backend=attention_backend, torch_dtype=dtype
    )
    report = warm_start_draft_model(
        model,
        checkpoint,
        draft_config=draft_config,
        strategy="eagle4",
        allow_missing_embedding=True,
        allowed_unexpected_keys={"t2d", "d2t"},
    )
    missing_head = [
        key for key in report.missing_keys if key.startswith("efficient_lm_head")
    ]
    if missing_head:
        raise ValueError(
            "trained-head checkpoint is missing low-rank factors "
            f"{missing_head}; refusing to SVD-initialize them"
        )
    model.load_embedding(
        target_model_path, embedding_key=embedding_key, cache_dir=cache_dir
    )
    model.freeze_embedding()
    return model.to(device=get_local_device(), dtype=dtype).eval()


def evaluate_trained_head(
    *,
    checkpoint: str,
    draft_config: str,
    target_model_path: str,
    hidden_states_path: str,
    ttt_length: int,
    max_length: int,
    batch_size: int,
    max_samples: Optional[int],
    num_candidates: list[int],
    attention_backend: str,
    embedding_key: str,
    cache_dir: Optional[str],
    dtype: Any,
    baseline_dense: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Score a trained candidate head. ``num_candidates`` only changes top-k."""

    from specforge.modeling.target.target_head import TargetHead
    from specforge.runtime.data_plane.offline_reader import list_feature_files

    files = list_feature_files(hidden_states_path)
    if not files:
        raise FileNotFoundError(f"no offline feature files under {hidden_states_path}")
    if max_samples is not None:
        files = files[:max_samples]
    model = load_trained_draft(
        checkpoint=checkpoint,
        draft_config_path=draft_config,
        target_model_path=target_model_path,
        attention_backend=attention_backend,
        dtype=dtype,
        embedding_key=embedding_key,
        cache_dir=cache_dir,
    )
    target_head = TargetHead.from_pretrained(
        target_model_path,
        embedding_key=embedding_key,
        cache_dir=cache_dir,
        dtype=dtype,
    )
    rank = int(model.efficient_lm_head_down_proj.out_features)
    runs = []
    for k in num_candidates:
        model.num_candidates = int(k)
        metrics = _eval_metrics(
            model,
            target_head=target_head,
            batches=iter_feature_batches(
                hidden_states_path,
                max_len=max_length,
                batch_size=batch_size,
                max_samples=max_samples,
            ),
            ttt_length=ttt_length,
            attention_backend=attention_backend,
        )
        metrics["rank"] = rank
        metrics["num_candidates"] = int(k)
        metrics["num_samples"] = len(files)
        metrics["head"] = "trained"
        runs.append(metrics)
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if baseline_dense is None:
        return {"trained": [jsonable(run) for run in runs]}
    compared = summarize_comparison(baseline_dense, runs)
    return {"dense": compared["dense"], "trained": compared["svd"]}


def evaluate_heads(
    *,
    checkpoint: str,
    draft_config: str,
    target_model_path: str,
    hidden_states_path: str,
    ttt_length: int,
    max_length: int,
    batch_size: int,
    max_samples: Optional[int],
    rank: int,
    num_candidates: list[int],
    attention_backend: str,
    embedding_key: str,
    cache_dir: Optional[str],
    dtype: Any,
) -> dict[str, Any]:
    from specforge.runtime.data_plane.offline_reader import list_feature_files

    files = list_feature_files(hidden_states_path)
    if not files:
        raise FileNotFoundError(f"no offline feature files under {hidden_states_path}")
    if max_samples is not None:
        files = files[:max_samples]
    dense = load_dense_draft(
        checkpoint=checkpoint,
        draft_config_path=draft_config,
        target_model_path=target_model_path,
        attention_backend=attention_backend,
        dtype=dtype,
        embedding_key=embedding_key,
        cache_dir=cache_dir,
    )
    from specforge.modeling.target.target_head import TargetHead

    target_head = TargetHead.from_pretrained(
        target_model_path,
        embedding_key=embedding_key,
        cache_dir=cache_dir,
        dtype=dtype,
    )
    dense_metrics = _eval_metrics(
        dense,
        target_head=target_head,
        batches=iter_feature_batches(
            hidden_states_path,
            max_len=max_length,
            batch_size=batch_size,
            max_samples=max_samples,
        ),
        ttt_length=ttt_length,
        attention_backend=attention_backend,
    )
    dense_metrics["num_samples"] = len(files)
    svd_runs = []
    for k in num_candidates:
        svd = load_svd_draft(
            dense,
            rank=rank,
            num_candidates=k,
            attention_backend=attention_backend,
            dtype=dtype,
        )
        metrics = _eval_metrics(
            svd,
            target_head=target_head,
            batches=iter_feature_batches(
                hidden_states_path,
                max_len=max_length,
                batch_size=batch_size,
                max_samples=max_samples,
            ),
            ttt_length=ttt_length,
            attention_backend=attention_backend,
        )
        metrics["rank"] = rank
        metrics["num_candidates"] = k
        metrics["num_samples"] = len(files)
        svd_runs.append(metrics)
        del svd
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summarize_comparison(dense_metrics, svd_runs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--draft-config", required=True)
    parser.add_argument("--target-model-path", required=True)
    parser.add_argument("--hidden-states-path", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--ttt-length", type=int, default=7)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument(
        "--num-candidates",
        type=int,
        nargs="+",
        default=[128, 32],
        help="Top-k values. dense-vs-svd builds one SVD head per value; trained reuses the loaded factors.",
    )
    parser.add_argument(
        "--mode",
        choices=("dense-vs-svd", "trained"),
        default="dense-vs-svd",
        help="trained loads a stage-2 checkpoint and does not SVD-initialize.",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Previous dense-vs-SVD JSON. Trained mode reports deltas against its dense head.",
    )
    parser.add_argument("--attention-backend", default="flex_attention")
    parser.add_argument("--embedding-key", default="model.embed_tokens.weight")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float16", "float32"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    dtype = getattr(torch, args.dtype)
    common = dict(
        checkpoint=args.checkpoint,
        draft_config=args.draft_config,
        target_model_path=args.target_model_path,
        hidden_states_path=args.hidden_states_path,
        ttt_length=args.ttt_length,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        num_candidates=list(args.num_candidates),
        attention_backend=args.attention_backend,
        embedding_key=args.embedding_key,
        cache_dir=args.cache_dir,
        dtype=dtype,
    )
    if args.mode == "trained":
        baseline_dense = None
        baseline_svd = None
        if args.baseline_json is not None:
            baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
            baseline_dense = baseline["dense"]
            baseline_svd = baseline.get("svd")
        payload = evaluate_trained_head(**common, baseline_dense=baseline_dense)
        if baseline_svd is not None:
            payload["svd_init"] = baseline_svd
    else:
        payload = evaluate_heads(**common, rank=args.rank)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
