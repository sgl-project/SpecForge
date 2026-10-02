"""Bounded real-checkpoint/real-feature smoke, not a convergence benchmark.

Run capture on one GPU, then train with two torchrun ranks. Outputs contain
aggregate metrics and source hashes only; captured token tensors stay local.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
import types
from pathlib import Path

import torch


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("capture", "train"))
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--target-path", help="Cached Qwen3.5-family target checkpoint")
    parser.add_argument(
        "--draft-path", help="Cached matching pretrained DFlash2 checkpoint"
    )
    parser.add_argument(
        "--dataset-path", help="ShareGPT conversations JSON; capture only"
    )
    parser.add_argument("--anchors", type=int, default=512)
    parser.add_argument("--eval-anchors", type=int, default=8)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--repeat-start", type=int, default=0)
    parser.add_argument(
        "--schedules",
        nargs="+",
        choices=("baseline", "online", "offline"),
        default=["baseline", "online", "offline"],
    )
    parser.add_argument("--report-name", default="report.json")
    parser.add_argument(
        "--source-revision",
        help="Source revision when running a tar checkout without .git",
    )
    args = parser.parse_args()
    if args.mode == "capture" and not all(
        (args.target_path, args.draft_path, args.dataset_path)
    ):
        parser.error("capture requires --target-path, --draft-path, --dataset-path")
    if args.warmup < 0 or args.steps <= args.warmup:
        parser.error("--steps must exceed nonnegative --warmup")
    return args


def capture(args):
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    from specforge.modeling.draft.dflash import extract_context_feature

    args.work_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.work_dir / "features"
    feature_dir.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.target_path, local_files_only=True)
    records = json.loads(Path(args.dataset_path).read_text())
    wanted = [128, 256, 512, 1024] * 6
    selected = []
    cursor = 0
    for sample, length in enumerate(wanted):
        while cursor < len(records):
            row = records[cursor]
            cursor += 1
            turns = row.get("conversations", [])
            if (
                len(turns) < 2
                or turns[0].get("from") != "human"
                or turns[1].get("from") != "gpt"
            ):
                continue
            messages = [
                {"role": "user", "content": turns[0]["value"]},
                {"role": "assistant", "content": turns[1]["value"]},
            ]
            full = tokenizer.apply_chat_template(
                messages, tokenize=True, return_dict=False, enable_thinking=False
            )
            prefix = tokenizer.apply_chat_template(
                messages[:1],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if (
                len(full) < length
                or len(prefix) > length - 48
                or full[: len(prefix)] != prefix
            ):
                continue
            ids = torch.tensor(full[:length], dtype=torch.long)
            mask = torch.zeros(length, dtype=torch.long)
            mask[len(prefix) :] = 1
            # Keep the full proposal's labels inside each sequence.
            selected.append(
                (
                    ids,
                    mask,
                    {
                        "sample_id": str(sample),
                        "length": length,
                        "supervised_tokens": int(mask.sum()),
                        "split": "train" if sample < 16 else "heldout",
                        "source_sha256": hashlib.sha256(
                            str(row.get("id", cursor - 1)).encode()
                        ).hexdigest(),
                    },
                )
            )
            break
        else:
            raise RuntimeError(
                "Could not select enough disjoint assistant-supervised real samples"
            )
    print(
        json.dumps({"phase": "selected", "count": len(selected), "scanned": cursor}),
        flush=True,
    )
    started = time.perf_counter()
    target = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            args.target_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to("cuda")
        .eval()
    )
    loaded = time.perf_counter() - started
    layer_ids = json.loads((Path(args.draft_path) / "config.json").read_text())[
        "dflash_config"
    ]["target_layer_ids"]
    text_model = target.model.language_model
    features = {}
    hooks = []
    for layer_id in layer_ids:

        def record(_module, _inputs, output, index=layer_id):
            features[index] = output[0] if isinstance(output, tuple) else output

        hooks.append(text_model.layers[layer_id].register_forward_hook(record))
    capture_times = []
    for ids, mask, metadata in selected:
        features.clear()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            # Post-decoder hooks are exactly HF hidden_states[layer_id + 1].
            text_model(input_ids=ids[None].cuda(), use_cache=False)
            indexed = [None] * (max(layer_ids) + 2)
            for layer_id in layer_ids:
                indexed[layer_id + 1] = features[layer_id]
            hidden = extract_context_feature(indexed, layer_ids).cpu()
        torch.cuda.synchronize()
        capture_times.append(time.perf_counter() - start)
        torch.save(
            {"input_ids": ids, "loss_mask": mask, "hidden_states": hidden},
            feature_dir / f"{metadata['sample_id']}.pt",
        )
        print(
            json.dumps({"phase": "captured", **metadata, "seconds": capture_times[-1]}),
            flush=True,
        )
    for hook in hooks:
        hook.remove()
    report = {
        "target": args.target_path,
        "draft": args.draft_path,
        "dataset": args.dataset_path,
        "capture": "Transformers BF16 SDPA, post-decoder target layers; no generated/random teacher features",
        "target_layer_ids": layer_ids,
        "hf_hidden_state_indices": [layer + 1 for layer in layer_ids],
        "target_load_seconds": loaded,
        "capture_seconds": capture_times,
        "samples": [item[2] for item in selected],
    }
    (args.work_dir / "capture.json").write_text(json.dumps(report, indent=2))


def pretrained_model(args):
    from safetensors import safe_open
    from transformers import Qwen3Config

    from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
    from specforge.modeling.draft.dflash2 import DFlash2DraftModel

    config = Qwen3Config.from_pretrained(args.draft_path, local_files_only=True)
    config._attn_implementation = "flex_attention"
    draft, loading = DFlash2DraftModel.from_pretrained(
        args.draft_path,
        config=config,
        local_files_only=True,
        dtype=torch.bfloat16,
        output_loading_info=True,
    )
    if (
        loading["missing_keys"]
        or loading["unexpected_keys"]
        or loading.get("mismatched_keys")
    ):
        raise RuntimeError(f"Pretrained draft mismatch: {loading}")
    index = json.loads(
        (Path(args.target_path) / "model.safetensors.index.json").read_text()
    )["weight_map"]

    def target_weight(key):
        with safe_open(
            str(Path(args.target_path) / index[key]), framework="pt", device="cpu"
        ) as weights:
            return weights.get_tensor(key)

    with torch.device("meta"):
        head = torch.nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, dtype=torch.bfloat16
        )
        embed = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=torch.bfloat16
        )
    head.weight = torch.nn.Parameter(
        target_weight("lm_head.weight"), requires_grad=False
    )
    embed.weight = torch.nn.Parameter(
        target_weight("model.language_model.embed_tokens.weight"), requires_grad=False
    )
    model = OnlineDFlashModel(
        draft_model=draft,
        target_lm_head=head,
        target_embed_tokens=embed,
        mask_token_id=config.dflash_config["mask_token_id"],
        block_size=config.dflash_config["block_size"],
        num_anchors=args.anchors,
        attention_backend="flex_attention",
        loss_type="dflash",
        objective_chunk_blocks=128,
        loss_decay_gamma=7.0,
        selector_loss_alpha=1.0,
        teacher_metrics=False,
    ).to(device=torch.cuda.current_device(), dtype=torch.bfloat16)
    native = model._sample_anchor_positions

    def anchors(self, seq_len, loss_mask, device, max_valid_anchors=None):
        if not self._smoke_eval:
            return native(seq_len, loss_mask, device, max_valid_anchors)
        positions = []
        for row in loss_mask:
            candidates = torch.where(row[: seq_len - self.block_size + 1] > 0.5)[0]
            # Same held-out positions for every schedule and every checkpoint.
            indices = torch.linspace(
                0, candidates.numel() - 1, args.eval_anchors, device=device
            ).long()
            positions.append(candidates[indices])
        positions = torch.stack(positions)
        return positions, torch.ones_like(positions, dtype=torch.bool)

    model._sample_anchor_positions = types.MethodType(anchors, model)
    model._smoke_eval = False
    return model


def train(args):
    import torch.distributed as dist

    from benchmarks.length_aware_training import (
        collate_batches,
        layout_report,
        rank_batches,
    )
    from specforge.algorithms.common.hidden_states_data import normalize_offline_sample
    from specforge.data.length_bucketing import bucket_by_length
    from specforge.optimizer import BF16Optimizer
    from specforge.runtime.contracts import SampleRef
    from specforge.runtime.data_plane.ref_distributor import _length_grouped_window
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
    from specforge.training.controller import TrainerCore
    from specforge.training.strategies.base import DFlashTrainStrategy, StepContext

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2
    args.algorithm, args.batch_size, args.accumulation_steps = "dflash2", 2, 4
    metadata = json.loads((args.work_dir / "capture.json").read_text())
    for name in ("target", "draft"):
        supplied = getattr(args, f"{name}_path")
        if (
            supplied is not None
            and Path(supplied).resolve() != Path(metadata[name]).resolve()
        ):
            raise ValueError(f"--{name}-path must match the feature capture manifest")
    args.target_path = args.target_path or metadata["target"]
    args.draft_path = args.draft_path or metadata["draft"]
    source_files = [
        "benchmarks/pretrained_length_smoke.py",
        "benchmarks/length_aware_training.py",
        "specforge/runtime/data_plane/ref_distributor.py",
        "specforge/data/length_bucketing.py",
        "specforge/algorithms/common/dflash_family_model.py",
        "specforge/modeling/draft/dflash2.py",
        "specforge/training/controller.py",
        "specforge/training/backend.py",
        "specforge/optimizer.py",
    ]
    revision = args.source_revision
    if revision is None:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, capture_output=True
        )
        revision = (
            git.stdout.strip()
            if git.returncode == 0
            else "unavailable; see file hashes"
        )
    provenance = {
        "git_head": revision,
        "source_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in source_files
        },
        "draft_config": json.loads((Path(args.draft_path) / "config.json").read_text()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
    }
    features, refs, heldout = {}, [], []
    for sample in metadata["samples"]:
        sid = sample["sample_id"]
        raw = torch.load(args.work_dir / "features" / f"{sid}.pt", weights_only=True)
        features[sid] = normalize_offline_sample(raw, 1024)
        ref = SampleRef(
            sid,
            "pretrained-smoke",
            None,
            "local://captured",
            {},
            {},
            "dflash",
            num_tokens=sample["length"],
        )
        (refs if sample["split"] == "train" else heldout).append(ref)
    random.Random(42).shuffle(refs)
    orders = {
        "baseline": refs,
        "online": _length_grouped_window(refs, world, args.batch_size),
        "offline": bucket_by_length(
            refs,
            length_fn=lambda ref: ref.num_tokens,
            batch_size=args.batch_size,
            dp_size=world,
            length_bucket_size=4,
            seed=42,
        ),
    }
    eval_batches = collate_batches(
        args, [[ref] for ref in heldout[rank::world]], features
    )
    results = []
    for repeat in range(args.repeat_start, args.repeat_start + args.repeats):
        schedule_names = (
            list(args.schedules) if repeat % 2 == 0 else list(reversed(args.schedules))
        )
        for name in schedule_names:
            torch.manual_seed(42)
            model = pretrained_model(args)
            backend = FSDPTrainingBackend(
                ParallelConfig.from_distributed(
                    sharding_strategy="SHARD_GRAD_OP", param_dtype=torch.bfloat16
                ),
                optimizer_factory=lambda module: BF16Optimizer(
                    module,
                    lr=1e-5,
                    max_grad_norm=1.0,
                    warmup_ratio=0.0,
                    total_steps=100,
                    lr_scheduler="constant",
                ),
            )
            backend.prepare_model(model, optimizer_target=model.draft_model)
            strategy = DFlashTrainStrategy(backend.module)
            supervised = []
            original_forward = strategy.forward_loss

            def observe(batch, ctx=None):
                out = original_forward(batch, ctx)
                if not model._smoke_eval:
                    supervised.append(out.metrics["accuracy_denom"].detach())
                return out

            strategy.forward_loss = observe
            core = TrainerCore(
                strategy, backend, accumulation_steps=args.accumulation_steps
            )

            def evaluate():
                model._smoke_eval = True
                backend.module.eval()
                totals = torch.zeros(4, device="cuda", dtype=torch.float64)
                with torch.no_grad():
                    for batch in eval_batches:
                        out = strategy.forward_loss(
                            batch,
                            StepContext(
                                global_step=0,
                                total_steps=100,
                                collect_detailed_metrics=False,
                            ),
                        )
                        numerator, denominator = out.loss_terms
                        correct, count = out.ratio_metrics["acc"]
                        totals += torch.stack(
                            [numerator, denominator, correct, count]
                        ).double()
                dist.all_reduce(totals)
                values = totals.cpu().tolist()
                model._smoke_eval = False
                backend.module.train()
                return {
                    "loss": values[0] / values[1],
                    "unary_hard_label_accuracy": values[2] / values[3],
                    "supervised_positions": values[3],
                    "finite": all(
                        map(
                            lambda value: torch.isfinite(torch.tensor(value)).item(),
                            values,
                        )
                    ),
                }

            before = evaluate()
            batches = collate_batches(
                args, rank_batches(orders[name], rank, world, args.batch_size), features
            )
            times, metrics, position_counts = [], [], []
            for step in range(args.steps):
                supervised.clear()
                dist.barrier()
                torch.cuda.synchronize()
                if step == args.warmup:
                    torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                for batch in batches:
                    result = core.train_step(
                        batch,
                        StepContext(
                            global_step=step,
                            total_steps=100,
                            collect_detailed_metrics=False,
                        ),
                    )
                torch.cuda.synchronize()
                elapsed = torch.tensor(time.perf_counter() - started, device="cuda")
                dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                times.append(elapsed.item())
                positions = torch.stack(supervised).sum()
                dist.all_reduce(positions)
                position_counts.append(int(positions.item()))
                metrics.append(result.materialize_metrics())
                if rank == 0:
                    print(
                        json.dumps(
                            {
                                "phase": "step",
                                "schedule": name,
                                "repeat": repeat,
                                "step": step,
                                "seconds": times[-1],
                                "metrics": metrics[-1],
                            }
                        ),
                        flush=True,
                    )
            peak = torch.tensor(
                torch.cuda.max_memory_allocated(), device="cuda", dtype=torch.int64
            )
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
            peak_gb = peak.item() / 1e9
            after = evaluate()
            record = {
                "schedule": name,
                "repeat": repeat,
                "seconds_per_step": times,
                "warmup_steps_excluded": args.warmup,
                "median_measured_seconds": statistics.median(times[args.warmup :]),
                "supervised_positions_per_step": position_counts,
                "heldout_before": before,
                "heldout_after": after,
                "train_metrics": metrics,
                "peak_allocated_gb": peak_gb,
                "peak_memory_scope": "Maximum allocated across ranks during measured training steps; counter reset after warmup, before timed steps; heldout evaluation excluded",
            }
            results.append(record)
            if rank == 0:
                report = {
                    "scope": "Real pretrained target/draft and ShareGPT assistant features; 16 training and 8 disjoint heldout sequences. Short fine-tuning smoke, not convergence or serving acceptance. Training anchors sampled natively; evaluation uses 8 fixed anchors/sample, a smaller diagnostic budget. Feature tensors preloaded; timing excludes target capture and disk IO. Offline bucket confined to one optimizer window. LR 1e-5 is a predeclared conservative smoke variation from the recipe's 5e-4.",
                    "anchors": args.anchors,
                    "eval_anchors": args.eval_anchors,
                    "attention_backend": "flex_attention",
                    "objective_chunk_blocks": 128,
                    "loss_type": "dflash",
                    "loss_decay_gamma": 7.0,
                    "learning_rate": 1e-5,
                    "precision": "bfloat16",
                    "backend": "FSDP SHARD_GRAD_OP, production TrainerCore and BF16Optimizer",
                    "protocol": {
                        "total_steps_per_run": args.steps,
                        "warmup_steps": args.warmup,
                        "measured_steps": args.steps - args.warmup,
                        "repeats": args.repeats,
                        "order": schedule_names,
                        "selected_schedules": args.schedules,
                        "training_seed": 42,
                        "train_samples": 16,
                        "heldout_samples": 8,
                        "batch_size_per_rank": 2,
                        "world_size": world,
                        "accumulation_steps": args.accumulation_steps,
                    },
                    "layouts": {
                        schedule: layout_report(args, order, world)
                        for schedule, order in orders.items()
                    },
                    "provenance": provenance,
                    "metric_definitions": {
                        "heldout_loss": "Combined decayed DFlash token objective plus selector objective with alpha=1",
                        "unary_hard_label_accuracy": "Unary-logit argmax accuracy before candidate-selector decisions; not serving acceptance",
                    },
                    "capture": metadata,
                    "runs": results,
                }
                (args.work_dir / args.report_name).write_text(
                    json.dumps(report, indent=2)
                )
            del core, strategy, backend, model, original_forward, observe, evaluate
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    args = arguments()
    (capture if args.mode == "capture" else train)(args)
