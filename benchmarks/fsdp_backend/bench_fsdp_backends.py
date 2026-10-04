"""Compare the FSDP1 and FSDP2 TrainingBackends on production-shaped drafts.

Builds the real SpecForge composite training models (EAGLE3, DFlash2, DSpark)
with random weights at production shapes, drives them through the real
``TrainerCore`` / strategy / backend seam with synthetic batches, and records
step time, peak memory, host synchronizations and NCCL collective counts.

Run under torchrun, one process per GPU::

    torchrun --nproc_per_node=4 benchmarks/fsdp_backend/bench_fsdp_backends.py \
        --algo dflash2 --backend fsdp2 --out-dir /tmp/bench

Every rank writes ``<out-dir>/<algo>-<backend>-rank<r>.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
import time
import traceback

import torch
import torch.distributed as dist
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", required=True, choices=["eagle3", "dflash2", "dspark"])
    p.add_argument("--backend", required=True, choices=["fsdp", "fsdp2"])
    p.add_argument("--sharding", default="SHARD_GRAD_OP")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--draft-config", default=None)
    p.add_argument("--attention-backend", default="flex_attention")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--ttt-length", type=int, default=7)
    p.add_argument("--num-anchors", type=int, default=512)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--objective-chunk-blocks", type=int, default=128)
    p.add_argument("--teacher-metrics", action="store_true")
    p.add_argument("--detailed-metrics", action="store_true")
    p.add_argument("--warmup", type=int, default=2, help="optimizer steps")
    p.add_argument("--steps", type=int, default=5, help="measured optimizer steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--profile", action="store_true", default=True)
    p.add_argument("--compile-blocks", action="store_true")
    p.add_argument("--fp8-linear", action="store_true")
    p.add_argument("--shard-frozen-tables", action="store_true")
    p.add_argument("--measure-checkpoint", action="store_true")
    p.add_argument("--label", default=None, help="result file label; defaults to backend")
    p.add_argument("--no-profile", dest="profile", action="store_false")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# Model builders (random weights, production shapes, no downloads)
# --------------------------------------------------------------------------- #


def _repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_draft_config(algo):
    root = _repo_root()
    return {
        "eagle3": os.path.join(root, "configs", "qwen3-8b-eagle3.json"),
        "dflash2": os.path.join(root, "configs", "qwen3-8b-dflash.json"),
        "dspark": os.path.join(root, "configs", "qwen3-8b-dspark.json"),
    }[algo]


def _write_target_config_dir(d, hidden, vocab):
    os.makedirs(d, exist_ok=True)
    cfg = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": hidden,
        "vocab_size": vocab,
        "num_hidden_layers": 1,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 4 * hidden,
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    return d


def build_eagle3(args, device, workdir, rank):
    from specforge.algorithms.eagle3.model import OnlineEagle3Model
    from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
    from specforge.modeling.target.target_head import TargetHead

    draft_config = AutoDraftModelConfig.from_file(args.draft_config)
    torch.manual_seed(args.seed)
    draft = AutoDraftModel.from_config(
        draft_config,
        attention_backend=args.attention_backend,
        torch_dtype=torch.bfloat16,
    ).to(device)
    vocab, draft_vocab = int(draft_config.vocab_size), int(draft_config.draft_vocab_size)
    g = torch.Generator().manual_seed(0)
    draft_ids = torch.randperm(vocab, generator=g)[:draft_vocab].sort().values
    t2d = torch.zeros(vocab, dtype=torch.bool)
    t2d[draft_ids] = True
    d2t = (draft_ids - torch.arange(draft_vocab)).to(torch.int64)
    vm_path = os.path.join(workdir, f"vocab_mapping_rank{rank}.pt")
    torch.save({"t2d": t2d, "d2t": d2t}, vm_path)
    draft.load_vocab_mapping(vm_path)
    draft.freeze_embedding()

    hidden = int(draft_config.hidden_size)
    head_dir = _write_target_config_dir(
        os.path.join(workdir, f"target_rank{rank}"), hidden, vocab
    )
    head = TargetHead(head_dir)
    torch.manual_seed(args.seed + 1)
    with torch.no_grad():
        nn.init.normal_(head.fc.weight, std=0.02)
    head.freeze_weights()
    head = head.eval().to(device=device, dtype=torch.bfloat16)

    model = OnlineEagle3Model(
        draft_model=draft,
        length=args.ttt_length,
        attention_backend=args.attention_backend,
    ).to(device)
    return model, head, {"hidden": hidden, "vocab": vocab}


def make_eagle3_batch(args, device, workdir, rank, shapes):
    """Go through the real offline reader / normalizer / collator / loader."""
    from specforge.algorithms.builtin import builtin_algorithm_registry
    from specforge.runtime.data_plane import FeatureDataLoader, LocalFeatureStore

    hidden, vocab = shapes["hidden"], shapes["vocab"]
    feat_dir = os.path.join(workdir, f"eagle3_features_rank{rank}")
    os.makedirs(feat_dir, exist_ok=True)
    g = torch.Generator().manual_seed(1234 + rank)
    seq = args.seq_len
    for i in range(args.batch):
        torch.save(
            {
                "input_ids": torch.randint(0, vocab, (seq,), generator=g),
                "loss_mask": torch.ones(seq, dtype=torch.long),
                "hidden_state": torch.randn(1, seq, hidden, generator=g).to(
                    torch.bfloat16
                ),
                "aux_hidden_state": torch.randn(1, seq, 3 * hidden, generator=g).to(
                    torch.bfloat16
                ),
            },
            os.path.join(feat_dir, f"{i:04d}.ckpt"),
        )
    algorithm = builtin_algorithm_registry().resolve("eagle3")
    provider = algorithm.providers.offline_for("text")
    reader = provider.build_reader(
        feat_dir, run_id=f"bench{rank}", ttt_length=args.ttt_length, max_len=seq
    )
    refs = reader.read()
    loader = FeatureDataLoader(
        LocalFeatureStore(f"bench{rank}-features"),
        refs=refs,
        batch_size=args.batch,
        collate_fn=provider.build_collator(),
        per_sample_transform=provider.build_normalizer(seq),
        strategy=algorithm.name,
    )
    batch = next(iter(loader))
    return batch


def build_dflash_family(args, device, kind):
    from transformers import Qwen3Config

    from specforge.algorithms.common.dflash_family_model import (
        OnlineDFlashModel,
        OnlineDSparkModel,
    )

    raw = json.load(open(args.draft_config))
    dc = dict(raw.get("dflash_config") or {})
    block_size = args.block_size or raw.get("block_size") or dc.get("block_size")
    cfg = Qwen3Config(
        hidden_size=raw["hidden_size"],
        num_hidden_layers=raw["num_hidden_layers"],
        num_attention_heads=raw["num_attention_heads"],
        num_key_value_heads=raw.get("num_key_value_heads", 8),
        head_dim=raw.get("head_dim", 128),
        intermediate_size=raw["intermediate_size"],
        vocab_size=raw["vocab_size"],
        max_position_embeddings=raw.get("max_position_embeddings", 40960),
        rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
        attention_bias=raw.get("attention_bias", False),
        attention_dropout=0.0,
        rope_theta=raw.get("rope_theta", 1000000.0),
        tie_word_embeddings=False,
    )
    cfg._attn_implementation = args.attention_backend
    cfg.layer_types = ["full_attention"] * raw["num_hidden_layers"]
    cfg.num_target_layers = raw["num_target_layers"]
    cfg.block_size = int(block_size)
    if kind == "dflash2":
        from specforge.modeling.draft.dflash2 import DFlash2DraftModel as DraftCls

        dc.setdefault("conv_kernel_size", 2)
        dc.setdefault("conv_group_size", 16)
        dc.setdefault("selector_rank", 256)
        dc.setdefault("selector_top_k", 16)
    else:
        from specforge.modeling.draft.dspark import DSparkDraftModel as DraftCls

        dc.setdefault("projector_type", "dspark")
        dc.setdefault("markov_rank", 256)
        dc.setdefault("markov_head_type", "vanilla")
        dc.setdefault("enable_confidence_head", True)
        dc.setdefault("confidence_head_alpha", 1.0)
        dc.setdefault("confidence_head_with_markov", True)
    cfg.dflash_config = dc
    mask_token_id = int(dc["mask_token_id"])

    torch.manual_seed(args.seed)
    draft = DraftCls(cfg).to(device=device, dtype=torch.bfloat16)
    draft.mask_token_id = mask_token_id
    hidden, vocab = int(cfg.hidden_size), int(cfg.vocab_size)
    torch.manual_seed(args.seed + 1)
    embed = nn.Embedding(vocab, hidden).to(device=device, dtype=torch.bfloat16)
    lm_head = nn.Linear(hidden, vocab, bias=False).to(device=device, dtype=torch.bfloat16)
    embed.requires_grad_(False)
    lm_head.requires_grad_(False)
    common = dict(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed,
        mask_token_id=mask_token_id,
        block_size=int(draft.block_size),
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        objective_chunk_blocks=args.objective_chunk_blocks,
    )
    if kind == "dflash2":
        model = OnlineDFlashModel(
            **common, loss_type="dflash", teacher_metrics=bool(args.teacher_metrics)
        )
    else:
        model = OnlineDSparkModel(
            **common, dspark_ce_loss_alpha=0.1, dspark_confidence_head_alpha=1.0
        )
    model = model.to(device=device, dtype=torch.bfloat16)
    width = len(draft.target_layer_ids) * hidden
    return model, {"hidden": hidden, "vocab": vocab, "width": width}


def make_dflash_batch(args, kind, rank, shapes):
    from specforge.runtime.contracts import TrainBatch

    g = torch.Generator().manual_seed(1234 + rank)
    B, S = args.batch, args.seq_len
    tensors = {
        "input_ids": torch.randint(0, shapes["vocab"], (B, S), generator=g),
        "loss_mask": torch.ones(B, S, dtype=torch.long),
        "hidden_states": torch.randn(B, S, shapes["width"], generator=g).to(
            torch.bfloat16
        ),
    }
    if kind == "dspark" or args.teacher_metrics:
        tensors["target_last_hidden_states"] = torch.randn(
            B, S, shapes["hidden"], generator=g
        ).to(torch.bfloat16)
    tensors = {k: v.pin_memory() for k, v in tensors.items()}
    return TrainBatch(
        sample_ids=[f"bench-{rank}-{i}" for i in range(B)],
        strategy="dflash" if kind == "dflash2" else "dspark",
        tensors=tensors,
        metadata={},
    )


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def _mb(x):
    return round(x / (1024**2), 1)


def _profile_summary(prof):
    sync_names = (
        "cudaStreamSynchronize",
        "cudaDeviceSynchronize",
        "cudaEventSynchronize",
        "cudaMemcpy",  # synchronous memcpy (D2H item() etc.)
    )
    sync_counts = {n: 0 for n in sync_names}
    comm = {"allgather": [0, 0.0], "reducescatter": [0, 0.0], "allreduce": [0, 0.0]}
    kernel_time = 0.0
    for evt in prof.events():
        name = evt.name
        is_cuda = str(getattr(evt, "device_type", "")).endswith("CUDA")
        if not is_cuda:
            for n in sync_names:
                if name == n:
                    sync_counts[n] += 1
            continue
        dur = float(getattr(evt, "self_device_time_total", 0.0) or 0.0)
        if dur <= 0:
            dur = float(getattr(evt, "device_time", 0.0) or 0.0)
        kernel_time += dur
        lname = name.lower()
        for key in comm:
            if key in lname.replace("_", ""):
                comm[key][0] += 1
                comm[key][1] += dur
    return {
        "host_syncs": sync_counts,
        "host_syncs_total": sum(sync_counts.values()),
        "nccl_kernels": {k: {"count": v[0], "ms": round(v[1] / 1000.0, 2)} for k, v in comm.items()},
        "device_kernel_ms_total": round(kernel_time / 1000.0, 2),
    }


def measure_checkpoint(backend, strategy, workdir, rank):
    """Time the full-state gather and the on-disk write (sync, and async when
    this checkout's CheckpointManager supports it)."""
    import inspect
    import shutil

    from specforge.training.checkpoint import CheckpointManager

    out = {}
    supports_async = "async_write" in inspect.signature(CheckpointManager).parameters
    variants = [("sync", {})] + ([("async", {"async_write": True})] if supports_async else [])
    for name, kwargs in variants:
        ckpt_root = os.path.join(workdir, f"ckpt_{name}")
        os.makedirs(ckpt_root, exist_ok=True)
        mgr = CheckpointManager(ckpt_root, "bench", **kwargs)
        timings = []
        for step in (1, 2):
            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            full = backend.state_dict()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            shared = None
            if rank == 0:
                shared = {
                    "draft_state_dict": strategy.checkpoint_state_filter(full["model"]),
                    "global_step": step,
                    "world_size": dist.get_world_size(),
                }
            rank_state = {"metadata": full.get("metadata"), "optimizer": full["optimizer"], "rng": full["rng"]}
            mgr.save(shared, step, rank_state=rank_state)
            t2 = time.perf_counter()
            if supports_async:
                mgr.wait()
            t3 = time.perf_counter()
            timings.append({"state_dict_s": t1 - t0, "save_blocking_s": t2 - t1, "save_total_s": t3 - t1})
            del full, shared, rank_state
        out[name] = {k: round(sum(t[k] for t in timings) / len(timings), 3) for k in timings[0]}
        size = 0
        for root, _, files in os.walk(ckpt_root):
            size += sum(os.path.getsize(os.path.join(root, f)) for f in files)
        out[name]["bytes_on_disk_mb"] = round(size / 1024**2, 1)
        shutil.rmtree(ckpt_root, ignore_errors=True)
    return out


def main():
    args = parse_args()
    if args.draft_config is None:
        args.draft_config = _default_draft_config(args.algo)
    os.environ.setdefault("SPECFORGE_DEVICE", "cuda")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from specforge.distributed import init_distributed

    init_distributed(timeout=60)
    rank, world = dist.get_rank(), dist.get_world_size()
    os.makedirs(args.out_dir, exist_ok=True)
    label = args.label or args.backend
    out_path = os.path.join(args.out_dir, f"{args.algo}-{label}-rank{rank}.json")
    result = {
        "algo": args.algo,
        "backend": args.backend,
        "label": label,
        "sharding": args.sharding,
        "world_size": world,
        "rank": rank,
        "args": vars(args),
        "torch": torch.__version__,
        "device_name": torch.cuda.get_device_name(local_rank),
    }
    workdir = tempfile.mkdtemp(prefix=f"bench_{args.algo}_{args.backend}_")
    try:
        from specforge.optimizer import BF16Optimizer
        from specforge.training.backend import ParallelConfig, create_training_backend
        from specforge.training.controller import TrainerCore
        from specforge.training.strategies.base import (
            DFlashTrainStrategy,
            DSparkTrainStrategy,
            Eagle3TrainStrategy,
            StepContext,
        )

        torch.cuda.reset_peak_memory_stats()
        if args.algo == "eagle3":
            model, head, shapes = build_eagle3(args, device, workdir, rank)
            batch = make_eagle3_batch(args, device, workdir, rank, shapes)
        else:
            model, shapes = build_dflash_family(args, device, args.algo)
            head = None
            batch = make_dflash_batch(args, args.algo, rank, shapes)
        torch.cuda.synchronize()
        result["mem_model_built_alloc_mb"] = _mb(torch.cuda.memory_allocated())
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        result["params_total_m"] = round(n_params / 1e6, 1)
        result["params_trainable_m"] = round(n_trainable / 1e6, 1)

        def opt_factory(m):
            return BF16Optimizer(
                m, lr=1e-4, max_grad_norm=0.5, warmup_ratio=0.0, total_steps=10_000
            )

        pc = ParallelConfig.from_distributed(sharding_strategy=args.sharding)
        backend_kwargs = {}
        requested = {
            "compile_blocks": bool(args.compile_blocks),
            "fp8_linear": bool(args.fp8_linear),
            "shard_frozen_tables": bool(args.shard_frozen_tables),
        }
        if any(requested.values()):
            # Each FSDP2 option lands in its own PR, so only pass the fields
            # this checkout's BackendOptions actually defines.
            from specforge.training.backend import BackendOptions

            fields = BackendOptions.__dataclass_fields__
            missing = [k for k, v in requested.items() if v and k not in fields]
            if missing:
                raise SystemExit(f"BackendOptions in this checkout lacks {missing}")
            backend_kwargs["options"] = BackendOptions(
                **{k: v for k, v in requested.items() if k in fields}
            )
        backend = create_training_backend(
            args.backend, pc, optimizer_factory=opt_factory, **backend_kwargs
        )
        wrapped = backend.prepare_model(model, optimizer_target=model.draft_model)
        torch.cuda.synchronize()
        dist.barrier()
        result["mem_after_wrap_alloc_mb"] = _mb(torch.cuda.memory_allocated())
        result["mem_after_wrap_reserved_mb"] = _mb(torch.cuda.memory_reserved())

        if args.algo == "eagle3":
            strategy = Eagle3TrainStrategy(wrapped, target_head=head)
        elif args.algo == "dflash2":
            strategy = DFlashTrainStrategy(wrapped)
        else:
            strategy = DSparkTrainStrategy(wrapped)
        core = TrainerCore(strategy, backend, accumulation_steps=args.accum)

        total_steps = (args.warmup + args.steps + 1) * args.accum
        step_idx = 0

        def one_optimizer_step():
            nonlocal step_idx
            rep = None
            for _ in range(args.accum):
                ctx = StepContext(
                    global_step=step_idx,
                    total_steps=total_steps,
                    collect_detailed_metrics=bool(args.detailed_metrics),
                )
                rep = core.train_step(batch, ctx)
                step_idx += 1
            return rep

        # Warmup (compile, allocator, NCCL communicators).
        t0 = time.perf_counter()
        for _ in range(args.warmup):
            rep = one_optimizer_step()
        torch.cuda.synchronize()
        dist.barrier()
        result["warmup_s"] = round(time.perf_counter() - t0, 2)
        result["first_loss"] = float(rep.loss) if rep is not None and rep.loss is not None else None
        result["first_grad_norm"] = (
            float(rep.grad_norm) if rep is not None and rep.grad_norm is not None else None
        )

        torch.cuda.reset_peak_memory_stats()
        step_times = []
        micro_times = []
        for _ in range(args.steps):
            dist.barrier()
            torch.cuda.synchronize()
            t_step = time.perf_counter()
            for _ in range(args.accum):
                t_m = time.perf_counter()
                ctx = StepContext(
                    global_step=step_idx,
                    total_steps=total_steps,
                    collect_detailed_metrics=bool(args.detailed_metrics),
                )
                rep = core.train_step(batch, ctx)
                step_idx += 1
                torch.cuda.synchronize()
                micro_times.append(time.perf_counter() - t_m)
            step_times.append(time.perf_counter() - t_step)
        torch.cuda.synchronize()
        result["optimizer_step_s_mean"] = round(statistics.mean(step_times), 4)
        result["optimizer_step_s_min"] = round(min(step_times), 4)
        result["microstep_ms_mean"] = round(1000 * statistics.mean(micro_times), 2)
        result["microstep_ms_median"] = round(1000 * statistics.median(micro_times), 2)
        result["samples_per_s_per_gpu"] = round(
            args.batch * args.accum / statistics.mean(step_times), 3
        )
        result["peak_alloc_mb"] = _mb(torch.cuda.max_memory_allocated())
        result["peak_reserved_mb"] = _mb(torch.cuda.max_memory_reserved())
        result["last_loss"] = float(rep.loss) if rep.loss is not None else None
        result["last_grad_norm"] = float(rep.grad_norm) if rep.grad_norm is not None else None

        if args.measure_checkpoint:
            result["checkpoint"] = measure_checkpoint(backend, strategy, workdir, rank)

        if args.profile:
            from torch.profiler import ProfilerActivity, profile

            dist.barrier()
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                one_optimizer_step()
                torch.cuda.synchronize()
            result["profile"] = _profile_summary(prof)
        result["ok"] = True
    except Exception as exc:  # noqa: BLE001
        result["ok"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    finally:
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        if rank == 0:
            print(json.dumps({k: v for k, v in result.items() if k != "traceback"}, indent=2))
            if not result.get("ok"):
                print(result.get("traceback"))
        try:
            dist.barrier()
        except Exception:  # noqa: BLE001
            pass
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
