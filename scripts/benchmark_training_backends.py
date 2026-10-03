#!/usr/bin/env python3
"""Compare native TorchTitan Trainer and SpecForge FSDP1 on cached features.

Run this script from a fresh torchrun process for every trial. --specforge-root
selects the checkout containing the native integration, independently of this
benchmark-only checkout. The torchtitan backend executes Trainer.train() and
its inherited training step, backward, optimizer and scheduler. It does not
exercise the earlier TorchTitan FSDP helper backend.

At the same data-parallel degree, both engines use identical initial values,
frozen target tables, cached random features and CPU-sampled anchors. Changing
the DP/TP mesh changes the per-DP-rank feature and anchor streams, even when
accumulation preserves the global batch size. BF16 compute is shared, but native Titan keeps
FP32 parameters/Adam state and FP32 gradient reductions; the legacy optimizer
keeps BF16 parameters/reductions and FP32 masters/Adam state. This is a runtime
comparison, not a claim of bitwise optimizer or convergence equivalence.

Timed windows include feature preparation, H2D for token IDs/masks, forward,
backward and optimizer. Hidden states are already cached on GPU for both paths.
Metadata hashing, construction, checkpoint/export and trial coordination are
outside measured windows. First-step and warmup times are recorded separately
from steady steps, including lazy compilation in those warmup measurements.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import training_benchmark_recipes as recipes


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specforge-root", required=True, type=Path)
    parser.add_argument("--backend", choices=("fsdp", "torchtitan"), required=True)
    parser.add_argument(
        "--algorithm", choices=("dflash", "dflash2", "dspark"), required=True
    )
    parser.add_argument(
        "--recipe", choices=("controlled", "stock"), default="controlled"
    )
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument(
        "--attention",
        choices=("eager", "sdpa", "flex_attention"),
        default="flex_attention",
    )
    parser.add_argument("--seq-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--num-anchors", type=int, default=512)
    parser.add_argument("--objective-chunk-blocks", type=int, default=128)
    parser.add_argument("--cache-batches", type=int, default=2)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--cuda-graphs", action="store_true")
    parser.add_argument(
        "--titan-engine", choices=("trainer", "graph"), default="trainer"
    )
    parser.add_argument(
        "--graph-inductor", choices=("regional", "full"), default="regional"
    )
    parser.add_argument(
        "--sharding", choices=("SHARD_GRAD_OP", "FULL_SHARD"), default="SHARD_GRAD_OP"
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    for key in (
        "seq_length",
        "batch_size",
        "accumulation_steps",
        "num_anchors",
        "cache_batches",
        "steps",
        "tp_size",
    ):
        if getattr(args, key) < 1:
            parser.error(f"{key} must be positive")
    if args.warmup_steps < 1 or args.objective_chunk_blocks < 0:
        parser.error(
            "At least one warmup step and nonnegative objective chunk size are required"
        )
    if args.backend == "fsdp" and (
        args.tp_size != 1
        or args.compile
        or args.cuda_graphs
        or args.titan_engine != "trainer"
    ):
        parser.error("The FSDP baseline supports TP1 without compile")
    if args.titan_engine == "graph" and (not args.compile or args.tp_size != 1):
        parser.error("GraphTrainer requires --compile and TP1")
    if args.titan_engine != "graph" and args.graph_inductor != "regional":
        parser.error("--graph-inductor requires --titan-engine=graph")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning_rate must be finite and positive")
    args.feature_file = None
    args.activation_checkpointing = False
    args.detailed_metrics = False
    args.precision = "bf16"
    return args


class Measurement:
    def __init__(self, args, device):
        self.args, self.device = args, device
        self.seconds, self.losses = [], []
        self.peak = None

    def start(self, step):
        import torch
        import torch.distributed as dist

        dist.barrier()
        torch.cuda.synchronize()
        if step == self.args.warmup_steps:
            torch.cuda.reset_peak_memory_stats()
            self.baseline = torch.cuda.memory_allocated()
        return time.perf_counter()

    def finish(
        self, started, loss, *, local_loss_sum=False, global_mean=False, group=None
    ):
        import torch
        import torch.distributed as dist

        # Stop the clock before reductions used only to record the benchmark.
        torch.cuda.synchronize()
        duration = time.perf_counter() - started
        elapsed = torch.tensor(duration, dtype=torch.float64, device=self.device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        self.seconds.append(elapsed.item())
        if callable(loss):
            loss = loss()
        if local_loss_sum:
            loss = torch.stack(loss).sum()
            if group is not None:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=group)
            loss = loss.item()
        if global_mean:
            loss = torch.tensor(float(loss), device=self.device)
            dist.all_reduce(loss, op=dist.ReduceOp.AVG, group=group)
            loss = loss.item()
        if not math.isfinite(float(loss)):
            raise FloatingPointError("Non-finite benchmark loss")
        self.losses.append(float(loss))

    def result(self, cache_bytes, data_degree):
        import torch
        import torch.distributed as dist

        memory = torch.tensor(
            [
                self.baseline,
                torch.cuda.max_memory_allocated(),
                torch.cuda.max_memory_reserved(),
                cache_bytes,
            ],
            dtype=torch.int64,
            device=self.device,
        )
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        steady = self.seconds[self.args.warmup_steps :]
        tokens = (
            self.args.batch_size
            * self.args.seq_length
            * self.args.accumulation_steps
            * data_degree
        )
        return {
            "warmup_step_seconds_max_rank": self.seconds[: self.args.warmup_steps],
            "first_step_seconds_max_rank": self.seconds[0],
            "step_seconds_max_rank": steady,
            "losses_including_warmup": self.losses,
            "baseline_allocated_bytes_max_rank": memory[0].item(),
            "peak_allocated_bytes_max_rank": memory[1].item(),
            "peak_reserved_bytes_max_rank": memory[2].item(),
            "feature_cache_gpu_bytes_max_rank": memory[3].item(),
            "input_context_tokens_per_optimizer_window": tokens,
            **recipes.summarize_times(steady, tokens),
        }


def anchor_plan_hash(model, batches, args, data_rank):
    import torch

    generator = torch.Generator().manual_seed(args.seed + data_rank)
    digest = hashlib.sha256()
    for index in range((args.warmup_steps + args.steps) * args.accumulation_steps):
        mask = batches[index % len(batches)].tensors["loss_mask"]
        anchors, keep = model._sample_anchor_positions(
            mask.shape[1], mask, torch.device("cpu"), generator=generator
        )
        digest.update(
            recipes._tensor_fingerprint({"anchors": anchors, "keep": keep}).encode()
        )
    return digest.hexdigest()


def run_fsdp(args, model, draft, batches, measurement):
    import torch
    import torch.distributed as dist

    from specforge.distributed import init_distributed
    from specforge.optimizer import BF16Optimizer
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
    from specforge.training.controller import TrainerCore
    from specforge.training.strategies.base import (
        DFlashTrainStrategy,
        DSparkTrainStrategy,
        StepContext,
    )

    init_distributed()
    model.to(device=measurement.device, dtype=torch.bfloat16)
    parallel = ParallelConfig.from_distributed(
        sharding_strategy=args.sharding, param_dtype=torch.bfloat16
    )
    if parallel.sharding_strategy != args.sharding:
        raise ValueError("FSDP_SHARDING conflicts with the benchmark sharding argument")
    backend = FSDPTrainingBackend(
        parallel,
        optimizer_factory=lambda module: BF16Optimizer(
            module,
            lr=args.learning_rate,
            weight_decay=0.0,
            max_grad_norm=1.0,
            total_steps=100000,
            warmup_ratio=0.0,
            lr_scheduler="constant",
        ),
    )
    prepared = {}

    def inject_anchors(module, positional, kwargs):
        kwargs.update(prepared)
        return positional, kwargs

    model.register_forward_pre_hook(inject_anchors, with_kwargs=True)
    wrapped = backend.prepare_model(model, optimizer_target=draft)
    strategy = (
        DSparkTrainStrategy(wrapped)
        if args.algorithm == "dspark"
        else DFlashTrainStrategy(wrapped)
    )
    core = TrainerCore(strategy, backend, accumulation_steps=args.accumulation_steps)
    generator = torch.Generator().manual_seed(args.seed + dist.get_rank())

    def train():
        for step in range(args.warmup_steps + args.steps):
            started = measurement.start(step)
            micro_results = []
            for micro in range(args.accumulation_steps):
                batch = batches[(step * args.accumulation_steps + micro) % len(batches)]
                mask = batch.tensors["loss_mask"]
                anchors, keep = model._sample_anchor_positions(
                    mask.shape[1], mask, torch.device("cpu"), generator=generator
                )
                prepared.update(
                    anchor_positions=anchors.to(measurement.device),
                    block_keep_mask=keep.to(measurement.device),
                )
                result = core.train_step(
                    batch,
                    StepContext(
                        global_step=step,
                        total_steps=100000,
                        collect_detailed_metrics=step == 0,
                    ),
                )
                micro_results.append(result)
            if not result.optimizer_stepped or core.accumulation_remainder:
                raise AssertionError("FSDP optimizer window did not complete")
            # DFlash reports the pooled window objective; DSpark's existing
            # strategy reports a logical-microbatch objective. Materialize all
            # host metrics after stopping the timing window in either case.
            measurement.finish(
                started,
                lambda: (
                    sum(item.loss for item in micro_results) / args.accumulation_steps
                    if args.algorithm == "dspark"
                    else result.loss
                ),
                global_mean=args.algorithm == "dspark",
            )

    return train, None


def run_titan(args, config, model, batches, measurement, directory):
    import torch
    from torchtitan.components.checkpoint import CheckpointManager
    from torchtitan.components.metrics import MetricsProcessor
    from torchtitan.components.optimizer.optimizer import default_adamw
    from torchtitan.config import (
        CompileConfig,
        DebugConfig,
        ParallelismConfig,
        TrainingConfig,
    )
    from torchtitan.protocols.model_spec import ModelSpec
    from torchtitan.tools.logging import init_logger

    from specforge.training.torchtitan.data import FeatureDataLoader
    from specforge.training.torchtitan.frontend import (
        _scheduler_config,
        _tokenizer_config,
    )
    from specforge.training.torchtitan.loss import SpecForgeObjectiveLoss
    from specforge.training.torchtitan.model import SpecForgeTitanModel
    from specforge.training.torchtitan.parallelize import (
        group_optimizer_parameters_by_mesh,
        parallelize_dflash,
    )
    from specforge.training.torchtitan.runtime import SpecForgeTitanTrainer

    trainer_type = SpecForgeTitanTrainer
    model_config = SpecForgeTitanModel.Config
    parallelize_fn = parallelize_dflash
    compile_options = CompileConfig(enable=args.compile, components=["model"])
    if args.titan_engine == "graph":
        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )

        from specforge.training.torchtitan.graph import (
            SpecForgeGraphModel,
            SpecForgeGraphTrainer,
        )
        from specforge.training.torchtitan.graph_parallelize import (
            parallelize_graph_dflash,
        )

        trainer_type = SpecForgeGraphTrainer
        model_config = SpecForgeGraphModel.Config
        parallelize_fn = parallelize_graph_dflash
        compile_options = GraphTrainerCompileConfig(
            enable=True,
            components=["model"],
            inductor_compilation=args.graph_inductor,
            disable_passes=[] if args.cuda_graphs else ["cudagraph_pass"],
        )

    teacher_path, draft_path = (
        Path(directory) / "teacher.pt",
        Path(directory) / "draft.pt",
    )
    torch.save(
        {
            "lm_head.weight": model.lm_head.weight,
            "embed_tokens.weight": model.embed_tokens.weight,
        },
        teacher_path,
    )
    torch.save(model.draft_model.state_dict(), draft_path)
    total_steps = args.warmup_steps + args.steps
    objective = dict(
        mask_token_id=config["dflash_config"]["mask_token_id"],
        block_size=model.block_size,
        attention_backend=args.attention,
        num_anchors=args.num_anchors,
        objective_chunk_blocks=args.objective_chunk_blocks,
    )
    if args.algorithm != "dspark":
        objective["teacher_metrics"] = False

    class CachedSource:
        def __call__(self, **kwargs):
            expected_rank = int(os.environ["RANK"]) // args.tp_size
            if (
                kwargs["dp_rank"] != expected_rank
                or kwargs["local_batch_size"] != args.batch_size
            ):
                raise AssertionError("Unexpected native data mesh")
            return (
                batches[index % len(batches)]
                for index in range(total_steps * args.accumulation_steps)
            )

    class TimedTrainer(trainer_type):
        def forward_backward_step(self, **kwargs):
            loss = super().forward_backward_step(**kwargs)
            # Record outside capture: Python callbacks inside loss_fn do not
            # execute on replay. Clone because graph output storage is reused
            # by subsequent accumulation microbatches.
            self.recorded_losses.append(loss.detach().clone())
            return loss

        def train_step(self, data_iterator):
            self.recorded_losses = []
            started = measurement.start(self.step - 1)
            super().train_step(data_iterator)
            group = self.parallel_dims.get_optional_mesh(
                "batch", include_singleton_axes=True
            ).get_group()
            measurement.finish(
                started, self.recorded_losses, local_loss_sum=True, group=group
            )

    native = trainer_type.Config(
        dump_folder=str(Path(directory) / "native"),
        model_spec=ModelSpec(
            name="specforge",
            flavor=args.algorithm,
            model=model_config(
                draft_config=config,
                algorithm=args.algorithm,
                objective=objective,
                teacher_state_path=str(teacher_path),
                draft_state_path=str(draft_path),
                init_seed=args.seed,
            ),
            parallelize_fn=parallelize_fn,
            pipelining_fn=None,
            post_optimizer_build_fn=group_optimizer_parameters_by_mesh,
            state_dict_adapter=None,
        ),
        tokenizer=_tokenizer_config(config["vocab_size"]),
        dataloader=FeatureDataLoader.Config(
            source_factory=CachedSource(),
            epochs=1,
            pad_to_seq_len=args.cuda_graphs or args.titan_engine == "graph",
        ),
        loss=SpecForgeObjectiveLoss.Config(algorithm=args.algorithm),
        optimizer=default_adamw(
            lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        ),
        lr_scheduler=_scheduler_config(
            SimpleNamespace(
                training=SimpleNamespace(warmup_ratio=0.0, lr_scheduler="constant")
            ),
            100000,
        ),
        training=TrainingConfig(
            local_batch_size=args.batch_size,
            global_batch_size=args.batch_size
            * (int(os.environ["WORLD_SIZE"]) // args.tp_size)
            * args.accumulation_steps,
            seq_len=args.seq_length,
            steps=total_steps,
            max_norm=1.0,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            disable_cuda_graphs=args.titan_engine == "graph" or not args.cuda_graphs,
        ),
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=args.tp_size,
            enable_sequence_parallel=False,
            spmd_backend="partial_dtensor",
            fsdp_reshard_after_forward=(
                "always" if args.sharding == "FULL_SHARD" else "never"
            ),
        ),
        activation_checkpoint=None,
        compile=compile_options,
        checkpoint=CheckpointManager.Config(enable=False),
        metrics=MetricsProcessor.Config(log_freq=1000000000),
        debug=DebugConfig(seed=args.seed),
        schedule_total_steps=100000,
    )
    init_logger()
    trainer = TimedTrainer(native)
    return trainer.train, trainer


def main():
    args = parse_args()
    sys.path.insert(0, str(args.specforge_root.resolve()))
    import torch
    import torch.distributed as dist

    if "RANK" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("Launch this CUDA benchmark with torchrun")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world % args.tp_size:
        raise ValueError("WORLD_SIZE must be divisible by TP")
    data_rank, data_degree = rank // args.tp_size, world // args.tp_size
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_files = [
        "specforge/training/backend.py",
        "specforge/training/controller.py",
        "specforge/optimizer.py",
        "specforge/algorithms/common/dflash_family_model.py",
        "specforge/training/torchtitan/runtime.py",
        "specforge/training/torchtitan/frontend.py",
        "specforge/training/torchtitan/model.py",
        "specforge/training/torchtitan/parallelize.py",
        "specforge/training/torchtitan/data.py",
        "specforge/training/torchtitan/loss.py",
        "specforge/training/torchtitan/metrics.py",
        "specforge/training/torchtitan/graph.py",
        "specforge/training/torchtitan/graph_parallelize.py",
    ]

    def source_hashes():
        return {
            name: hashlib.sha256((args.specforge_root / name).read_bytes()).hexdigest()
            for name in source_files
        }

    source_before = source_hashes()
    config, source = recipes.resolve_config(args.algorithm, args.recipe, tiny=args.tiny)
    trainer, failed = None, False
    started = time.perf_counter()
    try:
        with TemporaryDirectory(
            prefix=".native-benchmark-", dir=args.output.parent
        ) as directory:
            model, draft, counts, draft_hash, teacher_hash = recipes._build_model(
                args, config, "cpu", torch.bfloat16
            )
            batches, feature_hashes, cache_bytes = recipes._make_batches(
                args, config, data_rank, device, torch.bfloat16
            )
            anchors_hash = anchor_plan_hash(model, batches, args, data_rank)
            measurement = Measurement(args, device)
            if args.backend == "fsdp":
                train, trainer = run_fsdp(args, model, draft, batches, measurement)
            else:
                train, trainer = run_titan(
                    args, config, model, batches, measurement, directory
                )
                del model, draft
                gc.collect()
            build_seconds = torch.tensor(
                time.perf_counter() - started, device=device, dtype=torch.float64
            )
            dist.all_reduce(build_seconds, op=dist.ReduceOp.MAX)
            identity = {
                "data_rank": data_rank,
                "features": feature_hashes,
                "anchors": anchors_hash,
            }
            identities = [None] * world
            dist.all_gather_object(identities, identity)
            train()
            result = measurement.result(cache_bytes, data_degree)
            if source_hashes() != source_before:
                raise RuntimeError("Benchmark source changed during this trial")
            payload = {
                "benchmark": "native-torchtitan-v1",
                "backend": args.backend,
                "synthetic_features": True,
                "trainer_build_seconds_max_rank_including_cpu_init_and_hashes": build_seconds.item(),
                "recipe_source": source,
                "counts": counts,
                "comparison_contract": {
                    "config": config,
                    "seed": args.seed,
                    "world_size": world,
                    "data_degree": data_degree,
                    "batch_size": args.batch_size,
                    "accumulation_steps": args.accumulation_steps,
                    "seq_length": args.seq_length,
                    "num_anchors": args.num_anchors,
                    "objective_chunk_blocks": args.objective_chunk_blocks,
                    "attention": args.attention,
                    "sharding": args.sharding,
                    "learning_rate": args.learning_rate,
                    "initial_draft_sha256": draft_hash,
                    "frozen_target_sha256": teacher_hash,
                    "features_and_anchors_by_rank": identities,
                    "warmup_steps": args.warmup_steps,
                    "steps": args.steps,
                    "feature_residency": "pinned CPU input IDs/loss masks; cached GPU hidden states",
                },
                "runtime": {
                    "tp_size": args.tp_size,
                    "compile": args.compile,
                    "cuda_graphs": args.cuda_graphs,
                    "titan_engine": args.titan_engine,
                    "distributed_wrapper": (
                        "SimpleFSDP"
                        if args.titan_engine == "graph"
                        else "FSDP2" if args.backend == "torchtitan" else "FSDP1"
                    ),
                    "graph_memory_policy": (
                        trainer.config.compile.memory_policy
                        if args.titan_engine == "graph"
                        else None
                    ),
                    "fsdp_reshard_after_forward": (
                        trainer.config.parallelism.fsdp_reshard_after_forward
                        if trainer is not None
                        else None
                    ),
                    "graph_inductor": (
                        args.graph_inductor if args.titan_engine == "graph" else None
                    ),
                    "parameter_storage": (
                        "float32"
                        if args.backend == "torchtitan"
                        else "bfloat16+float32-master"
                    ),
                    "gradient_reduce_dtype": (
                        "float32" if args.backend == "torchtitan" else "bfloat16"
                    ),
                    "compute_dtype": "bfloat16",
                    "adam_state_dtype": "float32",
                    "kernel_environment": {
                        key: os.environ.get(key, "1")
                        for key in (
                            "SPECFORGE_DFLASH_FUSED_HEAD",
                            "SPECFORGE_DFLASH2_FUSED_CONV",
                        )
                    },
                },
                "versions": {
                    name: importlib.metadata.version(name)
                    for name in ("torch", "torchtitan", "transformers", "triton")
                },
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(),
                "source_sha256": source_before,
                "recipe_helpers_sha256": hashlib.sha256(
                    Path(recipes.__file__).read_bytes()
                ).hexdigest(),
                "benchmark_sha256": hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                **result,
            }
            if rank == 0:
                args.output.write_text(json.dumps(payload, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            "output": str(args.output),
                            **recipes.summarize_times(
                                result["step_seconds_max_rank"],
                                result["input_context_tokens_per_optimizer_window"],
                            ),
                        },
                        indent=2,
                    )
                )
    except BaseException:
        failed = True
        import traceback

        traceback.print_exc()
        if dist.is_initialized():
            abort = getattr(dist.distributed_c10d, "_abort_process_group", None)
            if abort is not None:
                abort()
        raise
    finally:
        if not failed:
            if trainer is not None:
                trainer.close()
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
