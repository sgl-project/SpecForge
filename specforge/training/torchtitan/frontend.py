"""Public offline-training bridge to the native TorchTitan Trainer.

No SpecForge process groups are initialized here. TorchTitan owns the model
mesh, optimization loop, distributed checkpoints, and trainer lifecycle.
Optional TorchTitan imports stay behind the configured backend boundary.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import unquote


def _numerical_resume_contract(titan) -> dict[str, str]:
    # Pure eager native training keeps its original arithmetic. Compilation now
    # preserves eager BF16 cast/division boundaries, and DFlash2 can retain its
    # fused convolution while compiling. Old compiled checkpoints must not
    # silently continue under this different numerical policy.
    if not titan.compile:
        return {}
    contract = {"compiler_numerics": "bf16-eager-boundaries-v1"}
    if titan.engine == "graph":
        contract["graph_parameter_materialization"] = "shared-bf16-per-joint-v1"
        contract["graph_inductor"] = titan.graph_inductor
    return contract


def _validate_runtime_versions() -> None:
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import Version

    instructions = (
        "See https://github.com/sgl-project/SpecForge/blob/main/"
        "docs/sections/basic_usage/torchtitan.md for the isolated runtime setup."
    )
    try:
        titan, torch = version("torchtitan"), version("torch")
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"Native TorchTitan requires torchtitan==0.3.0 and torch>=2.14. {instructions}"
        ) from exc
    if Version(titan) != Version("0.3.0") or Version(torch) < Version("2.14"):
        raise RuntimeError(
            "Native TorchTitan requires torchtitan==0.3.0 and torch>=2.14; "
            f"found torchtitan={titan}, torch={torch}. {instructions}"
        )


def _abort_titan_process_groups() -> None:
    """Failure teardown must not wait for a peer stuck in a collective."""
    import torch.distributed as dist

    if not dist.is_initialized():
        return
    abort = getattr(dist.distributed_c10d, "_abort_process_group", None)
    if callable(abort):
        try:
            abort()
        except Exception:
            logging.getLogger(__name__).exception(
                "TorchTitan process-group abort failed"
            )
    # Never fall back to collective destroy on the failure path. torchrun
    # observes this rank's original exception and terminates its peer workers.


def _parallelism_layout(cfg, world_size: int) -> tuple[dict[str, Any], int]:
    training = cfg.training
    titan = training.torchtitan
    model_degree = training.tp_size * titan.cp_size * titan.pp_size
    if world_size < 1 or world_size % model_degree:
        raise ValueError("WORLD_SIZE must be divisible by TorchTitan TP * CP * PP")
    data_degree = world_size // model_degree
    replicate, shard = titan.dp_replicate, titan.dp_shard
    if training.fsdp_sharding == "NO_SHARD":
        if shard not in (-1, 1) or replicate not in (1, data_degree):
            raise ValueError(
                "NO_SHARD requires dp_shard=1 and all data ranks replicated"
            )
        replicate, shard = data_degree, 1
    elif shard == -1:
        if data_degree % replicate:
            raise ValueError("Data parallel degree must be divisible by dp_replicate")
        shard = data_degree // replicate
    if replicate * shard != data_degree:
        raise ValueError("TorchTitan DP * TP * CP * PP must equal WORLD_SIZE")
    return {
        "data_parallel_replicate_degree": replicate,
        "data_parallel_shard_degree": shard,
        "tensor_parallel_degree": training.tp_size,
        "context_parallel_degree": titan.cp_size,
        "pipeline_parallel_degree": titan.pp_size,
        "pipeline_parallel_schedule": titan.pp_schedule,
        "pipeline_parallel_microbatch_size": titan.pp_microbatch_size,
        "enable_sequence_parallel": False,
        "spmd_backend": titan.spmd_backend,
        "fsdp_reshard_after_forward": (
            "always" if training.fsdp_sharding == "FULL_SHARD" else "never"
        ),
    }, data_degree


def _training_horizon(cfg, source_size: int, data_degree: int) -> tuple[int, int]:
    from specforge.training.schedule import (
        resolve_total_steps,
        validate_fixed_accumulation_plan,
    )

    training = cfg.training
    # Match the existing DistributedSampler padding and per-rank drop_last.
    local_samples = math.ceil(source_size / data_degree)
    validate_fixed_accumulation_plan(
        num_samples=local_samples,
        batch_size=training.batch_size,
        accumulation_steps=training.accumulation_steps,
        num_epochs=training.num_epochs,
        max_steps=training.max_steps,
    )
    natural_steps = (
        (local_samples // training.batch_size) * training.num_epochs
    ) // training.accumulation_steps
    if natural_steps < 1:
        raise ValueError("The offline feature dataset produces no optimizer step")
    stop_steps = min(natural_steps, training.max_steps or natural_steps)
    schedule_steps = resolve_total_steps(
        total_steps=training.total_steps,
        max_steps=training.max_steps,
        num_samples=local_samples,
        batch_size=training.batch_size,
        accumulation_steps=training.accumulation_steps,
        num_epochs=training.num_epochs,
    )
    return stop_steps, schedule_steps


def _objective_options(cfg, variant: str, draft) -> dict[str, Any]:
    training = cfg.training
    result = {
        "mask_token_id": draft.mask_token_id,
        "block_size": int(draft.block_size),
        "attention_backend": training.attention_backend,
        "num_anchors": training.num_anchors,
        "loss_decay_gamma": training.loss_decay_gamma,
        "objective_chunk_blocks": training.objective_chunk_blocks,
    }
    if variant == "dspark":
        result.update(
            dspark_ce_loss_alpha=training.dspark_ce_loss_alpha,
            dspark_l1_loss_alpha=training.dspark_l1_loss_alpha,
            dspark_confidence_head_alpha=training.dspark_confidence_head_alpha,
        )
    else:
        result.update(
            loss_type=training.loss_type,
            dpace_alpha=training.dpace_alpha,
            selector_loss_alpha=training.dflash2_selector_loss_alpha,
            selector_warmup_ratio=training.dflash2_selector_warmup_ratio,
            selector_ramp_ratio=training.dflash2_selector_ramp_ratio,
            selector_stop_gradient=training.dflash2_selector_stop_gradient,
            lk_loss_type=training.lk_loss_type,
            kl_scale=training.kl_scale,
            kl_decay=training.kl_decay,
            teacher_metrics=training.dflash_teacher_metrics,
        )
    return result


def _snapshot_model_sources(cfg, algorithm, directory: str):
    """Resolve real frozen teacher weights and warm starts without touching CUDA."""
    import torch

    from specforge.algorithms.common.dflash_family_model import (
        OnlineDFlashModel,
        OnlineDSparkModel,
    )
    from specforge.algorithms.common.providers import checkpoint_key_fingerprint
    from specforge.modeling.auto import AutoDraftModel
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
    from specforge.training.assembly import _load_input_tools
    from specforge.training.model_loading import (
        resolve_draft_config,
        warm_start_draft_model,
    )
    from specforge.training.model_utils import resolve_mask_token_id
    from specforge.training.provenance import model_resume_provenance

    provider = algorithm.providers.model
    draft_config = resolve_draft_config(cfg, provider=provider.draft_config)
    draft_config._attn_implementation = cfg.training.attention_backend
    architecture = draft_config.architectures[0]
    variants = {
        "DFlashDraftModel": "dflash",
        "DFlash2DraftModel": "dflash2",
        "DSparkDraftModel": "dspark",
    }
    if architecture not in variants:
        raise ValueError(
            f"TorchTitan does not support draft architecture {architecture!r}"
        )
    variant = variants[architecture]
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.manual_seed(cfg.training.seed)
        draft = AutoDraftModel.from_config(draft_config, torch_dtype=torch.bfloat16)
        if cfg.model.draft_checkpoint_path:
            warm_start_draft_model(
                draft,
                cfg.model.draft_checkpoint_path,
                draft_config=draft_config,
                strategy=algorithm.name,
                cache_dir=cfg.model.cache_dir,
                trust_remote_code=cfg.model.trust_remote_code,
            )
        tokenizer = (
            _load_input_tools(cfg, algorithm)
            if provider.needs_input_tools(cfg, draft)
            else None
        )
        draft.mask_token_id = resolve_mask_token_id(
            explicit=cfg.model.mask_token_id,
            tokenizer=tokenizer,
            embedding_vocab_size=int(draft_config.vocab_size),
            draft_model=draft,
        )
        method_config = dict(getattr(draft_config, "dflash_config", None) or {})
        method_config.update(
            mask_token_id=draft.mask_token_id,
            target_layer_ids=list(draft.target_layer_ids),
        )
        draft_config.dflash_config = method_config
        teacher = TargetEmbeddingsAndHead.from_pretrained(
            cfg.model.target_model_path,
            embed_key=cfg.model.embedding_key,
            lm_head_key=cfg.model.lm_head_key,
            cache_dir=cfg.model.cache_dir,
            device="cpu",
            dtype=torch.bfloat16,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        if tuple(teacher.embed_tokens.weight.shape) != (
            draft_config.vocab_size,
            draft_config.hidden_size,
        ):
            raise ValueError(
                "TorchTitan draft and teacher embedding dimensions must match"
            )
        objective = _objective_options(cfg, variant, draft)
        training_class = OnlineDSparkModel if variant == "dspark" else OnlineDFlashModel
        composite = training_class(
            draft_model=draft,
            target_lm_head=teacher.lm_head,
            target_embed_tokens=teacher.embed_tokens,
            **objective,
        )
        provenance = model_resume_provenance(
            cfg,
            draft_config,
            teacher.config,
            capture_layers=list(draft.target_layer_ids),
        )
        contract = dict(
            algorithm.providers.step.bind_runtime(
                cfg, draft, composite, model_provenance=provenance
            ).resume_contract
        )
        contract["frozen_teacher_sha256"] = checkpoint_key_fingerprint(
            teacher, {"lm_head.weight", "embed_tokens.weight"}
        )
        teacher_path = str(Path(directory) / "teacher.pt")
        draft_path = str(Path(directory) / "draft.pt")
        torch.save(teacher.state_dict(), teacher_path)
        # Native Titan owns FP32 parameters and Adam state, with BF16 forward
        # materialization. Preserve the provider's actual initial weight values.
        torch.save(draft.float().state_dict(), draft_path)
    return draft_config, variant, objective, teacher_path, draft_path, contract


@dataclass
class _OfflineSourceFactory:
    cfg: Any
    algorithm: Any
    refs: list

    def __call__(self, *, dp_rank, dp_world_size, local_batch_size, seq_len, epoch=0):
        from specforge.launch import _shard_offline_refs
        from specforge.runtime.data_plane import FeatureDataLoader, LocalFeatureStore

        provider = self.algorithm.providers.offline_for(self.cfg.model.input_modality)
        workers = self.cfg.data.dataloader_num_workers
        if workers is None:
            workers = self.algorithm.providers.model.default_dataloader_num_workers
        refs = _shard_offline_refs(
            self.refs,
            use_usp_preprocess=False,
            seed=self.cfg.training.seed,
            epoch=epoch,
            dp_rank=dp_rank,
            dp_size=dp_world_size,
        )
        # Pipeline microbatches must not expose a tail that the configured
        # local optimizer batch would drop at the end of each epoch.
        usable = (
            len(refs) // self.cfg.training.batch_size
        ) * self.cfg.training.batch_size
        refs = refs[:usable]
        return FeatureDataLoader(
            LocalFeatureStore(self.cfg.run_id),
            refs=refs,
            device="cpu",
            batch_size=local_batch_size,
            collate_fn=provider.build_collator(),
            per_sample_transform=provider.build_normalizer(
                seq_len,
                ttt_length=self.cfg.training.ttt_length,
                use_usp_preprocess=False,
            ),
            drop_last=True,
            strategy=self.algorithm.name,
            num_workers=workers,
            pin_memory=workers > 0,
        )


def _tokenizer_config(vocab_size: int):
    from torchtitan.components.tokenizer import BaseTokenizer

    class FeatureTokenizer(BaseTokenizer):
        @dataclass(kw_only=True, slots=True)
        class Config(BaseTokenizer.Config):
            vocab_size: int

        def __init__(self, config, *, tokenizer_path):
            super().__init__()
            self.vocab_size = config.vocab_size

        def get_vocab_size(self):
            return self.vocab_size

        def encode(self, *args, **kwargs):
            raise RuntimeError("Offline feature training already contains token IDs")

        def decode(self, *args, **kwargs):
            raise RuntimeError("Offline feature training does not decode token IDs")

    return FeatureTokenizer.Config(vocab_size=vocab_size)


@dataclass
class _OnlineSourceFactory:
    """Open the existing retained feature stream after native DCP restore."""

    cfg: Any
    algorithm: Any
    natural_steps: int
    store: Any = None

    def __call__(
        self, *, dp_rank, dp_world_size, local_batch_size, seq_len, resume_step
    ):
        del seq_len
        from specforge.launch import build_online_consumer_resources
        from specforge.runtime.data_plane import FeatureDataLoader
        from specforge.runtime.data_plane.streaming_ref_channel import (
            StreamingRefChannel,
        )
        from specforge.training.assembly import _dataloader_num_workers
        from specforge.training.disaggregated import (
            _consumer_database_path,
            _env,
            _mooncake_store,
        )

        channel = StreamingRefChannel(_env("DISAGG_REF_CHANNEL"))
        resources = None
        try:
            self.store = _mooncake_store(self.cfg, retain_on_release=True)
            resources = build_online_consumer_resources(
                run_id=self.cfg.run_id,
                feature_store=self.store,
                channel=channel,
                batch_size=local_batch_size,
                accumulation_steps=self.cfg.training.accumulation_steps,
                idle_timeout_s=float(os.environ.get("DISAGG_IDLE_TIMEOUT", "0"))
                or None,
                metadata_db_path=_consumer_database_path(self.cfg),
                dp_rank=dp_rank,
                dp_size=dp_world_size,
                inbox_dir=os.environ.get("DISAGG_INBOX_DIR") or None,
                resume_from=self.cfg.training.resume_from,
                resume_step=resume_step,
                async_ack=False,
            )
            source = FeatureDataLoader(
                self.store,
                queue=resources.queue,
                ack=False,
                device="cpu",
                batch_size=local_batch_size,
                collate_fn=self.algorithm.providers.server_streaming_for(
                    self.cfg.model.input_modality
                ).build_collator(),
                strategy=self.algorithm.name,
                num_workers=_dataloader_num_workers(self.cfg, self.algorithm),
                pin_memory=True,
            )
            return source, resources
        except BaseException as exc:
            cleanups = (
                (lambda: resources.on_failure(exc), resources.close)
                if resources is not None
                else (
                    lambda: channel.mark_consumer_failed(
                        f"{type(exc).__name__}: {exc}"
                    ),
                )
            )
            for cleanup in cleanups:
                try:
                    cleanup()
                except BaseException:
                    logging.getLogger(__name__).exception(
                        "Online source setup cleanup failed"
                    )
            raise

    def close(self):
        store, self.store = self.store, None
        if store is not None:
            store.close()


def _online_training_horizon(cfg, natural_steps):
    """Keep the producer's finite stream length separate from the LR horizon."""
    if natural_steps < 1:
        raise ValueError("The online producer must supply a complete optimizer window")
    training = cfg.training
    return (
        min(natural_steps, training.max_steps or natural_steps),
        training.total_steps or training.max_steps or natural_steps,
    )


def _scheduler_config(cfg, total_steps: int):
    from torchtitan.components.optimizer.lr_scheduler import LRSchedulersContainer

    @dataclass(kw_only=True, slots=True)
    class SpecForgeSchedule(LRSchedulersContainer.Config):
        schedule: str = "cosine"

        def build(self, *, optimizers, training_steps):
            warmup = self.warmup_steps
            horizon = self.total_steps
            cosine_start = 1.0
            if self.schedule == "cosine" and horizon - warmup > 1:
                import warnings

                import torch

                from specforge.lr_scheduler import CosineAnnealingWarmupLR

                # The existing two-stage scheduler queries CosineAnnealingLR
                # once more at its transition. Its initial factor differs
                # across PyTorch versions; derive it from the same installed
                # scheduler, rather than changing an established LR curve.
                dummy = torch.optim.SGD(
                    [torch.zeros((), device="cpu", requires_grad=True)], lr=1.0
                )
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="To get the last learning rate"
                    )
                    CosineAnnealingWarmupLR(
                        dummy, total_steps=horizon - warmup, warmup_steps=0
                    )
                cosine_start = dummy.param_groups[0]["lr"]
            # A one-step cosine horizon has no intermediate decay. Avoid the
            # legacy get_lr transition's division by zero on newer PyTorch.

            def multiplier(step):
                if step < warmup:
                    return (step + 1) / warmup
                if self.schedule == "constant":
                    return 1.0
                return (
                    cosine_start
                    * 0.5
                    * (1.0 + math.cos(math.pi * (step - warmup) / (horizon - warmup)))
                )

            return LRSchedulersContainer(optimizers, multiplier)

    warmup = int(cfg.training.warmup_ratio * total_steps)
    if warmup >= total_steps:
        raise ValueError("Warmup must finish before the scheduler horizon")
    return SpecForgeSchedule(
        total_steps=total_steps, warmup_steps=warmup, schedule=cfg.training.lr_scheduler
    )


def _export_draft(trainer, draft_config, directory: Path) -> None:
    import torch.distributed as dist
    from safetensors.torch import save_file
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
    )

    if len(trainer.model_parts) != 1:
        raise ValueError("Draft export currently requires one model part per rank")
    pp_degree = trainer.parallel_dims.pp
    state = get_model_state_dict(
        trainer.model_parts[0],
        # With PP, every stage needs its own full DP/TP state, not just the
        # global rank-zero stage. Offload those reconstructed tensors below.
        options=StateDictOptions(full_state_dict=True, cpu_offload=pp_degree == 1),
    )
    prefix = "training_model.draft_model."
    if pp_degree == 1:
        if dist.get_rank() != 0:
            return
        draft_state = {
            name[len(prefix) :]: value.contiguous()
            for name, value in state.items()
            if name.startswith(prefix)
        }
        if not draft_state:
            raise RuntimeError("Native TorchTitan checkpoint contains no draft weights")
        directory.mkdir(parents=True, exist_ok=True)
        draft_config.save_pretrained(str(directory))
        save_file(draft_state, str(directory / "model.safetensors"))
        return

    leader = all(
        trainer.parallel_dims.get_optional_mesh(
            name, include_singleton_axes=True
        ).get_local_rank()
        == 0
        for name in ("batch", "cp", "tp")
    )
    metadata = None
    if leader:
        stage = trainer.parallel_dims.get_mesh("pp").get_local_rank()
        filename = f"model-{stage + 1:05d}-of-{pp_degree:05d}.safetensors"
        draft_state = {
            name[len(prefix) :]: value.detach().cpu().contiguous()
            for name, value in state.items()
            if name.startswith(prefix)
        }
        directory.mkdir(parents=True, exist_ok=True)
        save_file(draft_state, str(directory / filename))
        metadata = {
            "weight_map": {name: filename for name in draft_state},
            "total_size": sum(
                value.numel() * value.element_size() for value in draft_state.values()
            ),
        }
    # Stage weights remain in ordinary HF shards on the output filesystem;
    # only small key/filename metadata crosses the world process group.
    gathered = [None] * dist.get_world_size() if dist.get_rank() == 0 else None
    dist.gather_object(metadata, gathered, dst=0)
    if dist.get_rank() == 0:
        weight_map, total_size = {}, 0
        for item in gathered:
            if item is None:
                continue
            duplicate = weight_map.keys() & item["weight_map"].keys()
            if duplicate:
                raise ValueError(
                    f"Pipeline export contains duplicate draft keys: {sorted(duplicate)}"
                )
            weight_map.update(item["weight_map"])
            total_size += item["total_size"]
        draft_config.save_pretrained(str(directory))
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                indent=2,
            )
            + "\n"
        )


def _resume_location(resume_from: str | None, checkpoint_dir: Path):
    """Honor an explicit checkpoint even when the output already has snapshots."""
    existing = [
        path
        for path in checkpoint_dir.glob("step-*")
        if path.name.removeprefix("step-").isdigit() and (path / ".metadata").is_file()
    ]
    if not resume_from:
        if existing:
            raise ValueError(
                "Existing TorchTitan checkpoints require explicit training.resume_from"
            )
        return None, -1
    local_path = (
        unquote(resume_from.removeprefix("file://"))
        if resume_from.startswith("file://")
        else resume_from
    )
    source = Path(local_path).expanduser().resolve()
    if not (source / ".metadata").is_file():
        candidates = []
        for root in (source, source / "checkpoint"):
            candidates.extend(
                path
                for path in root.glob("step-*")
                if path.name.removeprefix("step-").isdigit()
                and (path / ".metadata").is_file()
            )
        if not candidates:
            raise ValueError(
                "training.resume_from must contain a native TorchTitan DCP checkpoint"
            )
        source = max(candidates, key=lambda path: int(path.name.removeprefix("step-")))
    if source.parent.resolve() == checkpoint_dir.resolve():
        return str(source), int(source.name.removeprefix("step-"))
    if existing:
        raise ValueError(
            "Resuming an external checkpoint requires a fresh output directory"
        )
    return str(source), -1


def _native_config(
    cfg,
    algorithm,
    refs,
    model_sources,
    layout,
    data_degree,
    stop_steps,
    schedule_steps,
    *,
    online_source=None,
):
    from torchtitan.components.checkpoint import CheckpointManager
    from torchtitan.components.metrics import MetricsProcessor
    from torchtitan.components.optimizer.optimizer import default_adamw
    from torchtitan.components.validate import Validator
    from torchtitan.config import (
        CompileConfig,
        DebugConfig,
        ParallelismConfig,
        TrainingConfig,
    )
    from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
    from torchtitan.protocols.model_spec import ModelSpec

    from .data import FeatureDataLoader
    from .loss import SpecForgeObjectiveLoss
    from .model import SpecForgeTitanModel
    from .parallelize import group_optimizer_parameters_by_mesh, parallelize_dflash
    from .pipeline import pipeline_dflash
    from .runtime import SpecForgeTitanTrainer

    draft_config, variant, objective, teacher_path, draft_path, contract = model_sources
    training = cfg.training
    titan = training.torchtitan
    dump_folder = Path(cfg.output_dir) / cfg.run_id
    resume_path, load_step = _resume_location(
        training.resume_from, dump_folder / "checkpoint"
    )
    dataset_identity = []
    for ref in refs:
        path = Path(ref.feature_store_uri.removeprefix("file://"))
        stat = path.stat()
        dataset_identity.append((str(path), stat.st_size, stat.st_mtime_ns))
    contract.update(
        backend="torchtitan-native-v1",
        dataset_sha256=hashlib.sha256(
            json.dumps(dataset_identity).encode()
        ).hexdigest(),
        source_size=len(refs),
        data_degree=data_degree,
        batch_size=training.batch_size,
        accumulation_steps=training.accumulation_steps,
        num_epochs=training.num_epochs,
        max_length=cfg.data.max_length,
        schedule_total_steps=schedule_steps,
        learning_rate=training.learning_rate,
        lr_scheduler=training.lr_scheduler,
        warmup_ratio=training.warmup_ratio,
        max_grad_norm=training.max_grad_norm,
        seed=training.seed,
        parallelism=layout,
    )
    # Preserve old pure-eager native checkpoints. Compiled arithmetic and graph
    # parameter materialization carry explicit revisions so older compiled
    # runs cannot silently resume across a numerical-policy change.
    contract.update(_numerical_resume_contract(titan))
    if titan.engine != "trainer":
        contract.update(engine=titan.engine)
    if online_source is not None:
        from specforge.training.disaggregated import _consumer_database_path, _env

        contract.update(
            feature_source="online",
            stream_run_id=cfg.run_id,
            stream_natural_steps=online_source.natural_steps,
            stream_path=str(Path(_env("DISAGG_REF_CHANNEL")).resolve()),
            ledger_path=str(Path(_consumer_database_path(cfg)).resolve()),
        )
    activation_checkpoint = {
        "none": None,
        "full": FullAC.Config(),
        "selective": SelectiveAC.Config(),
    }[titan.activation_checkpoint]
    validator_options = Validator.Config(enable=False)
    if cfg.data.eval_hidden_states_path:
        from .validation import OfflineEvalSourceFactory, SpecForgeValidator

        eval_refs = (
            algorithm.providers.offline_for(cfg.model.input_modality)
            .build_reader(
                cfg.data.eval_hidden_states_path,
                run_id=f"{cfg.run_id}-eval",
                ttt_length=training.ttt_length,
                max_len=cfg.data.max_length,
            )
            .read()
        )
        if not eval_refs:
            raise ValueError("The offline evaluation dataset is empty")
        validator_options = SpecForgeValidator.Config(
            enable=True,
            freq=training.eval_interval,
            source_factory=OfflineEvalSourceFactory(cfg, algorithm, eval_refs),
            num_batches=math.ceil(len(eval_refs) / (data_degree * training.batch_size)),
            algorithm=variant,
            seed=training.seed,
            schedule_total_steps=schedule_steps,
        )
    trainer_config = SpecForgeTitanTrainer.Config
    model_config = SpecForgeTitanModel.Config
    parallelize_fn = parallelize_dflash
    compile_options = CompileConfig(enable=titan.compile, components=["model"])
    dataloader_options = FeatureDataLoader.Config(
        source_factory=_OfflineSourceFactory(cfg, algorithm, refs),
        epochs=training.num_epochs,
        pad_to_seq_len=(
            titan.pp_size > 1
            or not titan.disable_cuda_graphs
            or titan.engine == "graph"
        ),
    )
    if online_source is not None:
        from .online import OnlineFeatureDataLoader, SpecForgeOnlineTrainer

        trainer_config = SpecForgeOnlineTrainer.Config
        dataloader_options = OnlineFeatureDataLoader.Config(
            source_factory=online_source
        )
    if titan.engine == "graph":
        from torchtitan.experiments.graph_trainer.configs import (
            GraphTrainerCompileConfig,
        )

        from .graph import SpecForgeGraphModel, SpecForgeGraphTrainer
        from .graph_parallelize import parallelize_graph_dflash

        trainer_config = SpecForgeGraphTrainer.Config
        model_config = SpecForgeGraphModel.Config
        parallelize_fn = parallelize_graph_dflash
        compile_options = GraphTrainerCompileConfig(
            enable=True,
            components=["model"],
            inductor_compilation=titan.graph_inductor,
            disable_passes=["cudagraph_pass"] if titan.disable_cuda_graphs else [],
        )
    native = trainer_config(
        model_spec=ModelSpec(
            name="specforge",
            flavor=variant,
            model=model_config(
                draft_config=draft_config.to_dict(),
                algorithm=variant,
                objective=objective,
                teacher_state_path=teacher_path,
                draft_state_path=draft_path,
                init_seed=training.seed,
            ),
            parallelize_fn=parallelize_fn,
            pipelining_fn=pipeline_dflash,
            post_optimizer_build_fn=group_optimizer_parameters_by_mesh,
            state_dict_adapter=None,
        ),
        dump_folder=str(dump_folder),
        tokenizer=_tokenizer_config(int(draft_config.vocab_size)),
        dataloader=dataloader_options,
        loss=SpecForgeObjectiveLoss.Config(algorithm=variant),
        optimizer=default_adamw(
            lr=training.learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
        ),
        lr_scheduler=_scheduler_config(cfg, schedule_steps),
        training=TrainingConfig(
            local_batch_size=training.batch_size,
            global_batch_size=training.batch_size
            * data_degree
            * training.accumulation_steps,
            seq_len=cfg.data.max_length,
            steps=stop_steps,
            max_norm=training.max_grad_norm,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
            disable_cuda_graphs=(
                True if titan.engine == "graph" else titan.disable_cuda_graphs
            ),
        ),
        parallelism=ParallelismConfig(**layout),
        activation_checkpoint=activation_checkpoint,
        compile=compile_options,
        checkpoint=CheckpointManager.Config(
            enable=True,
            folder="checkpoint",
            interval=max(1, training.save_interval or stop_steps),
            initial_load_path=resume_path,
            load_step=load_step,
            initial_load_model_only=False,
            last_save_model_only=False,
            keep_latest_k=training.max_checkpoints,
        ),
        metrics=MetricsProcessor.Config(
            log_freq=training.log_interval,
            enable_tensorboard=cfg.tracking.report_to == "tensorboard",
            enable_wandb=False,
        ),
        debug=DebugConfig(seed=training.seed),
        resume_contract=contract,
        schedule_total_steps=schedule_steps,
        validator=validator_options,
    )
    native.comm.init_timeout_seconds = training.dist_timeout * 60
    native.comm.train_timeout_seconds = training.dist_timeout * 60
    return native


def build_torchtitan_training_run(cfg, *, algorithm):
    """Return the ordinary TrainingRun backed by an actual native Titan Trainer."""
    from specforge.training.assembly import TrainingRun

    _validate_runtime_versions()
    online = cfg.mode == "online"
    if online:
        if cfg.deployment.mode != "disaggregated" or cfg.training.role != "consumer":
            raise ValueError(
                "TorchTitan online training requires a disaggregated consumer"
            )
        titan = cfg.training.torchtitan
        if cfg.training.tp_size != 1 or titan.cp_size != 1 or titan.pp_size != 1:
            raise ValueError("TorchTitan online training currently supports DP only")
        if titan.engine != "trainer" or not titan.disable_cuda_graphs:
            raise ValueError(
                "TorchTitan online features do not yet support graph capture"
            )
    elif cfg.deployment.mode != "local_colocated":
        raise ValueError(
            "TorchTitan offline training currently requires colocated features"
        )
    if algorithm.name != cfg.training.strategy or algorithm.name not in {
        "dflash",
        "dspark",
    }:
        raise ValueError("TorchTitan requires a matching DFlash or DSpark registration")
    if cfg.model.torch_dtype != "bfloat16":
        raise ValueError("TorchTitan currently requires model.torch_dtype='bfloat16'")
    if cfg.model.use_liger_kernel or cfg.training.optimizer_cpu_offload:
        raise ValueError(
            "TorchTitan does not yet support Liger or optimizer CPU offload"
        )
    if cfg.tracking.report_to not in {"none", "tensorboard"}:
        raise ValueError(
            "TorchTitan currently supports tracking.report_to none or tensorboard"
        )
    if bool(cfg.data.eval_hidden_states_path) != bool(cfg.training.eval_interval):
        raise ValueError(
            "TorchTitan evaluation requires both eval_hidden_states_path and eval_interval"
        )
    if cfg.training.eval_interval and cfg.training.torchtitan.pp_size > 1:
        raise ValueError(
            "TorchTitan evaluation does not yet support pipeline parallelism"
        )
    if cfg.profiling.enabled:
        raise ValueError("SpecForge profiling options are not yet mapped to TorchTitan")
    if cfg.training.max_checkpoints == 1:
        raise ValueError(
            "TorchTitan checkpoint retention requires zero or at least two"
        )
    layout, data_degree = _parallelism_layout(
        cfg, int(os.environ.get("WORLD_SIZE", "1"))
    )
    online_source = None
    if online:
        from specforge.training.disaggregated import _env, _read_online_total_steps

        natural_steps = _read_online_total_steps(cfg, _env("DISAGG_REF_CHANNEL"))
        stop_steps, schedule_steps = _online_training_horizon(cfg, natural_steps)
        refs = []
        online_source = _OnlineSourceFactory(cfg, algorithm, natural_steps)
    else:
        provider = algorithm.providers.offline_for(cfg.model.input_modality)
        refs = provider.build_reader(
            cfg.data.hidden_states_path,
            run_id=cfg.run_id,
            ttt_length=cfg.training.ttt_length,
            max_len=cfg.data.max_length,
        ).read()
        stop_steps, schedule_steps = _training_horizon(cfg, len(refs), data_degree)

    def execute():
        import torch.distributed as dist
        from torchtitan.tools.logging import init_logger

        if dist.is_initialized():
            raise RuntimeError("TorchTitan must initialize its own process groups")
        trainer = None
        failed = False
        init_logger()
        try:
            # Per-process snapshots avoid shared-filesystem assumptions or
            # rank-zero writes before Titan initializes the process group.
            # Keep multi-GB tables on the configured training volume rather
            # than a potentially small container /tmp filesystem.
            output_root = Path(cfg.output_dir)
            output_root.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(
                prefix=".torchtitan-init-", dir=output_root
            ) as directory:
                model_sources = _snapshot_model_sources(cfg, algorithm, directory)
                native = _native_config(
                    cfg,
                    algorithm,
                    refs,
                    model_sources,
                    layout,
                    data_degree,
                    stop_steps,
                    schedule_steps,
                    online_source=online_source,
                )
                trainer = native.build()
                trainer.train()
                if cfg.training.torchtitan.export_draft:
                    _export_draft(
                        trainer, model_sources[0], Path(native.dump_folder) / "draft"
                    )
                return int(trainer.step)
        except BaseException as exc:
            failed = True
            logging.getLogger(__name__).exception("Native TorchTitan training failed")
            if online:
                from specforge.training.disaggregated import (
                    _publish_role_assembly_failure,
                )

                _publish_role_assembly_failure(cfg, exc)
            _abort_titan_process_groups()
            raise
        finally:
            if online_source is not None:
                try:
                    online_source.close()
                except BaseException:
                    logging.getLogger(__name__).exception(
                        "Online feature store cleanup failed"
                    )
                    if not failed:
                        _abort_titan_process_groups()
                        raise
            if not failed:
                try:
                    if trainer is not None:
                        trainer.close()
                except BaseException:
                    logging.getLogger(__name__).exception(
                        "Native TorchTitan cleanup failed"
                    )
                    _abort_titan_process_groups()
                    raise
                else:
                    if dist.is_initialized():
                        dist.destroy_process_group()

    return TrainingRun(execute=execute)


__all__ = ["build_torchtitan_training_run"]
