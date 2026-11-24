# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Training-object assembly driven by the application composition root.

The application resolves one immutable algorithm registration and passes it
through every builder.  This module never resolves an algorithm name and never
constructs an in-process online target engine; online capture comes exclusively
from an external SGLang server through the disaggregated runtime.

Heavy model/data dependencies stay lazy so importing :mod:`specforge.training`
does not load Transformers, datasets, or a target backend.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Sequence

from specforge.algorithms.contracts import FeatureMode
from specforge.algorithms.registry import AlgorithmRegistration
from specforge.config import Config

if TYPE_CHECKING:
    from specforge.distributed import Topology
    from specforge.training.contracts import TrainingRuntime


@dataclass
class ModelBundle:
    """Objects and capture metadata needed by one configured training run."""

    model: Any
    draft_model: Any
    draft_config: Any
    input_tools: Any = None
    target_head: Any = None
    target_hidden_size: int = 0
    target_vocab_size: int = 0
    draft_vocab_size: int = 0
    capture_layers: Optional[List[int]] = None
    strategy_kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class TrainingRun:
    """A fully assembled run with one lifecycle for rollout and training."""

    trainer: Any = None
    execute: Optional[Callable[[], int]] = None
    on_success: Optional[Callable[[int], None]] = None
    on_failure: Optional[Callable[[BaseException], None]] = None
    on_finally: Optional[Callable[[], None]] = None

    def __post_init__(self) -> None:
        if (self.trainer is None) == (self.execute is None):
            raise ValueError("a training run needs exactly one trainer or executor")
        if self.execute is not None and any(
            hook is not None
            for hook in (self.on_success, self.on_failure, self.on_finally)
        ):
            raise ValueError("lifecycle hooks belong to trainer-bearing runs only")

    def run(self) -> int:
        if self.execute is not None:
            return self.execute()
        try:
            result = self.trainer.fit()
            if self.on_success is not None:
                self.on_success(result)
            return result
        except BaseException as exc:
            if self.on_failure is not None:
                self.on_failure(exc)
            raise
        finally:
            if self.on_finally is not None:
                self.on_finally()


def _target_text_config(config):
    return getattr(config, "text_config", config)


def _torch_dtype(name: str):
    import torch

    return getattr(torch, name)


def _device():
    from specforge.utils import get_local_device

    return get_local_device()


def _load_text_tokenizer(cfg: Config):
    """Load tokenizer tooling used by current built-in text providers."""
    if cfg.model.input_modality != "text":
        raise ValueError(
            "built-in algorithms currently provide training-model input tooling "
            f"only for modality 'text', got {cfg.model.input_modality!r}; "
            "another modality must add its own input provider"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    if cfg.model.tokenizer_pad_token_id is not None:
        tokenizer.pad_token_id = cfg.model.tokenizer_pad_token_id
    elif tokenizer.pad_token_id is None:
        fallback_id = tokenizer.eos_token_id
        if isinstance(fallback_id, (list, tuple)):
            fallback_id = fallback_id[0] if fallback_id else None

        if fallback_id is None:
            fallback_id = tokenizer.unk_token_id
        if fallback_id is None:
            raise ValueError(
                "target tokenizer has no pad, EOS, or unknown token ID; set "
                "model.tokenizer_pad_token_id explicitly"
            )
        tokenizer.pad_token_id = fallback_id
    return tokenizer


def _load_input_tools(
    cfg: Config,
    algorithm: AlgorithmRegistration,
    *,
    input_adapter=None,
):
    """Load modality tooling through the provider port or the text default."""

    if input_adapter is None and cfg.mode == "online":
        streaming = algorithm.providers.server_streaming_for(cfg.model.input_modality)
        input_adapter = streaming.create_input_adapter(cfg)
    if input_adapter is not None:
        return input_adapter.load_input_tools(cfg)
    return _load_text_tokenizer(cfg)


def build_model_bundle(
    cfg: Config, *, algorithm: AlgorithmRegistration, topology: Topology
) -> ModelBundle:
    """Build the resolved canonical model, objective and frozen target pieces."""
    return algorithm.providers.model.build_bundle(cfg, algorithm, topology)


def _training_runtime(cfg: Config) -> TrainingRuntime:
    """The runtime that trains this run's draft, and its precision: decided here."""
    import torch

    from specforge.training.contracts import TrainingRuntime

    return TrainingRuntime.resolve(
        cfg.training.runtime,
        model_dtype=getattr(torch, cfg.model.torch_dtype),
        sharding_strategy=cfg.training.fsdp_sharding,
    )


def _optimizer_spec(cfg: Config):
    from specforge.training.contracts import OptimizerSpec

    t = cfg.training
    return OptimizerSpec(
        kind=t.optimizer,
        lr=t.learning_rate,
        weight_decay=t.weight_decay,
        param_group_policy=t.param_group_policy,
        offload_master=t.optimizer_cpu_offload,
        extra=dict(t.optimizer_options),
    )


def _scheduler_spec(cfg: Config):
    from specforge.training.contracts import SchedulerSpec

    t = cfg.training
    return SchedulerSpec(
        kind=t.lr_scheduler,
        total_steps=t.total_steps or t.max_steps,
        warmup_ratio=t.warmup_ratio,
        min_lr_ratio=t.min_lr_ratio,
        extra=dict(t.lr_scheduler_options),
    )


def _logger(metrics, step):
    printable = {}
    for key, value in metrics.items():
        try:
            printable[key] = float(value)
        except (TypeError, ValueError):
            try:
                printable[key] = [float(item) for item in value]
            except (TypeError, ValueError):
                continue
    print(f"step {step}: {printable}", flush=True)


def _configured_logger(cfg: Config):
    """Return this process's metric logger.

    Every rank keeps the console logger. An external tracker (W&B, MLflow,
    SwanLab, TensorBoard) is created on global rank zero only, so a job has one
    run and one writer instead of world-size copies.
    """
    if cfg.tracking.report_to == "none" or cfg.training.role == "producer":
        return _logger
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return _logger

    from types import SimpleNamespace

    from specforge.training.tracking import create_tracker_logger

    options = cfg.tracking.model_dump()
    options["wandb_project"] = options["wandb_project"] or "specforge"
    options["wandb_name"] = options["wandb_name"] or cfg.run_id
    options["swanlab_project"] = options["swanlab_project"] or "specforge"
    options["swanlab_name"] = options["swanlab_name"] or cfg.run_id
    options["mlflow_experiment_name"] = options["mlflow_experiment_name"] or "specforge"
    options["mlflow_run_name"] = options["mlflow_run_name"] or cfg.run_id
    if cfg.tracking.report_to == "wandb":
        # W&B is the canonical reproduction record for the disaggregated K3
        # runs. Keep the complete resolved config next to the metric stream;
        # tracker._public_config recursively redacts credentials before init.
        options["specforge_config"] = cfg.model_dump(mode="json")
    return create_tracker_logger(
        SimpleNamespace(**options), cfg.output_dir, console_logger=_logger
    )


def _close_configured_logger(logger) -> None:
    close = getattr(logger, "close", None)
    if callable(close):
        close()


def _tokenizer_chat_template_hash(tokenizer) -> Optional[str]:
    """Hash the tokenizer's effective chat template so cached tokenizations
    are invalidated when the model repository updates its template."""
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return None
    return hashlib.sha256(str(template).encode("utf-8")).hexdigest()[:12]


def _prompt_cache_key(
    cfg: Config, *, tokenizer=None, path: Optional[str] = None
) -> str:
    source_path = path or cfg.data.prompts_path or cfg.data.train_data_path
    content_hash = None
    if source_path and os.path.isfile(source_path):
        source_hasher = hashlib.sha256()
        with open(source_path, "rb") as source_file:
            for chunk in iter(lambda: source_file.read(8 * 1024 * 1024), b""):
                source_hasher.update(chunk)
        content_hash = source_hasher.hexdigest()

    identity = {
        "path": source_path,
        "content_hash": content_hash,
        "max_length": cfg.data.max_length,
        "chat_template": cfg.data.chat_template,
        "tokenizer_chat_template_hash": _tokenizer_chat_template_hash(tokenizer),
        "is_preformatted": cfg.data.is_preformatted,
        "train_only_last_turn": cfg.data.train_only_last_turn,
        "max_prompts": cfg.data.max_prompts,
        "target_model": cfg.model.target_model_path,
        "draft_config": cfg.draft,
        "strategy": cfg.training.objective,
        "input_modality": cfg.model.input_modality,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _prepare_prompts(
    cfg: Config,
    tokenizer,
    *,
    algorithm: AlgorithmRegistration,
    draft_config,
    path: Optional[str] = None,
    cache_key: Optional[str] = None,
) -> Sequence[dict]:
    """Prepare one prompt source with an optional path/cache namespace override.

    Training keeps the configured cache key. Evaluation supplies its own path
    and derived key so it can never read or overwrite the training prompt cache.
    """
    if cfg.model.input_modality != "text":
        raise ValueError(
            "the built-in prompt preparer supports only modality 'text'; "
            f"algorithm {algorithm.name!r} must provide a ServerInputAdapter "
            f"for {cfg.model.input_modality!r}"
        )
    from specforge.data.prompt_builder import prepare_prompt_tasks

    configured_path = cfg.data.prompts_path or cfg.data.train_data_path
    source_path = path or configured_path
    if not source_path:
        raise ValueError("prompt preparation requires a non-empty data path")
    if cache_key is None:
        cache_key = (
            cfg.data.cache_key
            if path is None and cfg.data.cache_key is not None
            else _prompt_cache_key(cfg, tokenizer=tokenizer, path=source_path)
        )
    min_loss_tokens = algorithm.providers.model.minimum_loss_tokens(cfg, draft_config)
    return prepare_prompt_tasks(
        source_path,
        tokenizer,
        chat_template=cfg.data.chat_template,
        max_length=cfg.data.max_length,
        is_preformatted=cfg.data.is_preformatted,
        train_only_last_turn=cfg.data.train_only_last_turn,
        cache_dir=cfg.data.cache_dir,
        cache_key=cache_key,
        num_proc=cfg.data.build_dataset_num_proc,
        min_loss_tokens=min_loss_tokens,
        max_prompts=cfg.data.max_prompts,
        loss_mask_filter=algorithm.providers.model.loss_mask_filter,
    )


def _install_dataset_vocab_mapping(
    cfg: Config,
    bundle: ModelBundle,
    *,
    counts: Counter,
    dataset_identity: str,
) -> None:
    """Build, cache, and install one deterministic EAGLE vocabulary map."""
    if (
        cfg.model.vocab_mapping_path
        or (
            hasattr(bundle.draft_model, "vocabulary")
            and bool(bundle.draft_model.vocabulary.ready)
        )
        or bundle.draft_vocab_size == bundle.target_vocab_size
    ):
        return

    key = hashlib.sha256(
        (
            f"{dataset_identity}:{bundle.target_vocab_size}:"
            f"{bundle.draft_vocab_size}"
        ).encode()
    ).hexdigest()
    directory = os.path.join(cfg.data.cache_dir, "vocab_mapping")
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{key}.pt")
    if os.path.exists(path):
        import torch

        mapping = torch.load(path, map_location="cpu", weights_only=True)
        bundle.draft_model.vocabulary.load_mapping(
            t2d=mapping["t2d"], d2t=mapping["d2t"]
        )
        return

    import torch

    from specforge.data.preprocessing import process_token_dict_to_mappings

    d2t, t2d = process_token_dict_to_mappings(
        counts,
        bundle.draft_vocab_size,
        bundle.target_vocab_size,
    )

    # Every rank derives the same mapping and installs it directly, so training
    # does not depend on a shared cache filesystem. Rank 0 alone persists the
    # reusable cache file, avoiding concurrent torch.save writes.
    bundle.draft_model.vocabulary.load_mapping(t2d=t2d, d2t=d2t)
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    if not distributed or torch.distributed.get_rank() == 0:
        temporary = f"{path}.{os.getpid()}.tmp"
        torch.save({"d2t": d2t, "t2d": t2d}, temporary)
        os.replace(temporary, path)


def _ensure_offline_vocab_mapping(
    cfg: Config,
    bundle: ModelBundle,
    algorithm: AlgorithmRegistration,
) -> None:
    """Derive a local-offline map from the exact feature ids and loss masks."""
    if FeatureMode.OFFLINE not in algorithm.providers.vocab_mapping_modes:
        return
    if (
        cfg.model.vocab_mapping_path
        or (
            hasattr(bundle.draft_model, "vocabulary")
            and bool(bundle.draft_model.vocabulary.ready)
        )
        or bundle.draft_vocab_size == bundle.target_vocab_size
    ):
        return

    from specforge.runtime.data_plane.offline_reader import list_feature_files
    from specforge.training.vocab_mapping import count_effective_feature_tokens

    identity_parts = []
    for path in list_feature_files(cfg.data.hidden_states_path):
        stat = os.stat(path)
        identity_parts.append((os.path.abspath(path), stat.st_size, stat.st_mtime_ns))
    identity = json.dumps(
        {
            "kind": "offline-features-v1",
            "files": identity_parts,
            "max_length": cfg.data.max_length,
        },
        sort_keys=True,
    )
    counts = count_effective_feature_tokens(
        cfg.data.hidden_states_path,
        max_length=cfg.data.max_length,
        target_vocab_size=bundle.target_vocab_size,
    )
    _install_dataset_vocab_mapping(
        cfg,
        bundle,
        counts=counts,
        dataset_identity=identity,
    )


def _dataloader_num_workers(cfg: Config, algorithm: AlgorithmRegistration) -> int:
    dataloader_num_workers = cfg.data.dataloader_num_workers
    if dataloader_num_workers is None:
        dataloader_num_workers = (
            algorithm.providers.model.default_dataloader_num_workers
        )
    return 4 if dataloader_num_workers is None else dataloader_num_workers


def _profiling_options(cfg: Config):
    from specforge.training.profiling import ProfilingOptions

    return ProfilingOptions(
        enabled=cfg.profiling.enabled,
        start_step=cfg.profiling.start_step,
        num_steps=cfg.profiling.num_steps,
        record_shapes=cfg.profiling.record_shapes,
    )


def _common_launch_kwargs(
    cfg: Config,
    bundle: ModelBundle,
    algorithm: AlgorithmRegistration,
    *,
    topology: Topology,
    logger=_logger,
) -> Dict[str, Any]:
    t = cfg.training
    # USP shards one logical sample over ``sp_size`` ranks.  Preserve the
    # legacy optimizer-window semantics: one user accumulation unit represents
    # a complete logical sequence, not one local sequence shard.
    accumulation_steps = t.accumulation_steps
    if t.attention_backend == "usp":
        accumulation_steps *= t.sp_ulysses_size * t.sp_ring_size
    return dict(
        algorithm=algorithm,
        modality=cfg.model.input_modality,
        optimizer_spec=_optimizer_spec(cfg),
        scheduler_spec=_scheduler_spec(cfg),
        max_grad_norm=t.max_grad_norm,
        runtime=_training_runtime(cfg),
        run_id=cfg.run_id,
        output_dir=cfg.output_dir,
        batch_size=t.batch_size,
        accumulation_steps=accumulation_steps,
        max_steps=t.max_steps,
        total_steps=t.total_steps,
        save_interval=t.save_interval,
        eval_interval=t.eval_interval,
        max_checkpoints=t.max_checkpoints,
        logger=logger,
        log_interval=t.log_interval,
        strategy_kwargs=bundle.strategy_kwargs,
        topology=topology,
        fsdp_sharding=t.fsdp_sharding,
        ddp_bucket_cap_mb=t.ddp_bucket_cap_mb,
        dataloader_num_workers=_dataloader_num_workers(cfg, algorithm),
        profiling_options=_profiling_options(cfg),
    )


def build_training_run(
    cfg: Config,
    *,
    algorithm: AlgorithmRegistration,
    topology: Optional[Topology] = None,
) -> TrainingRun:
    """Assemble one validated run from an already-resolved algorithm.

    Offline training may run in one process or with a disaggregated feature
    source.  Online training is always disaggregated and captures through
    external SGLang servers; colocated online execution is intentionally absent.
    ``topology`` is the job's shape; only a producer runs without one.
    """

    t = cfg.training
    if t.role != "producer":
        if topology is None:
            raise ValueError(
                "training needs the job's topology from init_distributed; "
                "only a producer runs without one"
            )
        cfg.validate_world_size(topology.world_size)
        configured = (t.tp_size, t.sp_ulysses_size, t.sp_ring_size)
        built = (topology.tp_size, topology.sp_ulysses_size, topology.sp_ring_size)
        if built != configured:
            raise ValueError(
                f"the topology has (tp, ulysses, ring) = {built} but the run "
                f"configures {configured}"
            )

    if cfg.mode == "online" and cfg.deployment.mode != "disaggregated":
        raise ValueError(
            "online training is server-only and requires "
            "deployment.mode='disaggregated'"
        )

    if cfg.deployment.mode == "disaggregated":
        from specforge.training.disaggregated import build_disaggregated_run
        from specforge.training.model_loading import preflight_draft_kernels

        run_logger = _configured_logger(cfg)
        try:
            return build_disaggregated_run(
                cfg,
                algorithm=algorithm,
                topology=topology,
                build_model_bundle=lambda run_cfg: build_model_bundle(
                    run_cfg, algorithm=algorithm, topology=topology
                ),
                preflight_model_bundle=lambda run_cfg: preflight_draft_kernels(
                    run_cfg, algorithm=algorithm
                ),
                prepare_prompts=lambda run_cfg, tokenizer, **kwargs: _prepare_prompts(
                    run_cfg,
                    tokenizer,
                    algorithm=algorithm,
                    **kwargs,
                ),
                training_runtime=_training_runtime,
                logger=run_logger,
            )
        except BaseException:
            _close_configured_logger(run_logger)
            raise

    if cfg.mode != "offline":
        raise ValueError("colocated execution supports offline training only")

    bundle = build_model_bundle(cfg, algorithm=algorithm, topology=topology)
    from specforge.launch import build_offline_runtime

    _ensure_offline_vocab_mapping(cfg, bundle, algorithm)
    run_logger = _configured_logger(cfg)
    try:
        trainer = build_offline_runtime(
            hidden_states_path=cfg.data.hidden_states_path,
            eval_hidden_states_path=cfg.data.eval_hidden_states_path or None,
            draft_model=bundle.model,
            target_head=bundle.target_head,
            ttt_length=t.objective_options.get("ttt_length", 7),
            max_len=cfg.data.max_length,
            num_epochs=t.num_epochs,
            use_usp_preprocess=(t.attention_backend == "usp"),
            seed=t.seed,
            resume_from=cfg.resume_path,
            **_common_launch_kwargs(
                cfg,
                bundle,
                algorithm,
                topology=topology,
                logger=run_logger,
            ),
        )
    except BaseException:
        _close_configured_logger(run_logger)
        raise
    return TrainingRun(trainer=trainer)


__all__ = [
    "ModelBundle",
    "TrainingRun",
    "build_model_bundle",
    "build_training_run",
]
