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
from typing import Any, Callable, Dict, List, Mapping, Optional

from specforge.algorithms.contracts import FeatureMode
from specforge.algorithms.registry import AlgorithmRegistration
from specforge.config import Config
from specforge.training.provenance import (
    model_resume_provenance as _model_resume_provenance,
)


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


def _load_draft(cfg: Config, algorithm: AlgorithmRegistration):
    """Construct the configured draft model without any legacy trainer code."""
    from specforge.modeling.draft.registry import resolve_draft
    from specforge.training.model_loading import resolve_draft_config

    provider = algorithm.providers.model
    draft_config = resolve_draft_config(cfg, provider=provider.draft_config)
    draft_model = provider.build_draft(cfg, draft_config)
    architecture = provider.draft_config.architecture
    expected_type = resolve_draft(architecture)
    if not isinstance(draft_model, expected_type):
        raise ValueError(
            f"training.strategy={algorithm.name!r} requires {architecture}, but "
            f"the resolved draft config builds "
            f"{type(draft_model).__name__}"
        )
    return draft_config, draft_model


def _load_text_tokenizer(cfg: Config):
    """Load tokenizer tooling used by current built-in text providers."""
    if cfg.model.input_modality != "text":
        raise ValueError(
            "built-in algorithms currently provide training-model input tooling "
            f"only for modality 'text', got {cfg.model.input_modality!r}; "
            "another modality must add its own input provider"
        )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )


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


def build_model_bundle(cfg: Config, *, algorithm: AlgorithmRegistration) -> ModelBundle:
    """Build the method-specific composite model and frozen target pieces."""
    import torch
    from transformers import AutoConfig

    provider = algorithm.providers.model
    draft_config, draft_model = _load_draft(cfg, algorithm)
    needs_input_tools = provider.needs_input_tools(cfg, draft_model)
    input_tools = _load_input_tools(cfg, algorithm) if needs_input_tools else None
    target_config = AutoConfig.from_pretrained(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    text_config = _target_text_config(target_config)
    target_hidden_size = int(text_config.hidden_size)
    target_vocab_size = int(text_config.vocab_size)
    draft_vocab_size = int(
        getattr(draft_config, "draft_vocab_size", draft_config.vocab_size)
    )

    parts = provider.build_training_model(
        cfg, draft_model, draft_config, target_config, input_tools
    )
    if cfg.mode == "online" and parts.capture_layers is None:
        parts.capture_layers = provider.resolve_capture_layers(
            cfg, draft_config, target_config
        )

    # Keep the composite and target parts bf16 while avoiding accidental target
    # gradients. The optimizer still receives the strategy's trainable module.
    if parts.target_head is not None and isinstance(parts.target_head, torch.nn.Module):
        parts.target_head.requires_grad_(False)

    return ModelBundle(
        model=parts.model,
        draft_model=draft_model,
        draft_config=draft_config,
        input_tools=input_tools,
        target_head=parts.target_head,
        target_hidden_size=target_hidden_size,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
        capture_layers=parts.capture_layers,
        # This mapping also carries the provider-bound checkpoint policy. It is
        # forwarded unchanged through every trainer-bearing topology.
        strategy_kwargs=algorithm.providers.step.bind_runtime(
            cfg,
            draft_model,
            parts.model,
            model_provenance=_model_resume_provenance(
                cfg,
                draft_config,
                target_config,
                capture_layers=parts.capture_layers,
            ),
        ),
    )


class _ConfiguredOptimizerFactory:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.total_steps = cfg.training.total_steps or cfg.training.max_steps

    def configure_total_steps(self, total_steps: int) -> None:
        if self.total_steps is None:
            self.total_steps = total_steps
        elif self.total_steps != total_steps:
            raise ValueError(
                "optimizer/controller schedule mismatch: "
                f"{self.total_steps} != {total_steps}"
            )

    def __call__(self, draft_module):
        from specforge.optimizer import BF16Optimizer

        if self.total_steps is None:
            raise RuntimeError("optimizer total_steps was not resolved before assembly")
        t = self.cfg.training
        return BF16Optimizer(
            draft_module,
            lr=t.learning_rate,
            max_grad_norm=t.max_grad_norm,
            warmup_ratio=t.warmup_ratio,
            total_steps=self.total_steps,
        )


def _optimizer_factory(cfg: Config):
    return _ConfiguredOptimizerFactory(cfg)


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
    """Create an external tracker only for a trainer-bearing run."""
    if cfg.tracking.report_to == "none" or cfg.training.role == "producer":
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
    return create_tracker_logger(
        SimpleNamespace(**options), cfg.output_dir, console_logger=_logger
    )


def _close_configured_logger(logger) -> None:
    close = getattr(logger, "close", None)
    if callable(close):
        close()


def _prompt_cache_key(cfg: Config, *, path: Optional[str] = None) -> str:
    import json

    identity = {
        "path": path or cfg.data.prompts_path or cfg.data.train_data_path,
        "max_length": cfg.data.max_length,
        "chat_template": cfg.data.chat_template,
        "is_preformatted": cfg.data.is_preformatted,
        "train_only_last_turn": cfg.data.train_only_last_turn,
        "max_prompts": cfg.data.max_prompts,
        "target_model": cfg.model.target_model_path,
        "draft_config": cfg.model.draft_model_config,
        "draft_checkpoint": cfg.model.draft_checkpoint_path,
        "draft_num_hidden_layers": cfg.model.draft_num_hidden_layers,
        "draft_block_size": cfg.model.draft_block_size,
        "strategy": cfg.training.strategy,
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
) -> List[dict]:
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
            else _prompt_cache_key(cfg, path=source_path)
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
        bundle.draft_model.load_vocab_mapping(path)
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
    bundle.draft_model.d2t.copy_(d2t)
    bundle.draft_model.t2d.copy_(t2d)
    bundle.draft_model.vocab_mapping_loaded = True
    distributed = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    if not distributed or torch.distributed.get_rank() == 0:
        temporary = f"{path}.{os.getpid()}.tmp"
        torch.save({"d2t": d2t, "t2d": t2d}, temporary)
        os.replace(temporary, path)


def _ensure_vocab_mapping(
    cfg: Config,
    bundle: ModelBundle,
    prompts: List[dict],
    algorithm: AlgorithmRegistration,
) -> None:
    """Generate the online EAGLE-family map from its prepared prompt plan."""
    # Avoid walking (and validating) the prompt payload when no mapping can be
    # needed.  This is common for equal-vocabulary targets and keeps assembly
    # independent of the online prompt schema for non-mapping runs.
    if FeatureMode.STREAMING not in algorithm.providers.vocab_mapping_modes:
        return
    if (
        cfg.model.vocab_mapping_path
        or bundle.draft_vocab_size == bundle.target_vocab_size
    ):
        return

    counts: Counter = Counter()
    for task in prompts:
        payload = task["payload"]
        for token_id, keep in zip(payload["input_ids"], payload["loss_mask"]):
            if keep:
                counts[int(token_id)] += 1
    _install_dataset_vocab_mapping(
        cfg,
        bundle,
        counts=counts,
        dataset_identity=cfg.data.cache_key or _prompt_cache_key(cfg),
    )


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
    return dataloader_num_workers


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
        optimizer_factory=_optimizer_factory(cfg),
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
        tp_size=t.tp_size,
        sp_ulysses_size=t.sp_ulysses_size,
        sp_ring_size=t.sp_ring_size,
        dataloader_num_workers=_dataloader_num_workers(cfg, algorithm),
        profiling_options=_profiling_options(cfg),
    )


def build_training_run(
    cfg: Config,
    *,
    algorithm: AlgorithmRegistration,
) -> TrainingRun:
    """Assemble one validated run from an already-resolved algorithm.

    Offline training may run in one process or with a disaggregated feature
    source.  Online training is always disaggregated and captures through
    external SGLang servers; colocated online execution is intentionally absent.
    """

    if algorithm.name != cfg.training.strategy:
        raise ValueError(
            "resolved algorithm does not match training.strategy: "
            f"{algorithm.name!r} != {cfg.training.strategy!r}"
        )

    t = cfg.training
    if t.role != "producer":
        import torch.distributed as dist

        cfg.validate_world_size(dist.get_world_size() if dist.is_initialized() else 1)

    if cfg.mode == "online" and cfg.deployment.mode != "disaggregated":
        raise ValueError(
            "online training is server-only and requires "
            "deployment.mode='disaggregated'"
        )

    if cfg.deployment.mode == "disaggregated":
        from specforge.training.disaggregated import build_disaggregated_run

        run_logger = _configured_logger(cfg)
        try:
            return build_disaggregated_run(
                cfg,
                algorithm=algorithm,
                build_model_bundle=lambda run_cfg: build_model_bundle(
                    run_cfg, algorithm=algorithm
                ),
                prepare_prompts=lambda run_cfg, tokenizer, **kwargs: _prepare_prompts(
                    run_cfg,
                    tokenizer,
                    algorithm=algorithm,
                    **kwargs,
                ),
                optimizer_factory=_optimizer_factory,
                logger=run_logger,
            )
        except BaseException:
            _close_configured_logger(run_logger)
            raise

    if cfg.mode != "offline":
        raise ValueError("colocated execution supports offline training only")

    bundle = build_model_bundle(cfg, algorithm=algorithm)
    from specforge.launch import build_offline_runtime

    _ensure_offline_vocab_mapping(cfg, bundle, algorithm)
    run_logger = _configured_logger(cfg)
    try:
        trainer = build_offline_runtime(
            hidden_states_path=cfg.data.hidden_states_path,
            eval_hidden_states_path=cfg.data.eval_hidden_states_path or None,
            draft_model=bundle.model,
            target_head=bundle.target_head,
            ttt_length=t.ttt_length,
            max_len=cfg.data.max_length,
            num_epochs=t.num_epochs,
            use_usp_preprocess=(t.attention_backend == "usp"),
            seed=t.seed,
            resume_from=t.resume_from,
            **_common_launch_kwargs(
                cfg,
                bundle,
                algorithm,
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
