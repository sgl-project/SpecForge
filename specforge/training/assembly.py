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
"""Package-level assembly for the single ``specforge train`` entry point.

The DataFlow launch module owns topology wiring and the strategy registry owns
per-step loss behavior.  This module is the missing seam between typed run
configuration and those two layers: it builds the draft composite, frozen target
pieces, prompt tasks and optimizer without importing anything from ``scripts``.

Heavy model/data dependencies stay lazy so importing :mod:`specforge.training`
does not load Transformers, datasets, or a target backend.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from specforge.config import Config


@dataclass
class ModelBundle:
    """Objects and capture metadata needed by one configured training run."""

    model: Any
    draft_model: Any
    tokenizer: Any = None
    processor: Any = None
    input_preparer: Any = None
    feature_schema: Any = None
    target_engine: Any = None
    target_head: Any = None
    target_hidden_size: int = 0
    target_vocab_size: int = 0
    draft_vocab_size: int = 0
    capture_layers: Optional[List[int]] = None
    strategy_kwargs: Dict[str, Any] = field(default_factory=dict)


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


def _strategy_spec(cfg: Config):
    from specforge.training.strategies.registry import resolve_strategy

    return resolve_strategy(cfg.training.strategy)


def _load_draft(cfg: Config):
    """Construct the configured draft model without any legacy trainer code."""
    from specforge.modeling.draft.registry import resolve_draft
    from specforge.training.model_loading import resolve_draft_config
    from specforge.training.strategies.registry import resolve_strategy

    spec = resolve_strategy(cfg.training.strategy)
    draft_config = resolve_draft_config(cfg)
    draft_model = spec.assembly.make_draft_model(cfg, draft_config)
    architecture = spec.assembly.draft_config.architecture
    expected_type = resolve_draft(architecture)
    if not isinstance(draft_model, expected_type):
        raise ValueError(
            f"training.strategy={spec.name!r} requires {architecture}, but "
            f"the resolved draft config builds "
            f"{type(draft_model).__name__}"
        )
    return draft_config, draft_model


def _load_online_inputs(cfg: Config):
    """Load text or multimodal input tooling for the shared rollout path."""
    if cfg.model.input_modality == "qwen2_5_vl":
        from transformers import AutoProcessor

        from specforge.data.vlm import QwenVLInputPreparer

        processor = AutoProcessor.from_pretrained(
            cfg.model.target_model_path,
            min_pixels=cfg.data.min_pixels,
            max_pixels=cfg.data.max_pixels,
            cache_dir=cfg.model.cache_dir,
            trust_remote_code=cfg.model.trust_remote_code,
        )
        return (
            processor.tokenizer,
            processor,
            QwenVLInputPreparer(processor, cfg.data.chat_template),
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    return tokenizer, None, None


def _build_target_engine(cfg: Config, capture_layers, spec):
    from specforge.inference.target_engine import get_target_engine

    engine_strategy = spec.assembly.capture_engine_strategy or spec.name
    backend_kwargs = {}
    if cfg.model.target_backend == "sglang":
        target_batch_size = cfg.training.tp_size * cfg.training.batch_size
        backend_kwargs = {
            "attention_backend": cfg.model.sglang_attention_backend,
            "mem_fraction_static": cfg.model.sglang_mem_fraction_static,
            "context_length": (cfg.model.sglang_context_length or cfg.data.max_length),
            "enable_nccl_nvls": cfg.model.sglang_enable_nccl_nvls,
            "enable_symm_mem": cfg.model.sglang_enable_symm_mem,
            "enable_torch_compile": cfg.model.sglang_enable_torch_compile,
            "enable_dp_attention": cfg.model.sglang_enable_dp_attention,
            "enable_dp_lm_head": cfg.model.sglang_enable_dp_lm_head,
            "ep_size": cfg.model.sglang_ep_size,
            "max_running_requests": (
                cfg.model.sglang_max_running_requests or target_batch_size
            ),
            "max_total_tokens": (
                cfg.model.sglang_max_total_tokens
                or target_batch_size * cfg.data.max_length
            ),
        }
    elif cfg.model.target_backend == "hf":
        backend_kwargs = {"input_modality": cfg.model.input_modality}
    target = get_target_engine(
        cfg.model.target_model_path,
        strategy=engine_strategy,
        backend=cfg.model.target_backend,
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=_torch_dtype(cfg.model.torch_dtype),
        device=str(_device()),
        cache_dir=cfg.model.cache_dir,
        **backend_kwargs,
    )
    target.set_capture_layers(capture_layers)
    return target


def _strategy_kwargs(cfg: Config) -> Dict[str, Any]:
    return _strategy_spec(cfg).assembly.make_strategy_kwargs(cfg)


def build_model_bundle(cfg: Config, *, load_target_engine: bool = True) -> ModelBundle:
    """Build the method-specific composite model and frozen target pieces."""
    import torch
    from transformers import AutoConfig

    from specforge.training.strategies.assembly import OnlineCaptureMode
    from specforge.training.strategies.registry import resolve_strategy

    spec = resolve_strategy(cfg.training.strategy)
    draft_config, draft_model = _load_draft(cfg)
    needs_input_tools = spec.assembly.needs_input_tools(cfg, draft_model)
    tokenizer, processor, input_preparer = (
        _load_online_inputs(cfg) if needs_input_tools else (None, None, None)
    )
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

    target_engine = None
    parts = spec.assembly.make_model(
        cfg, draft_model, draft_config, target_config, tokenizer
    )
    if cfg.mode == "online" and parts.capture_layers is None:
        parts.capture_layers = spec.assembly.resolve_capture_layers(
            cfg, draft_config, target_config
        )

    if load_target_engine and cfg.mode == "online":
        if spec.online_capture is OnlineCaptureMode.SERVER_ONLY:
            raise NotImplementedError(
                f"{spec.name} requires disaggregated online server capture; "
                "it cannot load a colocated target engine"
            )
        if spec.online_capture is OnlineCaptureMode.UNSUPPORTED:
            raise NotImplementedError(
                f"online capture for strategy {spec.name!r} is not wired"
            )
        target_engine = _build_target_engine(cfg, parts.capture_layers, spec)

    # Keep the composite and target parts bf16 while avoiding accidental target
    # gradients. The optimizer still receives the strategy's trainable module.
    if parts.target_head is not None and isinstance(parts.target_head, torch.nn.Module):
        parts.target_head.requires_grad_(False)

    return ModelBundle(
        model=parts.model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        processor=processor,
        input_preparer=input_preparer,
        feature_schema=spec.feature_schema_for(cfg.model.input_modality),
        target_engine=target_engine,
        target_head=parts.target_head,
        target_hidden_size=target_hidden_size,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
        capture_layers=parts.capture_layers,
        strategy_kwargs=spec.assembly.make_strategy_kwargs(cfg),
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
        "min_pixels": cfg.data.min_pixels,
        "max_pixels": cfg.data.max_pixels,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _prepare_prompts(
    cfg: Config,
    tokenizer,
    processor=None,
    *,
    path: Optional[str] = None,
    cache_key: Optional[str] = None,
) -> List[dict]:
    """Prepare one prompt source with an optional path/cache namespace override.

    Training keeps the configured cache key. Evaluation supplies its own path
    and derived key so it can never read or overwrite the training prompt cache.
    """
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
    min_loss_tokens = _strategy_spec(cfg).assembly.min_loss_tokens(cfg)
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
        input_modality=cfg.model.input_modality,
        processor=processor,
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
    cfg: Config, bundle: ModelBundle, prompts: List[dict]
) -> None:
    """Generate the online EAGLE-family map from its prepared prompt plan."""
    # Avoid walking (and validating) the prompt payload when no mapping can be
    # needed.  This is common for equal-vocabulary targets and keeps assembly
    # independent of the online prompt schema for non-mapping runs.
    if "online" not in _strategy_spec(cfg).assembly.vocab_mapping_modes:
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


def _ensure_offline_vocab_mapping(cfg: Config, bundle: ModelBundle) -> None:
    """Derive a local-offline map from the exact feature ids and loss masks."""
    if "offline" not in _strategy_spec(cfg).assembly.vocab_mapping_modes:
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


def _dataloader_num_workers(cfg: Config) -> int:
    dataloader_num_workers = cfg.data.dataloader_num_workers
    if dataloader_num_workers is None:
        dataloader_num_workers = _strategy_spec(
            cfg
        ).assembly.default_dataloader_num_workers
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
    cfg: Config, bundle: ModelBundle, *, logger=_logger
) -> Dict[str, Any]:
    t = cfg.training
    # USP shards one logical sample over ``sp_size`` ranks.  Preserve the
    # legacy optimizer-window semantics: one user accumulation unit represents
    # a complete logical sequence, not one local sequence shard.
    accumulation_steps = t.accumulation_steps
    if t.attention_backend == "usp":
        accumulation_steps *= t.sp_ulysses_size * t.sp_ring_size
    return dict(
        strategy=t.strategy,
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
        dataloader_num_workers=_dataloader_num_workers(cfg),
        profiling_options=_profiling_options(cfg),
    )


def build_training_run(cfg: Config) -> TrainingRun:
    """Assemble the configured colocated or disaggregated run role."""
    t = cfg.training
    if t.role != "producer":
        import torch.distributed as dist

        cfg.validate_world_size(dist.get_world_size() if dist.is_initialized() else 1)
    if t.deployment_mode == "disaggregated":
        from specforge.training.disaggregated import build_disaggregated_run

        run_logger = _configured_logger(cfg)
        try:
            return build_disaggregated_run(
                cfg,
                build_model_bundle=build_model_bundle,
                prepare_prompts=_prepare_prompts,
                optimizer_factory=_optimizer_factory,
                logger=run_logger,
            )
        except BaseException:
            _close_configured_logger(run_logger)
            raise

    bundle = build_model_bundle(cfg)
    if cfg.mode == "offline":
        from specforge.launch import build_offline_runtime

        _ensure_offline_vocab_mapping(cfg, bundle)

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
                **_common_launch_kwargs(cfg, bundle, logger=run_logger),
            )
        except BaseException:
            _close_configured_logger(run_logger)
            raise
        return TrainingRun(trainer=trainer)

    from specforge.launch import (
        _plan_online_prompt_stream,
        _preposition_online_prompts,
        _target_dp_layout,
        build_online_runtime,
    )

    prompts = _prepare_prompts(cfg, bundle.tokenizer, bundle.processor)
    if not prompts:
        raise ValueError("online data preparation produced no trainable prompts")
    eval_prompts = None
    if cfg.data.eval_data_path:
        eval_prompts = _prepare_prompts(
            cfg,
            bundle.tokenizer,
            bundle.processor,
            path=cfg.data.eval_data_path,
            cache_key="eval-" + _prompt_cache_key(cfg, path=cfg.data.eval_data_path),
        )
        if not eval_prompts:
            raise ValueError(
                "online eval data preparation produced no trainable prompts"
            )
    _ensure_vocab_mapping(cfg, bundle, prompts)
    source_prompt_count = len(prompts)
    prompts = _plan_online_prompt_stream(
        prompts,
        num_epochs=t.num_epochs,
        seed=t.seed,
        tp_size=t.tp_size,
        batch_size=t.batch_size,
        shuffle=True,
    )
    if not prompts:
        raise ValueError(
            "online prompt planning produced no complete target batch after "
            "target-DP sharding; provide at least tp_size * batch_size prompts "
            "per target-DP replica"
        )
    # The flattened target stream contains all logical epochs, each truncated
    # independently to complete TP-wide capture batches. Every TP rank trains
    # one ``1 / tp_size`` slice, so checkpoint dataset_size is rank-local.
    dataset_size = len(prompts) // t.tp_size
    if eval_prompts is not None:
        eval_prompts = _plan_online_prompt_stream(
            eval_prompts,
            num_epochs=1,
            seed=t.seed,
            tp_size=t.tp_size,
            batch_size=t.batch_size,
            shuffle=False,
        )

    resume_state = None
    remaining_prompts = prompts
    if t.resume_from is not None:
        from specforge.training.checkpoint import CheckpointManager

        resume_state = CheckpointManager.read_resume_state(t.resume_from)
        checkpoint_epoch = int(resume_state.get("epoch", 0))
        can_preposition = all(
            resume_state.get(key) in (None, current)
            for key, current in (
                ("dataset_size", dataset_size),
                ("batch_size", t.batch_size),
                ("tp_size", t.tp_size),
            )
        )
        if checkpoint_epoch == 0 and can_preposition:
            remaining_prompts = _preposition_online_prompts(
                prompts,
                local_samples=int(resume_state.get("epoch_samples", 0)),
                tp_size=t.tp_size,
            )
        elif checkpoint_epoch == 1 or not can_preposition:
            remaining_prompts = []
        else:
            # Trainer owns the canonical resume validation and error text. Avoid
            # indexing an invalid epoch while preserving the loaded state for it.
            remaining_prompts = []

    from specforge.training.schedule import resolve_total_steps

    effective_accumulation_steps = t.accumulation_steps
    if t.attention_backend == "usp":
        effective_accumulation_steps *= t.sp_ulysses_size * t.sp_ring_size
    total_steps = resolve_total_steps(
        total_steps=t.total_steps,
        max_steps=t.max_steps,
        num_samples=dataset_size,
        batch_size=t.batch_size,
        accumulation_steps=effective_accumulation_steps,
        num_epochs=1,
    )
    _, target_dp_size = _target_dp_layout()
    checkpoint_extra = {
        "online_prompt_plan_version": 1,
        "prompt_source_size": source_prompt_count,
        "prompt_seed": t.seed,
        "prompt_epochs": t.num_epochs,
        "target_dp_size": target_dp_size,
    }
    run_logger = _configured_logger(cfg)
    try:
        common = _common_launch_kwargs(cfg, bundle, logger=run_logger)
        common["total_steps"] = total_steps
        trainer = build_online_runtime(
            target_model=bundle.target_engine,
            prompts=remaining_prompts,
            eval_prompts=eval_prompts,
            draft_model=bundle.model,
            target_hidden_size=bundle.target_hidden_size,
            target_vocab_size=bundle.target_vocab_size,
            draft_vocab_size=bundle.draft_vocab_size,
            target_repr=_strategy_spec(cfg).assembly.colocated_target_repr,
            aux_hidden_state_layer_ids=bundle.capture_layers,
            t2d=getattr(bundle.draft_model, "t2d", None),
            prompt_epochs=1,
            feature_schema=bundle.feature_schema,
            input_preparer=bundle.input_preparer,
            shard_target_output=cfg.model.shard_target_output,
            device=str(_device()),
            resume_from=t.resume_from,
            resume_state=resume_state,
            dataset_size=dataset_size,
            checkpoint_extra=checkpoint_extra,
            **common,
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
