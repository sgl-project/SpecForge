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
constructs algorithm-specific target policy. Online capture uses either a local
SGLang runner or an external SGLang server through the disaggregated runtime.

Heavy model/data dependencies stay lazy so importing :mod:`specforge.training`
does not load Transformers, datasets, or a target backend.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from specforge.algorithms.contracts import FeatureMode
from specforge.algorithms.registry import AlgorithmRegistration
from specforge.config import SGLANG_CAPTURE_CONTEXT_HEADROOM, Config
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


def build_model_bundle(cfg: Config, *, algorithm: AlgorithmRegistration) -> ModelBundle:
    """Build the method-specific composite model and frozen target pieces."""
    from specforge.torch_environment import configure_flex_attention_inductor

    # Programmatic callers do not pass through the CLI's early environment
    # setup, so install the same safe default before importing model modules.
    configure_flex_attention_inductor(cfg.training.attention_backend)
    import torch

    from specforge.modeling.target.target_utils import (
        load_target_config,
        target_text_config,
    )
    from specforge.modeling.target.target_utils import (
        target_vocab_size as resolve_target_vocab_size,
    )

    provider = algorithm.providers.model
    draft_config, draft_model = _load_draft(cfg, algorithm)
    needs_input_tools = provider.needs_input_tools(cfg, draft_model)
    input_tools = _load_input_tools(cfg, algorithm) if needs_input_tools else None
    target_config = load_target_config(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )
    text_config = target_text_config(target_config)
    target_hidden_size = int(text_config.hidden_size)
    target_vocab_size = resolve_target_vocab_size(target_config)
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
            lr_scheduler=t.lr_scheduler,
            total_steps=self.total_steps,
            offload_master=t.optimizer_cpu_offload,
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
    if cfg.training.role != "producer":
        import torch.distributed as dist

        # A distributed run has one logical metric stream.  Letting every rank
        # create W&B/MLflow runs or write the same TensorBoard directory both
        # duplicates metrics and makes large jobs increasingly fragile.
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return None
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


def _training_prompt_cache_ready_file(cfg: Config, tokenizer) -> Optional[str]:
    """Return the completion sentinel for the training prompt map cache."""
    if not cfg.data.cache_dir:
        return None
    cache_key = cfg.data.cache_key or _prompt_cache_key(cfg, tokenizer=tokenizer)
    return os.path.join(cfg.data.cache_dir, f"{cache_key}.ready")


def _prepare_colocated_prompts(
    cfg: Config,
    tokenizer,
    *,
    algorithm: AlgorithmRegistration,
    draft_config,
) -> Sequence[dict]:
    """Populate a reusable prompt cache once before colocated ranks read it.

    Raw prompt tokenization can spawn many worker processes.  Starting that
    work independently on every training rank causes a startup storm on large
    HSDP jobs and lets several ranks race to write the same Arrow cache.  Rank
    zero populates shared storage first; for node-local cache paths, local rank
    zero on each remaining node fills that node's copy.  Every other rank then
    takes the normal cache-hit path and receives its own memory-mapped dataset.
    """

    ready_file = _training_prompt_cache_ready_file(cfg, tokenizer)

    def prepare(*, mark_ready: bool = False) -> Sequence[dict]:
        prompts = _prepare_prompts(
            cfg,
            tokenizer,
            algorithm=algorithm,
            draft_config=draft_config,
        )
        if mark_ready and ready_file is not None:
            os.makedirs(cfg.data.cache_dir, exist_ok=True)
            temporary = f"{ready_file}.{os.getpid()}.tmp"
            with open(temporary, "w", encoding="utf-8") as stream:
                stream.write("ready\n")
            os.replace(temporary, ready_file)
        return prompts

    import torch.distributed as dist

    if (
        ready_file is None
        or not dist.is_available()
        or not dist.is_initialized()
        or dist.get_world_size() == 1
    ):
        return prepare()

    def prepare_collectively(should_prepare: bool) -> Optional[Sequence[dict]]:
        prompts = None
        error = None
        if should_prepare:
            try:
                prompts = prepare(mark_ready=True)
            except BaseException as exc:
                error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
        errors = [None] * dist.get_world_size()
        dist.all_gather_object(errors, error)
        failures = [item for item in errors if item is not None]
        if failures:
            raise RuntimeError(
                "colocated prompt-cache preparation failed: " + "; ".join(failures)
            )
        return prompts

    prompts = prepare_collectively(dist.get_rank() == 0)

    # On a shared filesystem rank zero's cache is now visible everywhere.  On
    # node-local storage, exactly one process per missing node performs the
    # same deterministic build.
    node_prompts = prepare_collectively(
        prompts is None
        and int(os.environ.get("LOCAL_RANK", "0")) == 0
        and not os.path.isfile(ready_file)
    )
    if prompts is None:
        prompts = node_prompts

    return prompts if prompts is not None else prepare()


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


def _ensure_online_vocab_mapping(
    cfg: Config,
    bundle: ModelBundle,
    algorithm: AlgorithmRegistration,
    prompts: Sequence[dict],
) -> None:
    """Derive a colocated streaming vocabulary map from prepared prompts."""
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


def _colocated_sglang_kwargs(cfg: Config) -> Dict[str, Any]:
    """Map the typed ``sglang_*`` namespace to local ``ServerArgs`` fields."""
    context_length = cfg.model.sglang_context_length or (
        cfg.data.max_length + SGLANG_CAPTURE_CONTEXT_HEADROOM
    )
    target_batch_size = cfg.training.batch_size * cfg.training.tp_size
    overrides = {
        "sglang_context_length": context_length,
        "sglang_max_running_requests": (
            cfg.model.sglang_max_running_requests or target_batch_size
        ),
        "sglang_max_total_tokens": (
            cfg.model.sglang_max_total_tokens or target_batch_size * context_length
        ),
        "sglang_dp_size": 1,
    }
    kwargs = {}
    for name in type(cfg.model).model_fields:
        if not name.startswith("sglang_"):
            continue
        value = overrides.get(name, getattr(cfg.model, name))
        if value is not None:
            kwargs[name.removeprefix("sglang_")] = value
    return kwargs


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
    source. Online training selects either bounded in-process SGLang capture or
    the external-server disaggregated transport from ``deployment.mode``.
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

    bundle = build_model_bundle(cfg, algorithm=algorithm)
    if cfg.mode == "offline":
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

    from specforge.launch import (
        _plan_online_prompt_stream,
        _preposition_online_prompts,
        _target_dp_layout,
        build_colocated_online_runtime,
    )

    prompts = _prepare_colocated_prompts(
        cfg,
        bundle.input_tools,
        algorithm=algorithm,
        draft_config=bundle.draft_config,
    )
    if not prompts:
        raise ValueError("online data preparation produced no trainable prompts")
    _ensure_online_vocab_mapping(cfg, bundle, algorithm, prompts)
    source_prompt_count = len(prompts)
    prompt_seed = t.seed if t.prompt_seed is None else t.prompt_seed
    prompts = _plan_online_prompt_stream(
        prompts,
        num_epochs=t.num_epochs,
        seed=prompt_seed,
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
    dataset_size = len(prompts) // t.tp_size

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
        else:
            remaining_prompts = []

    from specforge.training.schedule import resolve_total_steps

    total_steps = resolve_total_steps(
        total_steps=t.total_steps,
        max_steps=t.max_steps,
        num_samples=dataset_size,
        batch_size=t.batch_size,
        accumulation_steps=t.accumulation_steps,
        num_epochs=1,
    )
    target_dp_rank, target_dp_size = _target_dp_layout()
    checkpoint_extra = {
        "online_prompt_plan_version": 2,
        "prompt_source_size": source_prompt_count,
        "prompt_seed": prompt_seed,
        "prompt_epochs": t.num_epochs,
        "target_dp_rank": target_dp_rank,
        "target_dp_size": target_dp_size,
        "colocated_capture": "sglang_hidden_state_v1",
    }
    policy = cfg.deployment.colocated
    synchronize_after_capture = (
        True if policy is None else policy.synchronize_after_capture
    )
    zero_copy_features = True if policy is None else policy.zero_copy_features

    run_logger = _configured_logger(cfg)
    try:
        import torch

        from specforge.inference.adapters import LocalSGLangCaptureAdapter
        from specforge.offline_capture import load_offline_capture

        target_capture = load_offline_capture(
            cfg.model.target_model_path,
            torch_dtype=getattr(torch, cfg.model.torch_dtype),
            trust_remote_code=cfg.model.trust_remote_code,
            **_colocated_sglang_kwargs(cfg),
        )
        streaming = algorithm.providers.server_streaming_for(cfg.model.input_modality)
        target_capture.set_capture_layers(
            bundle.capture_layers,
            capture_method=streaming.capture_method,
        )
        feature_source = LocalSGLangCaptureAdapter(
            target_capture,
            provider=streaming,
            synchronize_after_capture=synchronize_after_capture,
        )
        common = _common_launch_kwargs(
            cfg,
            bundle,
            algorithm,
            logger=run_logger,
        )
        common["total_steps"] = total_steps
        trainer = build_colocated_online_runtime(
            prompts=remaining_prompts,
            feature_source=feature_source,
            draft_model=bundle.model,
            target_head=bundle.target_head,
            target_hidden_size=bundle.target_hidden_size,
            target_vocab_size=bundle.target_vocab_size,
            draft_vocab_size=bundle.draft_vocab_size,
            target_repr=streaming.target_representation,
            aux_hidden_state_layer_ids=bundle.capture_layers,
            resume_from=t.resume_from,
            resume_state=resume_state,
            dataset_size=dataset_size,
            checkpoint_extra=checkpoint_extra,
            zero_copy_features=zero_copy_features,
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
