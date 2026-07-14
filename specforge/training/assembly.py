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


def _draft_config_path(path: str) -> str:
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"draft model config does not exist: {path}")
    return path


def _target_text_config(config):
    return getattr(config, "text_config", config)


def _torch_dtype(name: str):
    import torch

    return getattr(torch, name)


def _device():
    from specforge.utils import get_local_device

    return get_local_device()


def _load_draft(cfg: Config):
    """Construct the configured draft model without any legacy trainer code."""
    from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig

    strategy = cfg.training.strategy
    draft_config = AutoDraftModelConfig.from_file(
        _draft_config_path(cfg.model.draft_model_config)
    )
    dtype = _torch_dtype(cfg.model.torch_dtype)

    if strategy == "peagle":
        from specforge.modeling.draft.peagle import PEagleDraftModel

        draft_model = PEagleDraftModel(
            draft_config,
            norm_before_residual=cfg.training.norm_before_residual,
        )
    elif strategy == "eagle3":
        draft_model = AutoDraftModel.from_config(
            draft_config,
            attention_backend=cfg.training.attention_backend,
            torch_dtype=dtype,
        )
    else:
        # DFlash, Domino and DSpark resolve through the draft registry (including
        # the projector_type dispatch) and take their attention implementation
        # from the config rather than a constructor keyword.
        draft_config._attn_implementation = cfg.training.attention_backend
        draft_model = AutoDraftModel.from_config(
            draft_config,
            torch_dtype=dtype,
        )

    draft_model = draft_model.to(device=_device(), dtype=dtype)
    expected = {
        "dflash": "DFlashDraftModel",
        "domino": "DominoDraftModel",
        "dspark": "DSparkDraftModel",
        "peagle": "PEagleDraftModel",
    }.get(strategy)
    if expected is not None and type(draft_model).__name__ != expected:
        raise ValueError(
            f"training.strategy={strategy!r} requires {expected}, but "
            f"{cfg.model.draft_model_config!r} builds "
            f"{type(draft_model).__name__}"
        )

    if strategy in ("eagle3", "peagle"):
        if cfg.model.vocab_mapping_path:
            draft_model.load_vocab_mapping(cfg.model.vocab_mapping_path)
        if cfg.model.load_target_embedding:
            draft_model.load_embedding(
                cfg.model.target_model_path,
                embedding_key=cfg.model.embedding_key,
            )
        # P-EAGLE trains its copied embedding; EAGLE3 intentionally freezes it.
        if strategy == "eagle3":
            draft_model.freeze_embedding()
    return draft_config, draft_model


def _load_tokenizer(cfg: Config):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        cfg.model.target_model_path,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )


def _draft_block_size(cfg: Config) -> int:
    """Read the DFlash-family block size from its single source of truth."""
    import json

    with open(
        _draft_config_path(cfg.model.draft_model_config), encoding="utf-8"
    ) as stream:
        draft_config = json.load(stream)
    block_size = draft_config.get("block_size")
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size < 1
    ):
        raise ValueError(
            "DFlash-family draft config must define a positive integer block_size"
        )
    return block_size


def resolve_eagle_capture_layers(cfg: Config, draft_config, target_config) -> List[int]:
    """Resolve EAGLE capture layers: run override, draft config, then default."""
    layers = cfg.model.aux_hidden_state_layer_ids
    if layers is None:
        eagle_config = (
            draft_config.get("eagle_config", {})
            if isinstance(draft_config, dict)
            else getattr(draft_config, "eagle_config", {})
        ) or {}
        layers = eagle_config.get("eagle_aux_hidden_state_layer_ids")
    if layers is None:
        target_config = _target_text_config(target_config)
        num_layers = int(target_config.num_hidden_layers)
        layers = [1, num_layers // 2 - 1, num_layers - 4]
    layers = list(layers)
    if len(layers) != 3 or any(not isinstance(i, int) or i < 0 for i in layers):
        raise ValueError(
            "resolved EAGLE capture layers must contain exactly three "
            f"non-negative integers, got {layers!r}"
        )
    return layers


def _resolve_mask_token_id(cfg: Config, draft_model, tokenizer) -> int:
    from specforge.training.model_utils import resolve_mask_token_id

    return resolve_mask_token_id(
        explicit=cfg.model.mask_token_id,
        tokenizer=tokenizer,
        draft_model=draft_model,
        embedding_vocab_size=int(draft_model.config.vocab_size),
    )


def _build_target_engine(cfg: Config, capture_layers):
    from specforge.inference.target_engine import get_target_engine

    # P-EAGLE consumes the exact EAGLE3 capture contract.
    engine_strategy = (
        "eagle3" if cfg.training.strategy == "peagle" else cfg.training.strategy
    )
    backend_kwargs = {}
    if cfg.model.target_backend == "sglang":
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
            "max_running_requests": cfg.model.sglang_max_running_requests,
            "max_total_tokens": cfg.model.sglang_max_total_tokens,
        }
    target = get_target_engine(
        cfg.model.target_model_path,
        strategy=engine_strategy,
        backend=cfg.model.target_backend,
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=_torch_dtype(cfg.model.torch_dtype),
        device="cuda",
        cache_dir=cfg.model.cache_dir,
        **backend_kwargs,
    )
    target.set_capture_layers(capture_layers)
    return target


def _strategy_kwargs(cfg: Config) -> Dict[str, Any]:
    t = cfg.training
    if t.strategy == "domino":
        return {
            "lambda_start": t.lambda_base_start,
            "decay_ratio": t.lambda_base_decay_ratio,
        }
    return {}


def build_model_bundle(cfg: Config, *, load_target_engine: bool = True) -> ModelBundle:
    """Build the method-specific composite model and frozen target pieces."""
    import torch
    from transformers import AutoConfig

    strategy = cfg.training.strategy
    draft_config, draft_model = _load_draft(cfg)
    tokenizer = _load_tokenizer(cfg) if cfg.mode == "online" else None
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

    target_head = None
    target_engine = None
    capture_layers = (
        resolve_eagle_capture_layers(cfg, draft_config, target_config)
        if cfg.mode == "online" and strategy in ("eagle3", "peagle")
        else None
    )

    if strategy == "eagle3":
        from specforge.core.eagle3 import OnlineEagle3Model

        model = OnlineEagle3Model(
            draft_model=draft_model,
            length=cfg.training.ttt_length,
            attention_backend=cfg.training.attention_backend,
            lk_loss_type=cfg.training.lk_loss_type,
            kl_scale=cfg.training.kl_scale,
            kl_decay=cfg.training.kl_decay,
        ).to(device=_device(), dtype=_torch_dtype(cfg.model.torch_dtype))
        if cfg.mode == "offline":
            from specforge.modeling.target.target_head import TargetHead

            target_head = TargetHead.from_pretrained(
                cfg.model.target_model_path,
                lm_head_key=cfg.model.lm_head_key,
                cache_dir=cfg.model.cache_dir,
                trust_remote_code=cfg.model.trust_remote_code,
            )
    elif strategy == "peagle":
        if cfg.mode != "online":
            raise NotImplementedError(
                "P-EAGLE is supported by the unified entry only in colocated "
                "online text mode"
            )
        from specforge.core.peagle import OnlinePEagleModel

        mask_token_id = _resolve_mask_token_id(cfg, draft_model, tokenizer)
        model = OnlinePEagleModel(
            draft_model=draft_model,
            mask_token_id=mask_token_id,
            num_depths=cfg.training.num_depths,
            down_sample_ratio=cfg.training.down_sample_ratio,
            down_sample_ratio_min=cfg.training.down_sample_ratio_min,
        ).to(device=_device(), dtype=_torch_dtype(cfg.model.torch_dtype))
    else:
        if cfg.mode == "offline":
            raise NotImplementedError(
                f"{strategy} offline features are not a supported public run "
                "mode; use online capture"
            )
        from specforge.core.dflash import (
            OnlineDFlashModel,
            OnlineDominoModel,
            OnlineDSparkModel,
        )
        from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

        mask_token_id = _resolve_mask_token_id(cfg, draft_model, tokenizer)
        draft_model.mask_token_id = mask_token_id
        method_config = getattr(draft_model.config, "dflash_config", None)
        if method_config is None:
            draft_model.config.dflash_config = {}
            method_config = draft_model.config.dflash_config
        method_config["mask_token_id"] = mask_token_id
        method_config["target_layer_ids"] = list(draft_model.target_layer_ids)
        capture_layers = list(draft_model.target_layer_ids)

        target_parts = TargetEmbeddingsAndHead.from_pretrained(
            cfg.model.target_model_path,
            embed_key=cfg.model.embedding_key,
            lm_head_key=cfg.model.lm_head_key,
            cache_dir=cfg.model.cache_dir,
            device=_device().type,
            dtype=_torch_dtype(cfg.model.torch_dtype),
            trust_remote_code=cfg.model.trust_remote_code,
        )
        common = dict(
            draft_model=draft_model,
            target_lm_head=target_parts.lm_head,
            target_embed_tokens=target_parts.embed_tokens,
            mask_token_id=mask_token_id,
            block_size=int(draft_model.block_size),
            attention_backend=cfg.training.attention_backend,
            num_anchors=cfg.training.num_anchors,
            loss_decay_gamma=cfg.training.loss_decay_gamma,
        )
        if strategy == "dflash":
            model = OnlineDFlashModel(
                **common,
                loss_type=cfg.training.loss_type,
                dpace_alpha=cfg.training.dpace_alpha,
            )
        elif strategy == "domino":
            model = OnlineDominoModel(
                **common,
                shift_label=bool(getattr(draft_model, "shift_label", False)),
            )
        elif strategy == "dspark":
            model = OnlineDSparkModel(
                **common,
                dspark_ce_loss_alpha=cfg.training.dspark_ce_loss_alpha,
                dspark_l1_loss_alpha=cfg.training.dspark_l1_loss_alpha,
                dspark_confidence_head_alpha=(
                    cfg.training.dspark_confidence_head_alpha
                ),
            )
        else:  # guarded by Config's Literal, kept defensive for programmatic use
            raise ValueError(f"unsupported training strategy: {strategy}")
        model = model.to(device=_device(), dtype=_torch_dtype(cfg.model.torch_dtype))

    if load_target_engine and cfg.mode == "online":
        if strategy == "dspark":
            raise NotImplementedError(
                "DSpark requires disaggregated online server capture; it cannot "
                "load a colocated target engine"
            )
        target_engine = _build_target_engine(cfg, capture_layers)

    # Keep the composite and target parts bf16 while avoiding accidental target
    # gradients. The optimizer still receives the strategy's trainable module.
    if target_head is not None and isinstance(target_head, torch.nn.Module):
        target_head.requires_grad_(False)
    return ModelBundle(
        model=model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        target_engine=target_engine,
        target_head=target_head,
        target_hidden_size=target_hidden_size,
        target_vocab_size=target_vocab_size,
        draft_vocab_size=draft_vocab_size,
        capture_layers=capture_layers,
        strategy_kwargs=_strategy_kwargs(cfg),
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


def _prompt_cache_key(cfg: Config) -> str:
    import json

    identity = {
        "path": cfg.data.prompts_path or cfg.data.train_data_path,
        "max_length": cfg.data.max_length,
        "chat_template": cfg.data.chat_template,
        "is_preformatted": cfg.data.is_preformatted,
        "train_only_last_turn": cfg.data.train_only_last_turn,
        "max_prompts": cfg.data.max_prompts,
        "target_model": cfg.model.target_model_path,
        "draft_config": cfg.model.draft_model_config,
        "strategy": cfg.training.strategy,
    }
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def _prepare_prompts(cfg: Config, tokenizer) -> List[dict]:
    from specforge.data.prompt_builder import prepare_prompt_tasks

    path = cfg.data.prompts_path or cfg.data.train_data_path
    cache_key = cfg.data.cache_key or _prompt_cache_key(cfg)
    min_loss_tokens = (
        2 * _draft_block_size(cfg)
        if cfg.training.strategy in ("dflash", "domino", "dspark")
        else 1
    )
    return prepare_prompt_tasks(
        path,
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


def _ensure_vocab_mapping(
    cfg: Config, bundle: ModelBundle, prompts: List[dict]
) -> None:
    """Generate the EAGLE-family dataset vocabulary map when not supplied."""
    if cfg.training.strategy not in ("eagle3", "peagle"):
        return
    if (
        cfg.model.vocab_mapping_path
        or bundle.draft_vocab_size == bundle.target_vocab_size
    ):
        return

    key = hashlib.sha256(
        (
            f"{cfg.data.cache_key or _prompt_cache_key(cfg)}:"
            f"{bundle.target_vocab_size}:{bundle.draft_vocab_size}"
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

    counts: Counter = Counter()
    for task in prompts:
        payload = task["payload"]
        for token_id, keep in zip(payload["input_ids"], payload["loss_mask"]):
            if keep:
                counts[int(token_id)] += 1
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


def _common_launch_kwargs(cfg: Config, bundle: ModelBundle) -> Dict[str, Any]:
    t = cfg.training
    return dict(
        strategy=t.strategy,
        optimizer_factory=_optimizer_factory(cfg),
        run_id=cfg.run_id,
        output_dir=cfg.output_dir,
        batch_size=t.batch_size,
        accumulation_steps=t.accumulation_steps,
        max_steps=t.max_steps,
        total_steps=t.total_steps,
        save_interval=t.save_interval,
        max_checkpoints=t.max_checkpoints,
        logger=_logger,
        log_interval=t.log_interval,
        strategy_kwargs=bundle.strategy_kwargs,
    )


def build_training_run(cfg: Config) -> TrainingRun:
    """Assemble the configured colocated or disaggregated run role."""
    t = cfg.training
    if t.role != "producer":
        import torch.distributed as dist

        cfg.validate_world_size(dist.get_world_size() if dist.is_initialized() else 1)
    if t.deployment_mode == "disaggregated":
        from specforge.training.disaggregated import build_disaggregated_run

        return build_disaggregated_run(
            cfg,
            build_model_bundle=build_model_bundle,
            prepare_prompts=_prepare_prompts,
            optimizer_factory=_optimizer_factory,
            logger=_logger,
        )

    bundle = build_model_bundle(cfg)
    common = _common_launch_kwargs(cfg, bundle)
    if cfg.mode == "offline":
        from specforge.launch import build_offline_runtime

        trainer = build_offline_runtime(
            hidden_states_path=cfg.data.hidden_states_path,
            draft_model=bundle.model,
            target_head=bundle.target_head,
            ttt_length=t.ttt_length,
            max_len=cfg.data.max_length,
            num_epochs=t.num_epochs,
            resume_from=t.resume_from,
            **common,
        )
        return TrainingRun(trainer=trainer)

    from specforge.launch import build_online_runtime

    prompts = _prepare_prompts(cfg, bundle.tokenizer)
    if not prompts:
        raise ValueError("online data preparation produced no trainable prompts")
    _ensure_vocab_mapping(cfg, bundle, prompts)
    from specforge.training.schedule import resolve_total_steps

    common["total_steps"] = resolve_total_steps(
        total_steps=t.total_steps,
        max_steps=t.max_steps,
        num_samples=len(prompts) * t.num_epochs,
        batch_size=t.batch_size,
        accumulation_steps=t.accumulation_steps,
        num_epochs=1,
    )
    trainer = build_online_runtime(
        target_model=bundle.target_engine,
        prompts=prompts,
        draft_model=bundle.model,
        target_hidden_size=bundle.target_hidden_size,
        target_vocab_size=bundle.target_vocab_size,
        draft_vocab_size=bundle.draft_vocab_size,
        target_repr="logits" if t.strategy in ("eagle3", "peagle") else None,
        aux_hidden_state_layer_ids=bundle.capture_layers,
        t2d=getattr(bundle.draft_model, "t2d", None),
        prompt_epochs=t.num_epochs,
        **common,
    )
    return TrainingRun(trainer=trainer)


__all__ = [
    "ModelBundle",
    "TrainingRun",
    "build_model_bundle",
    "build_training_run",
    "resolve_eagle_capture_layers",
]
