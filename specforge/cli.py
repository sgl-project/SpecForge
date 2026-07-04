# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The ``specforge`` console entry point.

``specforge train --config run.yaml [section.field=value ...]`` builds the
validated :class:`~specforge.config.Config`, assembles the models, and runs
training through the DataFlow launch builders — the same wiring the
programmatic path uses, behind one typed config.

Run under torchrun for multi-rank:
    torchrun --standalone --nproc_per_node 8 $(which specforge) train --config run.yaml

v1 drives the ``eagle3`` strategy end-to-end (offline hidden-state features or
online rollout from pre-tokenized prompts). DFlash/Domino model assembly still
lives in their dedicated scripts; their strategy configs are accepted but
rejected here with a pointer.
"""

from __future__ import annotations

import argparse
import json
from typing import List, Optional

from specforge.config import Config, load_config


def _load_prompts(path: str) -> List[dict]:
    """Read pre-tokenized prompts jsonl into PromptTask payloads."""
    prompts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompts.append(
                {
                    "payload": {
                        "input_ids": record["input_ids"],
                        "loss_mask": record["loss_mask"],
                    }
                }
            )
    return prompts


def _build_eagle3_model(cfg: Config):
    import torch

    from specforge import AutoDraftModelConfig, AutoEagle3DraftModel, OnlineEagle3Model

    draft_config = AutoDraftModelConfig.from_file(cfg.model.draft_model_config)
    draft_model = AutoEagle3DraftModel.from_config(
        draft_config,
        attention_backend=cfg.training.attention_backend,
        torch_dtype=torch.bfloat16,
    ).cuda()
    if cfg.model.vocab_mapping_path:
        draft_model.load_vocab_mapping(cfg.model.vocab_mapping_path)
    if cfg.model.load_target_embedding:
        draft_model.load_embedding(
            cfg.model.target_model_path, embedding_key=cfg.model.embedding_key
        )
    draft_model.freeze_embedding()
    return OnlineEagle3Model(
        draft_model=draft_model,
        length=cfg.training.ttt_length,
        attention_backend=cfg.training.attention_backend,
    ).cuda()


def _optimizer_factory(cfg: Config):
    from specforge.optimizer import BF16Optimizer

    def factory(draft_module):
        return BF16Optimizer(
            draft_module,
            lr=cfg.training.learning_rate,
            max_grad_norm=cfg.training.max_grad_norm,
            warmup_ratio=cfg.training.warmup_ratio,
            total_steps=cfg.training.total_steps or cfg.training.max_steps or 10_000,
        )

    return factory


def build_from_config(cfg: Config):
    """Assemble ``(trainer, loader, drive_rollout | None)`` from a Config.

    The wiring is exactly the programmatic ``build_offline_runtime`` /
    ``build_online_runtime`` path — the config only parameterizes it.
    """
    if cfg.training.strategy != "eagle3":
        raise NotImplementedError(
            f"specforge train drives strategy 'eagle3'; for "
            f"{cfg.training.strategy!r} use its dedicated script "
            f"(scripts/train_{cfg.training.strategy}.py)"
        )

    t = cfg.training
    common = dict(
        strategy=t.strategy,
        optimizer_factory=_optimizer_factory(cfg),
        run_id=cfg.run_id,
        output_dir=cfg.output_dir,
        ttt_length=t.ttt_length,
        batch_size=t.batch_size,
        accumulation_steps=t.accumulation_steps,
        max_steps=t.max_steps,
        total_steps=t.total_steps,
        save_interval=t.save_interval,
        eval_interval=t.eval_interval,
        max_checkpoints=t.max_checkpoints,
        tp_size=t.tp_size,
        sp_ulysses_size=t.sp_ulysses_size,
        sp_ring_size=t.sp_ring_size,
        logger=lambda metrics, step: print(f"step {step}: {metrics}", flush=True),
        resume_from=t.resume_from,
    )
    eagle3_model = _build_eagle3_model(cfg)

    if cfg.mode == "offline":
        from specforge.launch import build_offline_runtime
        from specforge.modeling.target import TargetHead

        target_head = TargetHead.from_pretrained(
            cfg.model.target_model_path, lm_head_key=cfg.model.lm_head_key
        )
        trainer, loader = build_offline_runtime(
            hidden_states_path=cfg.data.hidden_states_path,
            eagle3_model=eagle3_model,
            target_head=target_head,
            max_len=cfg.data.max_length,
            num_epochs=t.num_epochs,
            log_interval=t.log_interval,
            deployment_mode=t.deployment_mode,
            metadata_db_path=t.metadata_db_path,
            **common,
        )
        return trainer, loader, None

    import torch as _torch
    from transformers import AutoConfig

    from specforge.inference.target_engine import get_target_engine
    from specforge.launch import build_online_runtime

    target = get_target_engine(
        cfg.model.target_model_path,
        strategy=t.strategy,
        backend=cfg.model.target_backend,
        trust_remote_code=cfg.model.trust_remote_code,
        torch_dtype=getattr(_torch, cfg.model.torch_dtype),
        device="cuda",
    )
    # The capture contract must be set BEFORE the first rollout: None derives
    # the backend defaults; explicit ids pin the aux layers.
    target.set_capture_layers(cfg.model.aux_hidden_state_layer_ids)
    target_config = AutoConfig.from_pretrained(
        cfg.model.target_model_path, trust_remote_code=cfg.model.trust_remote_code
    )
    trainer, loader, workers, controller, drive_rollout = build_online_runtime(
        target_model=target,
        prompts=_load_prompts(cfg.data.prompts_path),
        eagle3_model=eagle3_model,
        target_hidden_size=int(target_config.hidden_size),
        target_vocab_size=int(target_config.vocab_size),
        target_repr="logits",
        aux_hidden_state_layer_ids=cfg.model.aux_hidden_state_layer_ids,
        num_epochs=1,  # online rollout output is a consume-once stream
        log_interval=t.log_interval,
        **common,
    )
    return trainer, loader, drive_rollout


def _train(cfg: Config) -> int:
    from accelerate.utils import set_seed

    from specforge.distributed import destroy_distributed, init_distributed

    set_seed(cfg.training.seed)
    init_distributed(
        tp_size=cfg.training.tp_size,
        sp_ulysses_size=cfg.training.sp_ulysses_size,
        sp_ring_size=cfg.training.sp_ring_size,
    )
    try:
        trainer, loader, drive_rollout = build_from_config(cfg)
        if drive_rollout is not None:
            produced = drive_rollout()
            print(f"[online] rollout produced {produced} samples", flush=True)
        return trainer.fit(loader)
    finally:
        destroy_distributed()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="specforge")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train a draft model from a typed config")
    train.add_argument("--config", required=True, help="YAML or JSON run config")
    train.add_argument(
        "overrides",
        nargs="*",
        help="dotted overrides, e.g. training.learning_rate=1e-4",
    )
    args = parser.parse_args(argv)

    if args.command == "train":
        cfg = load_config(args.config, args.overrides)
        _train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
