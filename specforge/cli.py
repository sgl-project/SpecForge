# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The single public SpecForge training entry point.

``specforge train --config run.yaml [section.field=value ...]`` builds the
validated :class:`~specforge.config.Config`, assembles the models, and runs
training through the DataFlow launch builders — the same wiring the
programmatic path uses, behind one typed config.

``deployment.trainer`` defines the process topology. The CLI self-launches
multi-rank workers and recognizes an existing torchrun worker environment
without nesting another launcher.

Model/data assembly lives in :mod:`specforge.training.assembly`; this module is
deliberately limited to command parsing and distributed process lifecycle.
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
from contextlib import contextmanager
from typing import Iterator, List, Optional

from specforge.config import Config, load_config


class _WorkerTermination(BaseException):
    """Translate a process signal into normal Python stack unwinding."""

    def __init__(self, signum: int):
        self.signum = signum


@contextmanager
def _worker_signal_unwind() -> Iterator[None]:
    """Make worker termination run training and distributed cleanup blocks.

    Managed supervisors terminate worker process groups with SIGTERM.  Python's
    default SIGTERM action exits immediately, bypassing ``finally`` blocks.  The
    first managed signal is therefore raised as a ``BaseException``; subsequent
    signals are ignored while cleanup runs, after which the original handlers
    are restored.  A supervising parent may still enforce its grace period with
    SIGKILL if cleanup cannot finish.
    """
    managed_signals = [signal.SIGINT, signal.SIGTERM]
    if hasattr(signal, "SIGHUP"):
        managed_signals.append(signal.SIGHUP)
    previous_handlers = {}

    def unwind(signum, _frame):
        for installed in previous_handlers:
            signal.signal(installed, signal.SIG_IGN)
        raise _WorkerTermination(signum)

    try:
        for signum in managed_signals:
            try:
                previous_handlers[signum] = signal.signal(signum, unwind)
            except ValueError:
                # Embedded callers may execute the CLI from a non-main thread,
                # where Python does not permit signal handler installation.
                for installed, handler in previous_handlers.items():
                    signal.signal(installed, handler)
                previous_handlers.clear()
                break
        yield
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def _bootstrap_single_process_env() -> None:
    """Provide ``env://`` rendezvous values for a direct one-GPU invocation."""
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    present = [name for name in required if name in os.environ]
    if present:
        missing = [name for name in required if name not in os.environ]
        if missing:
            raise ValueError(
                "distributed environment is incomplete; present="
                f"{present}, missing={missing}. Launch with torchrun or unset the "
                "partial distributed variables for a one-process run."
            )
        return

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as rendezvous:
        rendezvous.bind(("127.0.0.1", 0))
        port = rendezvous.getsockname()[1]
    os.environ.update(
        {
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
        }
    )


def _validate_world_size(cfg: Config, world_size: int) -> None:
    cfg.validate_world_size(world_size)


def _train(resolved) -> int:
    from accelerate.utils import set_seed

    cfg = resolved.config
    # Make the typed recipe authoritative for the backend's existing FSDP
    # sharding seam in both direct and managed-local worker processes.
    os.environ["FSDP_SHARDING"] = cfg.training.fsdp_sharding
    set_seed(cfg.training.seed)
    if cfg.training.role == "producer":
        # A server-capture/offline-ingest producer owns no trainer process
        # group and must not initialize CUDA merely to publish feature refs.
        from specforge.application import build_application_run

        return build_application_run(resolved).run()

    from specforge.distributed import destroy_distributed, init_distributed

    _bootstrap_single_process_env()
    _validate_world_size(cfg, int(os.environ["WORLD_SIZE"]))
    init_distributed(
        timeout=cfg.training.dist_timeout,
        tp_size=cfg.training.tp_size,
        sp_ulysses_size=cfg.training.sp_ulysses_size,
        sp_ring_size=cfg.training.sp_ring_size,
    )
    try:
        import torch.distributed as dist

        _validate_world_size(cfg, dist.get_world_size())
        from specforge.application import build_application_run

        return build_application_run(resolved).run()
    finally:
        destroy_distributed()


def _config_for_role(
    cfg: Config, role: str, consumer_id: Optional[str] = None
) -> Config:
    """Resolve a launch role without changing the persisted run config.

    A shared disaggregated config may contain trainer-only state used by the
    consumer child.  The capture-only producer must ignore that state when the
    launcher derives its role from the shared config.
    """
    raw = cfg.model_dump()
    raw["training"]["role"] = role
    disaggregated = raw["deployment"].get("disaggregated")
    fanout = disaggregated.get("windowed_fanout") if disaggregated else None
    managed_local = disaggregated.get("managed_local") if disaggregated else None
    if fanout is not None and managed_local is not None:
        devices = managed_local["trainer_cuda_visible_devices"]
        for consumer, device in zip(fanout["consumers"], devices):
            consumer["cuda_visible_device"] = device
    if disaggregated is not None and disaggregated.get("managed_local") is not None:
        # This field describes services owned by the parent supervisor.  A role
        # child consumes the already-derived environment and must not attempt to
        # validate or own that stack again.
        disaggregated["managed_local"] = None
    if role == "consumer" and fanout is not None:
        consumer_id = consumer_id or os.environ.get("SPECFORGE_FANOUT_CONSUMER_ID")
        matches = [
            consumer
            for consumer in fanout["consumers"]
            if consumer["consumer_id"] == consumer_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"unknown or missing windowed fanout consumer {consumer_id!r}"
            )
        consumer = matches[0]
        raw["training"].update(
            {
                "seed": consumer["seed"],
                "loss_type": consumer["loss_type"],
                "loss_decay_gamma": consumer["loss_decay_gamma"],
                "dpace_alpha": consumer["dpace_alpha"],
                "num_anchors": consumer["num_anchors"],
                "learning_rate": consumer["learning_rate"],
                "warmup_ratio": consumer["warmup_ratio"],
            }
        )
        if consumer["draft_block_size"] is not None:
            raw["model"]["draft_block_size"] = consumer["draft_block_size"]
        raw["output_dir"] = os.path.join(raw["output_dir"], consumer_id)
    if role == "producer":
        raw["profiling"]["enabled"] = False
    return Config.model_validate(raw)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="specforge")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="train a draft model from a typed config")
    train.add_argument("-c", "--config", required=True, help="YAML or JSON run config")
    train.add_argument(
        "--role",
        choices=("auto", "all", "producer", "consumer", "both"),
        default="auto",
        help=(
            "launch selection (default: offline local all or online/disaggregated "
            "producer+consumer)"
        ),
    )
    train.add_argument(
        "--node-rank",
        type=int,
        default=None,
        help="node-local rank for an explicit multi-node trainer launch",
    )
    train.add_argument(
        "--consumer-id",
        default=None,
        help="select one independent consumer from windowed_fanout",
    )
    train.add_argument(
        "--plan",
        action="store_true",
        help="print the resolved process plan without starting workers",
    )
    train.add_argument(
        "overrides",
        nargs="*",
        help="dotted overrides, e.g. training.learning_rate=1e-4",
    )
    export = sub.add_parser(
        "export", help="materialize a runtime checkpoint as a model directory"
    )
    export.add_argument("--to", choices=("hf", "sglang"), required=True)
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--draft-config", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--vocab-mapping", default=None)
    export.add_argument(
        "--embedding-source",
        default=None,
        help="target model path supplying a frozen embedding for HF export",
    )
    export.add_argument("--embedding-key", default="model.embed_tokens.weight")
    args = parser.parse_args(argv)

    if args.command == "train":
        cfg = load_config(args.config, args.overrides)
        from specforge.application import bind_run, resolve_run
        from specforge.launch_plan import build_launch_plan, run_commands

        resolved = resolve_run(cfg)
        plan = build_launch_plan(
            resolved.config,
            algorithm=resolved.algorithm,
            config_path=args.config,
            overrides=args.overrides,
            requested_role=args.role,
            consumer_id=args.consumer_id,
            node_rank=args.node_rank,
        )
        if args.plan:
            print(plan.render())
            return 0
        if plan.kind == "worker":
            os.environ.update(plan.worker_env)
            role_config = _config_for_role(resolved.config, plan.role, args.consumer_id)
            try:
                with _worker_signal_unwind():
                    _train(bind_run(role_config, resolved.algorithm))
            except _WorkerTermination as received:
                return 128 + received.signum
            return 0
        return run_commands(plan)
    elif args.to == "hf":
        from specforge.export.to_hf import export_to_hf

        export_to_hf(
            args.checkpoint,
            args.draft_config,
            args.output_dir,
            vocab_mapping_path=args.vocab_mapping,
            embedding_source=args.embedding_source,
            embedding_key=args.embedding_key,
        )
    else:
        if args.embedding_source is not None:
            parser.error("--embedding-source is only valid with --to hf")
        from specforge.export.to_sglang import export_to_sglang

        export_to_sglang(
            args.checkpoint,
            args.draft_config,
            args.output_dir,
            vocab_mapping_path=args.vocab_mapping,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
