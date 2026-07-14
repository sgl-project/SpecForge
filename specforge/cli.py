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

Only an online-disaggregated consumer may run under multi-rank ``torchrun``:
    torchrun --standalone --nproc_per_node 8 $(which specforge) train --config run.yaml

Model/data assembly lives in :mod:`specforge.training.assembly`; this module is
deliberately limited to command parsing and distributed process lifecycle.
"""

from __future__ import annotations

import argparse
import os
import socket
from typing import List, Optional

from specforge.config import Config, load_config


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


def _train(cfg: Config) -> int:
    from accelerate.utils import set_seed

    from specforge.distributed import destroy_distributed, init_distributed

    set_seed(cfg.training.seed)
    if cfg.training.role == "producer":
        # A server-capture/offline-ingest producer owns no trainer process
        # group and must not initialize CUDA merely to publish feature refs.
        from specforge.training.assembly import build_training_run

        return build_training_run(cfg).run()
    _bootstrap_single_process_env()
    init_distributed(tp_size=1)
    try:
        import torch.distributed as dist

        _validate_world_size(cfg, dist.get_world_size())
        from specforge.training.assembly import build_training_run

        return build_training_run(cfg).run()
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
        _train(cfg)
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
