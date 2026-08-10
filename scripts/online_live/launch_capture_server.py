#!/usr/bin/env python3
# coding=utf-8
"""One-command live capture server: resolve flags from the run config and exec.

Reads an online-live SpecForge config, resolves the capture contract (method,
aux layer ids, context length) exactly like the producer does, applies the
online-live SGLang patch if needed, and execs ``sglang.launch_server`` with
the derived Mooncake environment and capture flags.

Example:
    python scripts/online_live/launch_capture_server.py \\
        --config examples/configs/qwen3-4b-dspark-live.yaml --cuda 0

Extra SGLang flags pass through after ``--``:
    ... --cuda 0 -- --mem-fraction-static 0.8
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="online-live run YAML")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument(
        "--intake-url", default=None, help="default: http://127.0.0.1:<live.port>"
    )
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument(
        "--skip-patch", action="store_true", help="do not (re)apply the live patch"
    )
    return parser.parse_known_args()


def main() -> None:
    from specforge.application import resolve_capture_contract
    from specforge.config import SGLANG_CAPTURE_CONTEXT_HEADROOM, Config
    from specforge.launch_plan import _sglang_argv

    args, extra = parse_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    cfg = Config.from_file(args.config)
    deployment = cfg.deployment.disaggregated
    live = deployment.live if deployment is not None else None
    if live is None:
        raise SystemExit(f"{args.config} is not an online-live config")
    contract = resolve_capture_contract(cfg)

    if not args.skip_patch:
        subprocess.check_call(
            [str(REPO_ROOT / "scripts/apply_sglang_spec_capture_patch.sh"), "--live"]
        )

    env = dict(os.environ)
    if live.mooncake is not None:
        mooncake = live.mooncake
        env.setdefault(
            "MOONCAKE_METADATA_SERVER",
            f"http://127.0.0.1:{mooncake.metadata_port}/metadata",
        )
        env.setdefault(
            "MOONCAKE_MASTER_SERVER_ADDR", f"127.0.0.1:{mooncake.rpc_port}"
        )
        env.setdefault("MOONCAKE_LOCAL_HOSTNAME", mooncake.local_hostname)
        env.setdefault("MOONCAKE_PROTOCOL", mooncake.protocol)
        env.setdefault(
            "MOONCAKE_GLOBAL_SEGMENT_SIZE", str(mooncake.global_segment_size_bytes)
        )
        env.setdefault(
            "MOONCAKE_LOCAL_BUFFER_SIZE", str(mooncake.local_buffer_size_bytes)
        )
        env.setdefault("MC_TRANSFER_TIMEOUT", "300")
        env.setdefault("MC_TCP_BIND_ADDRESS", mooncake.local_hostname)
    else:
        for name, value in (
            ("MOONCAKE_METADATA_SERVER", deployment.mooncake_metadata_server),
            ("MOONCAKE_MASTER_SERVER_ADDR", deployment.mooncake_master_server_addr),
        ):
            if value:
                env.setdefault(name, value)
    if args.cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda
    env.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

    intake_url = args.intake_url or f"http://127.0.0.1:{live.port}"
    context_length = cfg.model.sglang_context_length or (
        cfg.data.max_length + SGLANG_CAPTURE_CONTEXT_HEADROOM
    )
    argv = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        cfg.model.target_model_path,
        "--dtype",
        cfg.model.torch_dtype,
    ]
    if cfg.model.trust_remote_code:
        argv.append("--trust-remote-code")
    if cfg.model.cache_dir:
        argv.extend(("--download-dir", cfg.model.cache_dir))
    argv.extend(
        [
            "--tp-size",
            str(args.tp_size),
            "--chunked-prefill-size",
            "-1",
            "--enable-spec-capture",
            "--spec-capture-method",
            contract.method,
            "--spec-capture-aux-layer-ids",
            *[str(layer) for layer in contract.aux_layer_ids],
            "--spec-capture-intake-url",
            intake_url,
            "--spec-capture-sample-rate",
            str(args.sample_rate),
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
    )
    argv.extend(_sglang_argv(cfg.model, overrides={"sglang_context_length": context_length}))
    argv.extend(extra)
    print(f"exec: {' '.join(argv)}", flush=True)
    os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
