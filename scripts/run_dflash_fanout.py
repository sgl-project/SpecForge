#!/usr/bin/env python3
# coding=utf-8
"""Launch one online DFlash target producer and one to three trainers."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import shlex
import signal
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence, TextIO
from urllib.parse import urlparse

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from specforge.config.production_fanout import (  # noqa: E402
    FanoutManifest,
    ManifestError,
    VariantConfig,
    load_manifest,
    resolve_executable,
    validate_launch_inputs,
)
from specforge.runtime.gpu_monitor import (  # noqa: E402
    GpuMonitor,
    GpuOwnershipError,
    GpuRoleAssignment,
)
from specforge.runtime.process_supervisor import (  # noqa: E402
    GPUReservationSet,
    LauncherError,
    ProcessSupervisor,
    RoleCommand,
    gpu_busy_reasons,
    wait_for_free_gpus,
)

SPEC_CAPTURE_MAX_SAMPLE_BYTES = 512 << 20
_OWNER_BUDGET_NUMERATOR = 9
_OWNER_BUDGET_DENOMINATOR = 10


def _cuda_runtime_dirs(manifest: FanoutManifest) -> tuple[str, ...]:
    directories: list[str] = []
    for executable in (
        manifest.training.python_executable,
        manifest.server.python_executable,
    ):
        prefix = Path(executable).resolve().parent.parent
        for directory in prefix.glob(
            "lib/python*/site-packages/nvidia/cuda_runtime/lib"
        ):
            if (directory / "libcudart.so.12").is_file():
                directories.append(os.path.realpath(directory))
    return tuple(dict.fromkeys(directories))


def launcher_environment(manifest: FanoutManifest) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("SGLANG_SPEC_CAPTURE_TOKEN", None)
    python_paths = [_REPO_ROOT]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    library_paths = list(_cuda_runtime_dirs(manifest))
    if env.get("LD_LIBRARY_PATH"):
        library_paths.extend(env["LD_LIBRARY_PATH"].split(os.pathsep))
    mooncake = manifest.mooncake
    env.update(
        {
            "MOONCAKE_LOCAL_HOSTNAME": mooncake.local_hostname,
            "MOONCAKE_METADATA_SERVER": mooncake.metadata_server,
            "MOONCAKE_MASTER_SERVER_ADDR": mooncake.master_server_addr,
            "MOONCAKE_PROTOCOL": mooncake.protocol,
            "MOONCAKE_RDMA_DEVICES": mooncake.rdma_devices,
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": str(mooncake.global_segment_size),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "PYTHONUNBUFFERED": "1",
            "PYTHONPATH": os.pathsep.join(python_paths),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if library_paths:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(library_paths))
    return env


def build_role_commands(
    manifest: FanoutManifest,
    *,
    base_env: Optional[Mapping[str, str]] = None,
) -> tuple[RoleCommand, ...]:
    env = dict(base_env) if base_env is not None else launcher_environment(manifest)
    token = secrets.token_urlsafe(32)
    script = os.path.realpath(__file__)
    commands: list[RoleCommand] = []
    if manifest.mooncake.mode == "managed":
        master_env = dict(env)
        master_env["CUDA_VISIBLE_DEVICES"] = ""
        commands.append(
            RoleCommand(
                role="mooncake-master",
                argv=(
                    resolve_executable(
                        manifest.mooncake.master_executable,
                        sibling_of=manifest.training.python_executable,
                    ),
                    "--enable-http-metadata-server=true",
                    f"--metrics_port={manifest.mooncake.metrics_port}",
                ),
                env=master_env,
                log_path=os.path.join(manifest.log_dir, "mooncake-master.log"),
                persistent=True,
            )
        )

    server_url = urlparse(manifest.server.url)
    server_argv = [
        manifest.server.python_executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        manifest.capture.target_model_path,
        "--skip-tokenizer-init",
        "--tp-size",
        "1",
        "--mem-fraction-static",
        str(manifest.server.mem_fraction_static),
        "--chunked-prefill-size",
        "-1",
        "--disable-radix-cache",
        "--enable-spec-capture",
        "--spec-capture-store-id",
        manifest.run_id,
        "--spec-capture-max-sample-bytes",
        str(SPEC_CAPTURE_MAX_SAMPLE_BYTES),
        "--spec-capture-inventory-db",
        manifest.capture_inventory_db_path,
        "--spec-capture-lifecycle-db",
        manifest.lifecycle_db_path,
        "--spec-capture-method",
        "dflash",
        "--spec-capture-aux-layer-ids",
        *(str(layer) for layer in manifest.capture.capture_layer_ids),
        "--host",
        server_url.hostname,
        "--port",
        str(server_url.port),
    ]
    if manifest.server.trust_remote_code:
        server_argv.append("--trust-remote-code")
    server_env = dict(env)
    server_env["CUDA_VISIBLE_DEVICES"] = str(manifest.server.gpu)
    server_env["SGLANG_SPEC_CAPTURE_TOKEN"] = token
    commands.append(
        RoleCommand(
            "target-server",
            tuple(server_argv),
            server_env,
            os.path.join(manifest.log_dir, "target-server.log"),
            persistent=True,
        )
    )

    def role_command(
        role: str,
        env_for_role: Mapping[str, str],
        *extra: str,
    ) -> RoleCommand:
        return RoleCommand(
            role,
            (
                manifest.training.python_executable,
                script,
                "role",
                "--manifest",
                manifest.path,
                "--manifest-sha256",
                manifest.digest,
                "--role",
                role,
                *extra,
            ),
            env_for_role,
            os.path.join(manifest.log_dir, f"{role}.log"),
        )

    producer_env = dict(env)
    producer_env["CUDA_VISIBLE_DEVICES"] = ""
    producer_env["SGLANG_SPEC_CAPTURE_TOKEN"] = token
    commands.append(role_command("producer", producer_env))
    cleanup_env = dict(env)
    cleanup_env["CUDA_VISIBLE_DEVICES"] = ""
    commands.append(role_command("cleanup", cleanup_env))

    for variant in manifest.variants:
        role = f"consumer:{variant.subscription_id}"
        consumer_env = dict(env)
        consumer_env["CUDA_VISIBLE_DEVICES"] = str(variant.gpu)
        consumer_env["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
            manifest.runtime.run_dir, "torchinductor", variant.subscription_id
        )
        argv = (
            manifest.training.python_executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=1",
            "--max-restarts=0",
            script,
            "role",
            "--manifest",
            manifest.path,
            "--manifest-sha256",
            manifest.digest,
            "--role",
            "consumer",
            "--subscription-id",
            variant.subscription_id,
        )
        commands.append(
            RoleCommand(
                role,
                argv,
                consumer_env,
                os.path.join(manifest.log_dir, f"{role.replace(':', '-')}.log"),
            )
        )
    return tuple(commands)


def print_dry_run(commands: Sequence[RoleCommand], output: TextIO) -> None:
    print("validated DFlash fan-out commands (environment omitted):", file=output)
    for command in commands:
        gpu = command.env.get("CUDA_VISIBLE_DEVICES", "<inherited>")
        print(
            f"[{command.role}] gpu={gpu!r} log={command.log_path}\n  "
            f"{shlex.join(command.argv)}",
            file=output,
        )


def _prepare_run_directories(manifest: FanoutManifest) -> None:
    os.makedirs(manifest.runtime.run_dir, mode=0o750, exist_ok=False)
    directories = {
        manifest.log_dir,
        manifest.cache_dir,
        os.path.dirname(manifest.producer_metadata_db_path),
        os.path.join(manifest.runtime.run_dir, "torchinductor"),
    }
    if manifest.runtime.gpu_monitor.enabled:
        directories.add(manifest.metrics_dir)
    for variant in manifest.variants:
        directories.update(
            {
                os.path.dirname(manifest.variant_metadata_db_path(variant)),
                manifest.variant_output_dir(variant),
            }
        )
    for directory in sorted(directories):
        os.makedirs(directory, mode=0o750, exist_ok=True)


def _gpu_monitor_assignments(
    manifest: FanoutManifest,
) -> tuple[GpuRoleAssignment, ...]:
    return (
        GpuRoleAssignment(
            gpu=manifest.server.gpu,
            logical_role="producer",
            process_role="target-server",
        ),
        *(
            GpuRoleAssignment(
                gpu=variant.gpu,
                logical_role=f"consumer:{variant.subscription_id}",
                process_role=f"consumer:{variant.subscription_id}",
            )
            for variant in manifest.variants
        ),
    )


def _owner_max_resident_bytes(manifest: FanoutManifest) -> int:
    return (
        manifest.mooncake.global_segment_size
        * _OWNER_BUDGET_NUMERATOR
        // _OWNER_BUDGET_DENOMINATOR
    )


def _window_max_live_bytes(manifest: FanoutManifest) -> int:
    return manifest.runtime.max_live_bytes or _owner_max_resident_bytes(manifest)


def _fanout_store(manifest: FanoutManifest, *, lifetime_owner: bool):
    from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

    max_resident_bytes = _owner_max_resident_bytes(manifest) if lifetime_owner else None
    max_release_attempts = max(
        3,
        math.ceil(manifest.runtime.finalize_timeout_s / manifest.runtime.gc_poll_s) + 2,
    )
    return MooncakeFeatureStore(
        store_id=manifest.run_id,
        setup_kwargs={
            "local_hostname": manifest.mooncake.local_hostname,
            "metadata_server": manifest.mooncake.metadata_server,
            "master_server_addr": manifest.mooncake.master_server_addr,
            "protocol": manifest.mooncake.protocol,
            "rdma_devices": manifest.mooncake.rdma_devices,
            "global_segment_size": 0,
            "local_buffer_size": manifest.mooncake.client_buffer_size,
        },
        lifetime_owner=lifetime_owner,
        lifecycle_db_path=manifest.lifecycle_db_path,
        max_resident_bytes=max_resident_bytes,
        max_release_attempts=max_release_attempts,
    )


def _draft_contract(manifest: FanoutManifest) -> tuple[int, int, tuple[int, ...]]:
    with open(manifest.capture.draft_config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    dflash = config["dflash_config"]
    return (
        int(config["hidden_size"]),
        int(config.get("block_size", dflash.get("block_size", 16))),
        tuple(int(value) for value in dflash["target_layer_ids"]),
    )


def _windowed_capture_contract(manifest: FanoutManifest):
    from specforge.launch import build_disagg_windowed_capture_contract

    target_hidden_size, _, layer_ids = _draft_contract(manifest)
    return build_disagg_windowed_capture_contract(
        strategy="dflash",
        target_hidden_size=target_hidden_size,
        target_model_version=manifest.capture.target_model_path,
        tokenizer_version=manifest.capture.tokenizer_path,
        target_repr=None,
        aux_hidden_state_layer_ids=layer_ids,
    )


def _iter_training_records(manifest: FanoutManifest) -> Iterator[dict[str, Any]]:
    path = manifest.capture.train_data_path
    if manifest.capture.is_pretokenized:
        required_fields = ("input_ids", "loss_mask")
    elif manifest.capture.is_preformatted:
        required_fields = ("text",)
    else:
        required_fields = ("conversations",)
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in training data {path}:{line_number}: {error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"training data {path}:{line_number} must contain a JSON object"
                )
            missing = [field for field in required_fields if field not in record]
            if missing:
                raise ValueError(
                    f"training data {path}:{line_number} is missing {missing}"
                )

            projected = {field: record[field] for field in required_fields}
            if (
                not manifest.capture.is_preformatted
                and not manifest.capture.is_pretokenized
            ):
                tools = record.get("tools")
                if tools is not None and not isinstance(tools, str):
                    tools = json.dumps(tools)
                projected["tools"] = tools
            yield projected


def _pretokenized_prompt(
    manifest: FanoutManifest,
    record: Mapping[str, Any],
    *,
    index: int,
    minimum_valid_tokens: int,
) -> dict[str, Any] | None:
    input_ids = record["input_ids"]
    loss_mask = record["loss_mask"]
    if not isinstance(input_ids, list) or not isinstance(loss_mask, list):
        raise ValueError(f"pretokenized row {index} must contain list tensors")
    if not input_ids or len(input_ids) != len(loss_mask):
        raise ValueError(
            f"pretokenized row {index} has invalid aligned lengths "
            f"{len(input_ids)} and {len(loss_mask)}"
        )
    if len(input_ids) > manifest.capture.max_length:
        raise ValueError(
            f"pretokenized row {index} length {len(input_ids)} exceeds "
            f"capture.max_length={manifest.capture.max_length}"
        )
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in input_ids
    ):
        raise ValueError(f"pretokenized row {index} input_ids must be integers")
    if any(value not in (0, 1) for value in loss_mask):
        raise ValueError(f"pretokenized row {index} loss_mask must contain only 0/1")
    if sum(loss_mask) < minimum_valid_tokens:
        return None
    return {
        "task_id": f"prompt-{index:08d}",
        "payload": {"input_ids": input_ids, "loss_mask": loss_mask},
        "target_model_version": manifest.capture.target_model_path,
        "metadata": {"tokenizer_version": manifest.capture.tokenizer_path},
    }


def _producer_prompts(manifest: FanoutManifest, tokenizer) -> list[dict[str, Any]]:
    _, default_block_size, _ = _draft_contract(manifest)
    maximum_block_size = max(
        variant.block_size if variant.block_size is not None else default_block_size
        for variant in manifest.variants
    )
    prompts: list[dict[str, Any]] = []
    records = iter(_iter_training_records(manifest))
    if manifest.capture.is_pretokenized:
        for record in records:
            prompt = _pretokenized_prompt(
                manifest,
                record,
                index=len(prompts),
                minimum_valid_tokens=2 * maximum_block_size,
            )
            if prompt is not None:
                prompts.append(prompt)
            if len(prompts) == manifest.capture.max_prompts:
                return prompts
        raise ValueError(
            f"requested {manifest.capture.max_prompts} prompts but pretokenized "
            f"input produced only {len(prompts)} eligible samples"
        )

    from datasets import Dataset
    from specforge.data import build_eagle3_dataset

    while len(prompts) < manifest.capture.max_prompts:
        remaining = manifest.capture.max_prompts - len(prompts)
        source_rows = list(islice(records, remaining))
        if not source_rows:
            break
        dataset = build_eagle3_dataset(
            dataset=Dataset.from_list(source_rows),
            tokenizer=tokenizer,
            chat_template=manifest.capture.chat_template,
            max_length=manifest.capture.max_length,
            num_proc=min(manifest.capture.dataset_num_proc, len(source_rows)),
            is_preformatted=manifest.capture.is_preformatted,
            minimum_valid_tokens=2 * maximum_block_size,
        )
        for row in islice(dataset, remaining):
            input_ids = row["input_ids"][0]
            loss_mask = row["loss_mask"][0]
            attention_mask = row.get("attention_mask")
            length = (
                int(attention_mask[0].sum().item())
                if attention_mask is not None
                else int(input_ids.shape[0])
            )
            index = len(prompts)
            prompts.append(
                {
                    "task_id": f"prompt-{index:08d}",
                    "payload": {
                        "input_ids": input_ids[:length].tolist(),
                        "loss_mask": loss_mask[:length].tolist(),
                    },
                    "target_model_version": manifest.capture.target_model_path,
                    "metadata": {
                        "tokenizer_version": manifest.capture.tokenizer_path,
                    },
                }
            )

    if len(prompts) < manifest.capture.max_prompts:
        raise ValueError(
            f"requested {manifest.capture.max_prompts} prompts but preprocessing "
            f"produced only {len(prompts)} eligible samples before reaching the end "
            f"of {manifest.capture.train_data_path}"
        )
    return prompts


def run_producer(manifest: FanoutManifest) -> None:
    from specforge.inference.adapters.server_capture import SGLangServerCaptureAdapter
    from specforge.launch import build_disagg_online_windowed_producer

    tokenizer = None
    if not manifest.capture.is_pretokenized:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            manifest.capture.tokenizer_path,
            trust_remote_code=manifest.training.trust_remote_code,
        )
    prompts = _producer_prompts(manifest, tokenizer)
    target_hidden_size, _, layer_ids = _draft_contract(manifest)
    owner_store = _fanout_store(manifest, lifetime_owner=True)
    adapter = SGLangServerCaptureAdapter(
        manifest.server.url,
        owner_store,
        run_id=manifest.run_id,
        strategy="dflash",
        target_model_version=manifest.capture.target_model_path,
    )
    runtime = build_disagg_online_windowed_producer(
        prompts=prompts,
        feature_store=owner_store,
        feature_source=adapter,
        run_id=manifest.run_id,
        consumer_ids=tuple(variant.subscription_id for variant in manifest.variants),
        registry_db_path=manifest.window_registry_db_path,
        max_live_refs=manifest.runtime.max_live_refs,
        max_live_bytes=_window_max_live_bytes(manifest),
        capture_reservation_bytes=(manifest.runtime.capture_reserve_bytes_per_sample),
        target_hidden_size=target_hidden_size,
        target_model_version=manifest.capture.target_model_path,
        tokenizer_version=manifest.capture.tokenizer_path,
        strategy="dflash",
        target_repr=None,
        aux_hidden_state_layer_ids=layer_ids,
        capture_batch_size=manifest.runtime.capture_batch_size,
        capture_batch_wait_s=manifest.runtime.capture_batch_wait_s,
        registry_poll_s=manifest.runtime.registry_poll_s,
        max_capture_retries=manifest.runtime.max_capture_retries,
        retry_backoff_s=manifest.runtime.capture_retry_backoff_s,
        consumer_registration_timeout_s=(
            manifest.runtime.consumer_registration_timeout_s
        ),
        consumer_heartbeat_timeout_s=(manifest.runtime.consumer_heartbeat_timeout_s),
    )
    print(
        f"[producer] run_id={manifest.run_id} mode={manifest.runtime.delivery_mode} "
        f"prompts={len(prompts)} "
        f"consumers={[variant.subscription_id for variant in manifest.variants]}",
        flush=True,
    )
    drive_started = time.monotonic()
    try:
        produced = runtime.drive()
        drive_wall_s = time.monotonic() - drive_started
        accounting = runtime.accounting_snapshot()
        accounting["profile"] = {
            "drive_wall_s": drive_wall_s,
            "capture_adapter": adapter.health(),
        }
        print(
            f"[producer] complete produced={produced} "
            f"accounting={json.dumps(accounting, sort_keys=True)}",
            flush=True,
        )
    finally:
        runtime.close()


def _build_consumer_model(manifest: FanoutManifest, variant: VariantConfig):
    import torch
    from transformers import AutoConfig, AutoTokenizer

    from specforge.core.dflash import OnlineDFlashModel
    from specforge.modeling.draft.dflash import DFlashDraftModel
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
    from specforge.ops.dflash_kernels import validate_dflash_draft_kernel_backend
    from specforge.ops.fused_linear_cross_entropy import validate_liger_installation

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "consumer requires exactly one visible GPU, got "
            f"{torch.cuda.device_count()}"
        )
    if manifest.training.linear_cross_entropy_backend == "liger":
        validate_liger_installation()
    validate_dflash_draft_kernel_backend(manifest.training.draft_kernel_backend)
    tokenizer = AutoTokenizer.from_pretrained(
        manifest.capture.tokenizer_path,
        trust_remote_code=manifest.training.trust_remote_code,
    )
    draft_config = AutoConfig.from_pretrained(
        manifest.capture.draft_config_path,
        trust_remote_code=manifest.training.trust_remote_code,
    )
    if variant.block_size is not None:
        draft_config.block_size = variant.block_size
    draft_config._attn_implementation = manifest.training.attention_backend
    if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
        draft_config.dflash_config = {}
    draft_model = DFlashDraftModel(
        draft_config,
        draft_kernel_backend=manifest.training.draft_kernel_backend,
    ).to(device="cuda", dtype=torch.bfloat16)
    mask_token_id = manifest.training.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        raise ValueError(
            "training.mask_token_id is required when the tokenizer has no mask token"
        )
    draft_model.mask_token_id = mask_token_id
    draft_model.config.dflash_config["mask_token_id"] = mask_token_id
    draft_model.config.dflash_config["target_layer_ids"] = list(
        manifest.capture.capture_layer_ids
    )
    target = TargetEmbeddingsAndHead.from_pretrained(
        manifest.capture.target_model_path,
        embed_key=manifest.training.embedding_key,
        lm_head_key=manifest.training.lm_head_key,
        device="cuda",
        trust_remote_code=manifest.training.trust_remote_code,
    )
    return OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target.lm_head,
        target_embed_tokens=target.embed_tokens,
        mask_token_id=mask_token_id,
        block_size=draft_model.block_size,
        attention_backend=manifest.training.attention_backend,
        num_anchors=variant.num_anchors,
        loss_decay_gamma=variant.loss_decay_gamma,
        loss_type=variant.loss_type,
        dpace_alpha=variant.dpace_alpha,
        flex_kernel_options=manifest.training.flex_kernel_options,
        draft_kernel_backend=manifest.training.draft_kernel_backend,
        linear_cross_entropy_backend=(manifest.training.linear_cross_entropy_backend),
        compact_zero_weight_ce_rows=(manifest.training.compact_zero_weight_ce_rows),
    )


def _training_logger(label: str):
    def logger(metrics: Mapping[str, Any], step: int) -> None:
        summary: dict[str, float] = {}
        for key, value in metrics.items():
            if isinstance(value, (list, tuple)):
                values = [float(item) for item in value]
                if values:
                    summary[f"{key}_mean"] = sum(values) / len(values)
                continue
            try:
                summary[key] = float(value)
            except (TypeError, ValueError):
                continue
        for required in ("loss", "grad_norm"):
            if required not in summary or not math.isfinite(summary[required]):
                raise FloatingPointError(
                    f"{label} step {step} has invalid {required}: "
                    f"{summary.get(required)!r}"
                )
        print(
            json.dumps(
                {
                    "role": label,
                    "step": step,
                    "timestamp_s": time.time(),
                    "monotonic_s": time.monotonic(),
                    "metrics": summary,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    return logger


def run_consumer(manifest: FanoutManifest, subscription_id: str) -> None:
    from specforge.launch import build_disagg_online_windowed_consumer
    from specforge.optimizer import BF16Optimizer

    variant = manifest.variant(subscription_id)
    from specforge.runtime.data_plane.windowed_capture import (
        SQLiteWindowedCaptureRegistry,
    )
    from specforge.runtime.data_plane.windowed_capture_runtime import (
        start_windowed_consumer_control,
    )

    registry = SQLiteWindowedCaptureRegistry(
        manifest.window_registry_db_path,
        max_live_refs=manifest.runtime.max_live_refs,
        max_live_bytes=_window_max_live_bytes(manifest),
        capture_reservation_bytes=(manifest.runtime.capture_reserve_bytes_per_sample),
        poll_s=manifest.runtime.registry_poll_s,
    )
    control = None
    try:
        initialized = registry.wait_initialized(
            manifest.runtime.consumer_registration_timeout_s
        )
        _, contract_digest = _windowed_capture_contract(manifest)
        expected = (
            manifest.run_id,
            contract_digest,
            manifest.capture.max_prompts,
        )
        observed = (
            initialized["run_id"],
            initialized["contract_digest"],
            initialized["total_samples"],
        )
        if observed != expected:
            raise RuntimeError(
                "windowed capture registry preflight mismatch: "
                f"expected={expected!r}, observed={observed!r}"
            )
        lookbehind, lookahead, max_prefetch = manifest.window_config(variant)
        control = start_windowed_consumer_control(
            registry,
            subscription_id,
            lookbehind=lookbehind,
            lookahead=lookahead,
            prefetch_depth=max_prefetch,
            max_outstanding=manifest.runtime.max_outstanding_per_consumer,
            heartbeat_interval_s=manifest.runtime.consumer_heartbeat_interval_s,
        )
        model = _build_consumer_model(manifest, variant)
        total_steps = manifest.capture.max_prompts // (
            manifest.training.batch_size * manifest.training.accumulation_steps
        )
        periodic_checkpoint = manifest.training.checkpoint.periodic

        def optimizer_factory(draft_module):
            return BF16Optimizer(
                draft_module,
                lr=variant.learning_rate,
                max_grad_norm=manifest.training.max_grad_norm,
                warmup_ratio=variant.warmup_ratio,
                total_steps=total_steps,
                adamw_backend=(
                    "fused"
                    if manifest.training.gradient_clip_backend == "fused_adamw"
                    else "torch"
                ),
            )

        common = {
            "consumer_id": subscription_id,
            "registry_db_path": manifest.window_registry_db_path,
            "max_live_refs": manifest.runtime.max_live_refs,
            "max_live_bytes": _window_max_live_bytes(manifest),
            "capture_reservation_bytes": (
                manifest.runtime.capture_reserve_bytes_per_sample
            ),
            "contract_digest": contract_digest,
            "total_samples": manifest.capture.max_prompts,
            "feature_store": _fanout_store(manifest, lifetime_owner=False),
            "eagle3_model": model,
            "optimizer_factory": optimizer_factory,
            "run_id": manifest.run_id,
            "output_dir": manifest.variant_output_dir(variant),
            "metadata_db_path": manifest.variant_metadata_db_path(variant),
            "lookbehind": lookbehind,
            "lookahead": lookahead,
            "prefetch_depth": max_prefetch,
            "max_outstanding": manifest.runtime.max_outstanding_per_consumer,
            "strategy": "dflash",
            "batch_size": manifest.training.batch_size,
            "accumulation_steps": manifest.training.accumulation_steps,
            "num_epochs": 1,
            "max_steps": None,
            "total_steps": total_steps,
            "save_interval": (
                periodic_checkpoint.step_interval if periodic_checkpoint else 0
            ),
            "max_checkpoints": (
                periodic_checkpoint.sliding_window_size if periodic_checkpoint else 0
            ),
            "eval_interval": 0,
            "idle_timeout_s": manifest.runtime.idle_timeout_s,
            "resume": False,
            "resume_from": None,
            "logger": _training_logger(f"consumer:{subscription_id}"),
            "log_interval": manifest.training.log_interval,
            "consumer_control": control,
            "loader_prefetch_batches": (manifest.runtime.consumer_prefetch_batches),
        }
        runtime = build_disagg_online_windowed_consumer(**common)
    except BaseException as exc:
        if control is not None:
            try:
                control.fail(exc)
            except BaseException as cleanup_error:
                exc.add_note(
                    f"failed to report consumer initialization failure: "
                    f"{cleanup_error!r}"
                )
            try:
                control.close()
            except BaseException as cleanup_error:
                exc.add_note(f"failed to close consumer control: {cleanup_error!r}")
        try:
            registry.close()
        except BaseException as cleanup_error:
            exc.add_note(f"failed to close capture registry: {cleanup_error!r}")
        raise
    print(
        f"[consumer] subscription_id={subscription_id} "
        f"mode={manifest.runtime.delivery_mode} "
        f"block_size={model.block_size} anchors={variant.num_anchors} "
        f"loss_type={variant.loss_type}",
        flush=True,
    )
    try:
        step = runtime.run()
        if manifest.training.checkpoint.save_epoch_end and (
            periodic_checkpoint is None or step % periodic_checkpoint.step_interval != 0
        ):
            runtime.trainer.save_checkpoint(step)
        print(
            f"[consumer] complete subscription_id={subscription_id} step={step} "
            f"accounting={json.dumps(runtime.accounting_snapshot(), sort_keys=True)}",
            flush=True,
        )
    finally:
        runtime.close()


def run_cleanup(manifest: FanoutManifest) -> None:
    store = _fanout_store(manifest, lifetime_owner=True)
    store.abort_all(reason="launcher-failure-cleanup", force=True)
    deadline = time.monotonic() + manifest.runtime.finalize_timeout_s
    while True:
        store.gc()
        health = store.health()
        if (
            health["resident_samples"] == 0
            and health["release_pending"] == 0
            and health["required_reclaims_pending"] == 0
        ):
            print("[cleanup] owner inventory fully reclaimed", flush=True)
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "owner cleanup did not drain within "
                f"{manifest.runtime.finalize_timeout_s:.1f}s: {health}"
            )
        time.sleep(manifest.runtime.gc_poll_s)


def _manifest_for_role(path: str, expected_digest: str) -> FanoutManifest:
    try:
        digest = bytes.fromhex(expected_digest)
    except ValueError as exc:
        raise ManifestError("--manifest-sha256 must be SHA-256 hex") from exc
    if len(digest) != 32 or len(expected_digest) != 64:
        raise ManifestError("--manifest-sha256 must be SHA-256 hex")
    manifest = load_manifest(path)
    if manifest.digest != expected_digest:
        raise ManifestError(
            "manifest changed after launcher validation: "
            f"expected {expected_digest}, got {manifest.digest}"
        )
    return manifest


def run_role(
    manifest_path: str,
    manifest_sha256: str,
    role: str,
    subscription_id: Optional[str],
) -> None:
    manifest = _manifest_for_role(manifest_path, manifest_sha256)
    if role == "producer":
        if subscription_id is not None:
            raise ManifestError("producer does not accept --subscription-id")
        run_producer(manifest)
        return
    if role == "cleanup":
        if subscription_id is not None:
            raise ManifestError("cleanup does not accept --subscription-id")
        run_cleanup(manifest)
        return
    if subscription_id is None:
        raise ManifestError("consumer requires --subscription-id")
    variant = manifest.variant(subscription_id)
    from accelerate.utils import set_seed

    from specforge.distributed import destroy_distributed, init_distributed

    set_seed(variant.seed)
    init_distributed(tp_size=1)
    try:
        run_consumer(manifest, subscription_id)
    finally:
        destroy_distributed()


def run_launcher(
    manifest_path: str,
    *,
    dry_run: bool,
    wait_for_gpus_enabled: bool,
    output: TextIO = sys.stdout,
) -> int:
    manifest = load_manifest(manifest_path)
    validate_launch_inputs(manifest)
    commands = build_role_commands(manifest)
    if dry_run:
        print_dry_run(commands, output)
        return 0
    command_by_role = {command.role: command for command in commands}
    gpu_ids = [manifest.server.gpu, *(variant.gpu for variant in manifest.variants)]
    reservations = GPUReservationSet(
        gpu_ids, lock_path_pattern=manifest.runtime.gpu_lock_path_pattern
    )

    def pre_start_check(command: RoleCommand) -> None:
        visible = command.env.get("CUDA_VISIBLE_DEVICES", "")
        if not visible:
            return
        if not visible.isdigit():
            raise LauncherError(
                f"role {command.role} must name one physical GPU, got {visible!r}",
                role=command.role,
                log_path=command.log_path,
            )
        reasons = gpu_busy_reasons(
            manifest.runtime.nvidia_smi_executable,
            [int(visible)],
            max_used_memory_mib=manifest.runtime.gpu_max_used_memory_mib,
        )
        if reasons:
            raise LauncherError(
                f"GPU exclusivity changed before {command.role}: {'; '.join(reasons)}",
                role=command.role,
                log_path=command.log_path,
            )
        print(
            f"[gpu-exclusive] role={command.role} physical_gpu={visible}",
            file=output,
            flush=True,
        )

    supervisor = ProcessSupervisor(
        termination_grace_s=manifest.runtime.termination_grace_s,
        kill_grace_s=manifest.runtime.kill_grace_s,
        poll_s=manifest.runtime.process_poll_s,
        cwd=str(Path(__file__).resolve().parents[1]),
        output=output,
        pre_start_check=pre_start_check,
    )
    old_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }

    cleanup_failure: Optional[BaseException] = None
    gpu_monitor: Optional[GpuMonitor] = None
    producer_started = False
    handlers_installed = False
    completed = False
    try:
        reservations.acquire()
        print(
            f"[gpu-reserved] physical_gpus={list(reservations.gpu_ids)}",
            file=output,
            flush=True,
        )
        wait_for_free_gpus(
            manifest.runtime.nvidia_smi_executable,
            gpu_ids,
            max_used_memory_mib=manifest.runtime.gpu_max_used_memory_mib,
            wait=wait_for_gpus_enabled,
            poll_s=manifest.runtime.gpu_poll_s,
            output=output,
        )
        _prepare_run_directories(manifest)
        if manifest.runtime.gpu_monitor.enabled:
            gpu_monitor = GpuMonitor(
                _gpu_monitor_assignments(manifest),
                manifest.gpu_samples_path,
                manifest.gpu_summary_path,
                poll_s=manifest.runtime.gpu_monitor.poll_s,
                max_compute_processes=1,
                strict_process_ownership=(
                    manifest.runtime.gpu_monitor.strict_process_ownership
                ),
                output=output,
            )
            gpu_monitor.start()
        for signum in old_handlers:
            signal.signal(signum, supervisor.request_shutdown)
        handlers_installed = True
        master_child = None
        if manifest.mooncake.mode == "managed":
            master_child = supervisor.start(command_by_role["mooncake-master"])
        supervisor.wait_for_tcp(
            manifest.mooncake.master_server_addr,
            timeout_s=manifest.runtime.master_readiness_timeout_s,
            poll_s=manifest.runtime.master_readiness_poll_s,
            child=master_child,
        )
        metadata_url = urlparse(manifest.mooncake.metadata_server)
        supervisor.wait_for_tcp(
            f"{metadata_url.hostname}:{metadata_url.port}",
            timeout_s=manifest.runtime.master_readiness_timeout_s,
            poll_s=manifest.runtime.master_readiness_poll_s,
            child=master_child,
        )
        print("[ready] mooncake", file=output, flush=True)
        server_child = supervisor.start(command_by_role["target-server"])
        if gpu_monitor is not None:
            gpu_monitor.register_process_group("target-server", server_child.pgid)
        supervisor.wait_for_http(
            manifest.server.url + "/health",
            timeout_s=manifest.server.readiness_timeout_s,
            poll_s=manifest.server.readiness_poll_s,
            child=server_child,
        )
        print("[ready] target-server", file=output, flush=True)
        consumer_roles = [
            f"consumer:{variant.subscription_id}" for variant in manifest.variants
        ]
        for role in consumer_roles:
            child = supervisor.start(command_by_role[role])
            if gpu_monitor is not None:
                gpu_monitor.register_process_group(role, child.pgid)
        supervisor.start(command_by_role["producer"])
        producer_started = True
        if (
            gpu_monitor is not None
            and manifest.runtime.gpu_monitor.strict_process_ownership
        ):

            def check_gpu_ownership() -> None:
                try:
                    gpu_monitor.raise_if_ownership_violated()
                except GpuOwnershipError as exc:
                    raise LauncherError(str(exc), role="gpu-monitor") from exc

            supervisor.monitor(
                ["producer", *consumer_roles], health_check=check_gpu_ownership
            )
        else:
            supervisor.monitor(["producer", *consumer_roles])
        print("[success] all finite roles completed", file=output, flush=True)
        completed = True
    finally:
        if not completed and producer_started:
            writer_roles = [
                role
                for role in (
                    "target-server",
                    "producer",
                    *(
                        f"consumer:{variant.subscription_id}"
                        for variant in manifest.variants
                    ),
                )
                if role in supervisor.children
            ]
            try:
                supervisor.stop_roles(writer_roles)
                # The first signal requests a graceful failure cleanup. Consume it
                # only after writers are down so a second signal can still abort
                # the bounded cleanup wait.
                supervisor.consume_shutdown_request()
                supervisor.start(command_by_role["cleanup"])
                required_alive = (
                    ("mooncake-master",) if manifest.mooncake.mode == "managed" else ()
                )
                supervisor.wait_for_role(
                    "cleanup",
                    timeout_s=(
                        manifest.runtime.finalize_timeout_s
                        + manifest.runtime.termination_grace_s
                        + manifest.runtime.kill_grace_s
                    ),
                    required_alive_roles=required_alive,
                )
            except (LauncherError, OSError, TimeoutError) as exc:
                cleanup_failure = exc
                print(
                    f"[failure] automatic cleanup: {exc}",
                    file=output,
                    flush=True,
                )
        try:
            supervisor.shutdown()
        except LauncherError as exc:
            cleanup_failure = cleanup_failure or exc
        if gpu_monitor is not None:
            try:
                gpu_monitor.stop()
            except Exception as exc:
                print(
                    f"[gpu-monitor] shutdown failed: {type(exc).__name__}: {exc}",
                    file=output,
                    flush=True,
                )
        reservations.close()
        if handlers_installed:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
    if cleanup_failure is not None:
        raise cleanup_failure
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    launch = subparsers.add_parser("launch", help="launch the complete topology")
    launch.add_argument("--manifest", required=True)
    launch.add_argument("--dry-run", action="store_true")
    launch.add_argument(
        "--wait-for-gpus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="wait for every reserved GPU to have no compute process",
    )
    role = subparsers.add_parser("role", help=argparse.SUPPRESS)
    role.add_argument("--manifest", required=True)
    role.add_argument("--manifest-sha256", required=True)
    role.add_argument(
        "--role", required=True, choices=("producer", "consumer", "cleanup")
    )
    role.add_argument("--subscription-id")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "launch":
            return run_launcher(
                args.manifest,
                dry_run=args.dry_run,
                wait_for_gpus_enabled=args.wait_for_gpus,
            )
        run_role(
            args.manifest,
            args.manifest_sha256,
            args.role,
            args.subscription_id,
        )
        return 0
    except (ManifestError, LauncherError) as exc:
        role = getattr(exc, "role", None) or args.command
        log_path = getattr(exc, "log_path", None)
        suffix = f" log={log_path}" if log_path else ""
        print(f"[failure] role={role}: {exc}{suffix}", file=sys.stderr, flush=True)
        return getattr(exc, "returncode", 1)


if __name__ == "__main__":
    raise SystemExit(main())
