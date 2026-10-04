"""Measure an overlapping SGLang -> Mooncake -> online trainer pipeline.

The server and Mooncake master must already be running. A fresh bounded producer
and consumer run concurrently for every arm; features are never precaptured.
The primary interval starts at the first HTTP capture dispatch and ends after
the final optimizer update, synchronous durable acknowledgement and CUDA sync.
Configured periodic checkpoints occur inside the primary interval; the final
checkpoint/cleanup is outside it and included in fit wall time. All checkpoint
time is also reported separately. Warmup runs precede measured ABBA arms.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sqlite3
import statistics
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoConfig, Qwen3Config

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.inference.adapters.server_capture import (
    ServerCaptureSchema,
    SGLangServerCaptureAdapter,
)
from specforge.launch import build_disagg_online_consumer, build_disagg_online_producer
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from specforge.training.checkpoint import STATE_FILE


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-config", type=Path)
    parser.add_argument(
        "--prompts-path",
        type=Path,
        help="Pretokenized JSONL input_ids/loss_mask in the exact desired order",
    )
    parser.add_argument(
        "--algorithm", choices=("dflash", "dflash2", "both"), default="both"
    )
    parser.add_argument(
        "--capture-layers",
        help="Comma-separated target layer IDs; defaults to draft config",
    )
    parser.add_argument(
        "--draft-layers", type=int, default=2, help="Used only without --draft-config"
    )
    parser.add_argument("--lengths", default="128,256,512,2048")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--anchors", type=int, default=512)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        help="Defaults to a full untimed replay of all measured steps",
    )
    parser.add_argument(
        "--repeats", type=int, default=3, help="Number of ABBA blocks (4 runs each)"
    )
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--prompt-fraction", type=float, default=0.25)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--objective-chunk-blocks", type=int, default=128)
    parser.add_argument("--capture-batch-size", type=int)
    parser.add_argument(
        "--backlog",
        type=int,
        default=8,
        help="High watermark refs; one capture batch may overshoot",
    )
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument(
        "--teacher-metrics", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument(
        "--receive-buffers", choices=("pageable", "pinned"), default="pinned"
    )
    parser.add_argument("--segment-mib", type=int, default=1024)
    parser.add_argument("--local-buffer-mib", type=int, default=256)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--dist-port", type=int, default=29712)
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.warmup_steps = args.steps if args.warmup_steps is None else args.warmup_steps
    args.lengths = [int(value) for value in args.lengths.split(",")]
    if args.capture_layers:
        args.capture_layers = [int(value) for value in args.capture_layers.split(",")]
    args.capture_batch_size = args.capture_batch_size or args.batch_size
    positive = (
        args.batch_size,
        args.accumulation_steps,
        args.anchors,
        args.steps,
        args.warmup_steps,
        args.repeats,
        args.capture_batch_size,
    )
    if min(positive) < 1 or not args.lengths or min(args.lengths) < 4:
        parser.error("batch/step/anchor counts must be positive and lengths >= 4")
    if not 0 <= args.prompt_fraction < 1:
        parser.error("prompt-fraction must be in [0,1)")
    quantum = args.batch_size * args.accumulation_steps
    if args.backlog < 2 * quantum:
        parser.error("backlog must be at least twice the optimizer sample quantum")
    if args.log_interval < 1 or args.save_interval < 0:
        parser.error("log-interval must be positive and save-interval nonnegative")
    return args


def _draft_config(args, target_config, architecture):
    if args.draft_config:
        payload = json.loads(args.draft_config.read_text())
    else:
        payload = target_config.to_dict()
        payload.update(
            num_hidden_layers=args.draft_layers,
            num_target_layers=target_config.num_hidden_layers,
            layer_types=["full_attention"] * args.draft_layers,
            block_size=16,
        )
    method = dict(payload.get("dflash_config") or {})
    layers = args.capture_layers or method.get("target_layer_ids")
    if not layers:
        raise ValueError(
            "supply --capture-layers or a draft config with target_layer_ids"
        )
    if min(layers) < 0 or max(layers) >= target_config.num_hidden_layers:
        raise ValueError("capture layers are outside the target model")
    if int(payload["hidden_size"]) != int(target_config.hidden_size) or int(
        payload["vocab_size"]
    ) != int(target_config.vocab_size):
        raise ValueError("draft hidden size and vocabulary must match the target")
    method["target_layer_ids"] = list(layers)
    method.setdefault("mask_token_id", int(target_config.vocab_size) - 1)
    if architecture == "dflash2":
        method.setdefault("conv_group_size", 32)
        method.setdefault("conv_kernel_size", 4)
        method.setdefault("selector_rank", 16)
        method.setdefault("selector_top_k", 16)
    payload["architectures"] = [
        "DFlash2DraftModel" if architecture == "dflash2" else "DFlashDraftModel"
    ]
    payload["dflash_config"] = method
    payload["num_target_layers"] = int(target_config.num_hidden_layers)
    payload["attention_dropout"] = 0.0
    config = Qwen3Config(**payload)
    config._attn_implementation = "flex_attention"
    return config


def _fingerprint(model):
    """Check every parameter's shape and deterministic boundary values cheaply."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(f"{name}:{tuple(value.shape)}:{value.dtype}".encode())
        flat = value.detach().reshape(-1)
        digest.update(
            torch.cat((flat[:64], flat[-64:])).float().cpu().numpy().tobytes()
        )
    return digest.hexdigest()


def _parameter_inventory(model):
    """Inventory unique Parameters before wrapping; component counts may overlap."""
    named = dict(model.named_parameters())

    def counts(parameters):
        unique = {id(parameter): parameter for parameter in parameters}.values()
        unique = list(unique)
        total = sum(parameter.numel() for parameter in unique)
        trainable = sum(
            parameter.numel() for parameter in unique if parameter.requires_grad
        )
        return {
            "tensors": len(unique),
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }

    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    components = {
        "draft": model.draft_model,
        "target_embedding": model.embed_tokens,
        "target_lm_head": model.lm_head,
    }
    return named, {
        "whole_model_unique": counts(named.values()),
        "components": {
            name: counts(module.parameters()) for name, module in components.items()
        },
        "component_count_note": "Each component deduplicates Parameters; tied embedding/head weights appear in both component counts, but only once in whole_model_unique.",
        "draft_layer_count": len(model.draft_model.layers),
        "parameters": [
            {
                "name": name,
                "aliases": aliases[id(parameter)],
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
                "dtype": str(parameter.dtype),
                "trainable": parameter.requires_grad,
            }
            for name, parameter in named.items()
        ],
    }


def _optimizer_coverage(model, named, optimizer):
    """Check original-parameter and FP32-master identities, not just counts."""
    draft_ids = {
        id(parameter)
        for parameter in model.draft_model.parameters()
        if parameter.requires_grad
    }
    target_ids = {
        id(parameter)
        for component in (model.embed_tokens, model.lm_head)
        for parameter in component.parameters()
    }
    trainable_ids = {
        id(parameter) for parameter in named.values() if parameter.requires_grad
    }
    optimizer_ids = [id(parameter) for parameter in optimizer.model_params]
    master_ids = [id(parameter) for parameter in optimizer.fp32_params]
    adam_ids = [
        id(parameter)
        for group in optimizer.optimizer.param_groups
        for parameter in group["params"]
    ]
    if (
        set(optimizer_ids) != draft_ids
        or draft_ids != trainable_ids
        or len(optimizer_ids) != len(draft_ids)
        or target_ids & set(optimizer_ids)
    ):
        raise AssertionError(
            "optimizer must include every trainable draft Parameter exactly once and no target Parameter"
        )
    if (
        len(master_ids) != len(optimizer_ids)
        or len(adam_ids) != len(master_ids)
        or len(set(master_ids)) != len(master_ids)
        or set(adam_ids) != set(master_ids)
        or any(
            parameter.shape != master.shape or master.dtype != torch.float32
            for parameter, master in zip(optimizer.model_params, optimizer.fp32_params)
        )
    ):
        raise AssertionError(
            "AdamW must optimize one matching FP32 master per draft Parameter"
        )
    return {
        "all_trainable_draft_parameters_included": True,
        "target_parameters_excluded": True,
        "one_fp32_adamw_master_per_trainable_parameter": True,
        "optimized_tensors": len(optimizer_ids),
        "optimized_elements": sum(
            parameter.numel() for parameter in optimizer.model_params
        ),
        "parameter_names": [
            name for name, parameter in named.items() if id(parameter) in draft_ids
        ],
    }


def _parameter_sample_indices(numel, *, device="cpu"):
    """Integer interpolation keeps large tensor endpoints exact and in bounds."""
    count = min(128, numel)
    return (
        torch.arange(count, dtype=torch.int64, device=device)
        * max(0, numel - 1)
        // max(1, count - 1)
    )


def _parameter_samples(named):
    """Sample up to 128 evenly spaced values per tensor outside pipeline timing."""
    snapshots = {}
    for name, parameter in named.items():
        flat = parameter.detach().reshape(-1)
        indices = _parameter_sample_indices(flat.numel(), device=flat.device)
        values = flat[indices].float().cpu()
        snapshots[name] = {
            "sha256": hashlib.sha256(values.numpy().tobytes()).hexdigest(),
            "sampled_elements": values.numel(),
            "sampled_values_finite": bool(torch.isfinite(values).all()),
        }
    return snapshots


def _parameter_update_evidence(named, before, layer_count):
    """Full finiteness scan plus sampled change evidence, after the timed fit."""
    after = _parameter_samples(named)
    trainable = [
        (name, parameter)
        for name, parameter in named.items()
        if parameter.requires_grad
    ]
    finite = (
        torch.stack(
            [torch.isfinite(parameter.detach()).all() for _, parameter in trainable]
        )
        .cpu()
        .tolist()
    )
    if not all(finite):
        raise AssertionError("training left nonfinite values in a trainable Parameter")
    changed = {name: before[name]["sha256"] != after[name]["sha256"] for name in named}
    frozen_changed = [
        name
        for name, parameter in named.items()
        if not parameter.requires_grad and changed[name]
    ]
    if frozen_changed:
        raise AssertionError(f"frozen target/draft samples changed: {frozen_changed}")
    layers = []
    for index in range(layer_count):
        prefix = f"draft_model.layers.{index}."
        layer_names = [name for name, _ in trainable if name.startswith(prefix)]
        changed_names = [name for name in layer_names if changed[name]]
        layers.append(
            {
                "layer": index,
                "trainable_tensors": len(layer_names),
                "changed_sampled_tensors": changed_names,
                "sampled_update_observed": bool(changed_names),
            }
        )
    return {
        "timing": "Snapshots before first capture and after fit; full finite scan after fit. No measured-step hooks or synchronizations.",
        "semantics": "All trainable elements are checked finite. Change hashes cover at most 128 evenly spaced elements per tensor; an unchanged hash does not prove an entire tensor was unchanged. Zero initial gradients or sparse selector updates are legitimate.",
        "all_trainable_elements_finite": True,
        "all_frozen_parameter_samples_unchanged": True,
        "all_decoder_layers_have_sampled_updates": all(
            layer["sampled_update_observed"] for layer in layers
        ),
        "layers": layers,
        "parameters": {
            name: {
                "before": before[name],
                "after": after[name],
                "sampled_update_observed": changed[name],
            }
            for name in named
        },
    }


def _observe_first_warmup_step(optimizer, named):
    """Observe presence without reading gradient tensors; never used in timed arms."""
    evidence = {
        "performed": False,
        "scope": "first optimizer step of untimed warmup only",
    }
    original_step = optimizer.step
    names = {id(parameter): name for name, parameter in named.items()}

    def step(**kwargs):
        evidence["gradient_present"] = {
            names[id(parameter)]: parameter.grad is not None
            for parameter in optimizer.model_params
        }
        result = original_step(**kwargs)
        evidence["performed"] = True
        evidence["_gradient_norm"] = optimizer.last_grad_norm.detach().clone()
        optimizer.step = original_step
        return result

    optimizer.step = step
    return evidence


def _optimizer_step_evidence(optimizer, named, expected_steps):
    names = {id(parameter): name for name, parameter in named.items()}
    steps = {}
    for parameter, master in zip(optimizer.model_params, optimizer.fp32_params):
        value = optimizer.optimizer.state.get(master, {}).get("step", 0)
        steps[names[id(parameter)]] = int(
            value.item() if isinstance(value, torch.Tensor) else value
        )
    return {
        "expected_steps": expected_steps,
        "adamw_steps_by_parameter": steps,
        "all_trainable_parameters_received_every_optimizer_step": all(
            value == expected_steps for value in steps.values()
        ),
        "semantics": "AdamW step counters prove optimizer participation, including legitimate zero gradients; they do not imply every element changed.",
    }


def _model(args, target_config, architecture):
    torch.manual_seed(args.seed)
    config = _draft_config(args, target_config, architecture)
    draft_type = DFlash2DraftModel if architecture == "dflash2" else DFlashDraftModel
    draft = draft_type(config).to(dtype=torch.bfloat16)
    fingerprint = _fingerprint(draft)
    components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model,
        device="cuda",
        dtype=torch.bfloat16,
    ).requires_grad_(False)
    model = OnlineDFlashModel(
        draft_model=draft.cuda(),
        target_lm_head=components.lm_head,
        target_embed_tokens=components.embed_tokens,
        mask_token_id=int(config.dflash_config["mask_token_id"]),
        block_size=int(config.block_size),
        attention_backend="flex_attention",
        num_anchors=args.anchors,
        objective_chunk_blocks=args.objective_chunk_blocks,
        teacher_metrics=args.teacher_metrics,
    ).cuda()
    return model, config, fingerprint


def _prompts(args, steps, vocab_size):
    count = steps * args.batch_size * args.accumulation_steps
    generator = torch.Generator().manual_seed(args.seed + 1)
    desired = []
    digest = hashlib.sha256()
    useful_tokens = supervised_tokens = 0
    supplied = None
    if args.prompts_path:
        with args.prompts_path.open() as stream:
            supplied = [json.loads(line) for line in stream if line.strip()]
        if len(supplied) < count:
            raise ValueError(
                f"{args.prompts_path} has {len(supplied)} prompts; this run requires {count}"
            )
    for index in range(count):
        if supplied is None:
            length = args.lengths[index % len(args.lengths)]
            ids = torch.randint(0, vocab_size, (length,), generator=generator).tolist()
            prompt_length = min(int(length * args.prompt_fraction), length - 2)
            mask = [0] * prompt_length + [1] * (length - prompt_length)
        else:
            row = supplied[index].get("payload", supplied[index])
            ids, mask = list(row["input_ids"]), list(row["loss_mask"])
            length = len(ids)
            if (
                len(mask) != length
                or not ids
                or not all(
                    isinstance(token, int) and 0 <= token < vocab_size for token in ids
                )
            ):
                raise ValueError(f"invalid token IDs or mask shape in prompt {index}")
            if not any(a > 0.5 and b > 0.5 for a, b in zip(mask, mask[1:])):
                raise ValueError(
                    f"prompt {index} has no two consecutive supervised tokens"
                )
        payload = {"input_ids": ids, "loss_mask": mask}
        digest.update(json.dumps(payload, separators=(",", ":")).encode())
        desired.append(
            {
                "task_id": f"prompt-{index:08d}",
                "source_id": (
                    "pretokenized-jsonl"
                    if supplied is not None
                    else "synthetic-length-pattern"
                ),
                "payload": payload,
                "max_length": length,
            }
        )
        useful_tokens += length
        supervised_tokens += sum(mask)
    # Compensate for the canonical producer's deterministic shuffle so every
    # warm/measured microbatch sees the same repeating length pattern. Capture
    # order is checked below; an upstream ordering change fails this benchmark.
    order = list(range(count))
    random.Random(args.seed).shuffle(order)
    prompts = [None] * count
    for position, shuffled_index in enumerate(order):
        prompts[shuffled_index] = desired[position]
    return prompts, digest.hexdigest(), useful_tokens, supervised_tokens


def _compiler_counters():
    from torch._dynamo.utils import counters

    return {
        str(namespace): {str(key): int(value) for key, value in counts.items()}
        for namespace, counts in counters.items()
    }


def _counter_delta(before, after):
    return {
        namespace: {
            key: after.get(namespace, {}).get(key, 0)
            - before.get(namespace, {}).get(key, 0)
            for key in set(before.get(namespace, {})) | set(after.get(namespace, {}))
            if after.get(namespace, {}).get(key, 0)
            != before.get(namespace, {}).get(key, 0)
        }
        for namespace in set(before) | set(after)
    }


class _TimedSource:
    def __init__(self, adapter):
        self.adapter = adapter
        self.calls = []
        self.task_ids = []
        self.first_dispatch = None
        self.request_digest = hashlib.sha256()
        original_post = adapter.post_fn

        def timed_post(url, *, json_body, timeout):
            # Hash the actual HTTP inputs and masks after canonical request
            # construction, excluding fresh transport/cache namespaces.
            normalized = {
                key: value
                for key, value in json_body.items()
                if key not in ("extra_key", "spec_capture")
            }
            normalized["spec_capture"] = [
                {
                    **{
                        key: value
                        for key, value in capture.items()
                        if key not in ("store_id", "sample_id")
                    },
                    "sample_id": capture["sample_id"].split(":", 1)[1],
                }
                for capture in json_body["spec_capture"]
            ]
            self.request_digest.update(json.dumps(normalized, sort_keys=True).encode())
            begin = time.perf_counter()
            if self.first_dispatch is None:
                self.first_dispatch = begin
            result = original_post(url, json_body=json_body, timeout=timeout)
            self.calls.append(
                {
                    "start": begin,
                    "end": time.perf_counter(),
                    "samples": len(json_body["spec_capture"]),
                }
            )
            return result

        adapter.post_fn = timed_post

    def produce_refs(self, tasks, *, capture):
        result = self.adapter.produce_refs(tasks, capture=capture)
        self.task_ids.extend(task.task_id for task in tasks)
        return result

    def __getattr__(self, name):
        return getattr(self.adapter, name)


class _TimedChannel(StreamingRefChannel):
    def __init__(self, path):
        super().__init__(path)
        self.publications = []
        self.max_backlog = 0

    def begin_publish(self, refs):
        transaction = super().begin_publish(refs)
        channel = self

        class Transaction:
            def commit(self):
                result = transaction.commit()
                backlog = channel.in_flight_remote()
                channel.max_backlog = max(channel.max_backlog, backlog)
                channel.publications.append(
                    {
                        "time": time.perf_counter(),
                        "samples": len(refs),
                        "backlog": backlog,
                        "task_ids": [ref.source_task_id for ref in refs],
                    }
                )
                return result

            def __getattr__(self, name):
                return getattr(transaction, name)

        return Transaction()


def _store(args, run_id):
    required = ("MOONCAKE_MASTER_SERVER_ADDR", "MOONCAKE_METADATA_SERVER")
    for name in required:
        if not os.environ.get(name):
            raise ValueError(f"set {name} for the existing server's Mooncake master")
    return MooncakeFeatureStore(
        store_id=run_id,
        retain_on_release=True,
        receive_buffers=args.receive_buffers,
        setup_kwargs={
            "local_hostname": os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "127.0.0.1"),
            "metadata_server": os.environ["MOONCAKE_METADATA_SERVER"],
            "master_server_addr": os.environ["MOONCAKE_MASTER_SERVER_ADDR"],
            "global_segment_size": args.segment_mib << 20,
            "local_buffer_size": args.local_buffer_mib << 20,
            "protocol": os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
            "rdma_devices": os.environ.get("MOONCAKE_RDMA_DEVICES", ""),
        },
    )


def run_pipeline(args, target_config, architecture, packing, steps, label):
    run_id = f"{uuid.uuid4().hex[:12]}-{architecture}-{label}-{'packed' if packing else 'padded'}"
    work = args.work_dir / run_id
    work.mkdir(parents=True, exist_ok=False)
    assembled_at = time.perf_counter()
    model, config, fingerprint = _model(args, target_config, architecture)
    named_parameters, parameter_inventory = _parameter_inventory(model)
    before_parameter_samples = _parameter_samples(named_parameters)
    prompts, prompt_hash, useful_tokens, supervised_tokens = _prompts(
        args, steps, config.vocab_size
    )
    algorithm = builtin_algorithm_registry().resolve("dflash")
    provider = algorithm.providers.server_streaming_for("text")
    layout = provider.layout
    schema = ServerCaptureSchema(
        aux_feature=layout.aux_feature,
        last_hidden_feature=(
            layout.last_hidden_feature if args.teacher_metrics else None
        ),
        passthrough=layout.passthrough,
        attention_mask_feature=layout.attention_mask_feature,
    )
    store = _store(args, run_id)
    channel = _TimedChannel(str(work / "refs.jsonl"))
    source = _TimedSource(
        SGLangServerCaptureAdapter(
            args.server_url,
            store,
            run_id=run_id,
            algorithm="dflash",
            schema=schema,
            timeout_s=args.request_timeout,
            target_model_version=args.target_model,
        )
    )
    producer_thread = None
    trainer = None
    stop = threading.Event()
    producer_state = {}
    acknowledgements = []
    checkpoints = []
    logged = []
    loader_counter_windows = []
    fit_started = False
    try:
        trainer = build_disagg_online_consumer(
            algorithm=algorithm,
            feature_store=store,
            channel=channel,
            draft_model=model,
            optimizer_factory=lambda module: BF16Optimizer(
                module,
                lr=args.learning_rate,
                max_grad_norm=0.5,
                warmup_ratio=0.0,
                total_steps=steps,
            ),
            run_id=run_id,
            output_dir=str(work / "output"),
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            max_steps=steps,
            sequence_packing=packing,
            save_interval=args.save_interval,
            log_interval=args.log_interval,
            metadata_db_path=str(work / "consumer.sqlite"),
            async_ack=False,
            idle_timeout_s=args.request_timeout * 2,
            dataloader_num_workers=args.dataloader_workers,
            logger=lambda metrics, step: logged.append(
                {"step": step, "metrics": dict(metrics)}
            ),
        )
        optimizer = trainer.backend.optimizer
        optimizer_coverage = _optimizer_coverage(model, named_parameters, optimizer)
        first_step_evidence = (
            _observe_first_warmup_step(optimizer, named_parameters)
            if label == "warmup"
            else {
                "performed": False,
                "scope": "measured arms have no gradient-observation hook; see matching warmup",
            }
        )
        backend_evidence = {
            "wrapper_kind": trainer.backend._wrapper_kind,
            "configured_sharding_strategy": trainer.backend.parallel_config.sharding_strategy,
            "effective_sharding_strategy": str(
                getattr(trainer.backend.module, "sharding_strategy", "not applicable")
            ),
        }
        original_snapshot = trainer._loader.perf_counters_snapshot

        def tracked_snapshot(reset=False):
            snapshot = original_snapshot(reset=reset)
            if reset:
                loader_counter_windows.append(snapshot)
            return snapshot

        trainer._loader.perf_counters_snapshot = tracked_snapshot
        original_ack = trainer._controller.ack_fn

        def timed_ack(ids, step):
            original_ack(ids, step)
            # Add a CUDA barrier only at the final timing endpoint. Intermediate
            # event timestamps record canonical durable-ack completion.
            if step == steps:
                torch.cuda.synchronize()
            acknowledgements.append(
                {"step": step, "time": time.perf_counter(), "sample_ids": list(ids)}
            )

        trainer._controller.ack_fn = timed_ack
        original_checkpoint = trainer._controller.save_checkpoint

        def timed_checkpoint(step):
            start = time.perf_counter()
            result = original_checkpoint(step)
            torch.cuda.synchronize()
            checkpoints.append({"step": step, "seconds": time.perf_counter() - start})
            return result

        trainer._controller.save_checkpoint = timed_checkpoint
        _, drive = build_disagg_online_producer(
            algorithm=algorithm,
            prompts=prompts,
            feature_store=store,
            channel=channel,
            run_id=run_id,
            target_hidden_size=config.hidden_size,
            target_vocab_size=config.vocab_size,
            target_repr=provider.target_representation,
            aux_hidden_state_layer_ids=config.dflash_config["target_layer_ids"],
            feature_source=source,
            lease=args.capture_batch_size,
            producer_concurrency=1,
            num_rollout_workers=1,
            in_flight_high_watermark=args.backlog,
            in_flight_low_watermark=max(
                args.batch_size * args.accumulation_steps, args.backlog // 2
            ),
            prompt_seed=args.seed,
            prompt_ingest_batch_size=max(64, args.backlog),
            backpressure_poll_s=0.01,
            peer_wait_timeout_s=args.request_timeout * 2,
            max_prompt_attempts=1,
            max_worker_failures=1,
        )

        def produce():
            try:
                producer_state["samples"] = drive(
                    should_stop=lambda: stop.is_set() or channel.consumer_stopped()
                )
            except BaseException as exc:
                producer_state["error"] = repr(exc)
                channel.fail(repr(exc))
            finally:
                producer_state["end"] = time.perf_counter()

        torch.manual_seed(args.seed + 2)
        torch.cuda.synchronize()
        compiler_before = _compiler_counters()
        setup_seconds = time.perf_counter() - assembled_at
        fit_begin = time.perf_counter()
        producer_thread = threading.Thread(
            target=produce, name=f"capture-{run_id}", daemon=True
        )
        producer_thread.start()
        fit_started = True
        final_step = trainer.fit()
        torch.cuda.synchronize()
        fit_end = time.perf_counter()
        producer_thread.join(timeout=args.request_timeout + 10)
        if producer_thread.is_alive():
            raise RuntimeError("producer did not finish after the final optimizer step")
        if "error" in producer_state:
            raise RuntimeError(producer_state["error"])
        expected_count = steps * args.batch_size * args.accumulation_steps
        if producer_state.get("samples") != expected_count:
            raise AssertionError("producer did not publish the complete workload")
        if not channel.consumer_stopped() or channel.consumer_failure() is not None:
            raise AssertionError("consumer did not publish a clean completion")
        expected_order = [f"prompt-{i:08d}" for i in range(expected_count)]
        published_order = [
            task for event in channel.publications for task in event["task_ids"]
        ]
        if source.task_ids != expected_order or published_order != expected_order:
            raise AssertionError(
                "capture/publication ordering changed or capture retried; comparison invalid"
            )
        with sqlite3.connect(work / "consumer.sqlite") as connection:
            acked = connection.execute(
                "SELECT sample_id FROM acked ORDER BY sample_id"
            ).fetchall()
        ack_ids = [
            sample for event in acknowledgements for sample in event["sample_ids"]
        ]
        if (
            final_step != steps
            or len(ack_ids) != expected_count
            or len(acked) != expected_count
        ):
            raise AssertionError(
                "optimizer steps or durable sample acknowledgements do not match the workload"
            )
        if (
            set(ack_ids) != {row[0] for row in acked}
            or len(set(ack_ids)) != expected_count
        ):
            raise AssertionError(
                "sample identities changed or duplicate acknowledgements occurred"
            )
        ack_order = [sample.split(":", 1)[1] for sample in ack_ids]
        if ack_order != expected_order:
            raise AssertionError("consumed sample order or logical batching changed")
        final_checkpoint = work / "output" / f"{run_id}-step{steps}" / STATE_FILE
        if not final_checkpoint.is_file():
            raise AssertionError("canonical final checkpoint is missing")
        state = torch.load(
            final_checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        checkpoint_proof = {
            key: state[key]
            for key in ("global_step", "epoch", "epoch_batch", "epoch_samples")
        }
        del state
        if (
            checkpoint_proof["global_step"] != steps
            or checkpoint_proof["epoch_samples"] != expected_count
        ):
            raise AssertionError(
                "checkpoint step/sample progress disagrees with durable acks"
            )
        loader_counter_windows.append(original_snapshot(reset=False))
        loader_counters = {
            key: sum(window.get(key, 0) for window in loader_counter_windows)
            for key in loader_counter_windows[-1]
        }
        if trainer.micro_step != steps * args.accumulation_steps:
            raise AssertionError("logical microbatch count changed")
        final_loss = trainer._controller._last_result.loss
        final_loss = (
            float(final_loss.detach().cpu())
            if isinstance(final_loss, torch.Tensor)
            else float(final_loss)
        )
        if not math.isfinite(final_loss):
            raise AssertionError("nonfinite final training loss")
        start = source.first_dispatch
        finish = acknowledgements[-1]["time"]
        capture_end = max(call["end"] for call in source.calls)
        if start is None or finish <= start:
            raise AssertionError("invalid pipeline timing boundaries")
        elapsed = finish - start
        # These checks run after the final ack AND the checkpoint/fit endpoint;
        # their tensor reads and finite scans affect neither reported interval.
        parameter_updates = _parameter_update_evidence(
            named_parameters,
            before_parameter_samples,
            parameter_inventory["draft_layer_count"],
        )
        optimizer_steps = _optimizer_step_evidence(optimizer, named_parameters, steps)
        if first_step_evidence["performed"]:
            first_step_evidence["global_gradient_norm"] = float(
                first_step_evidence.pop("_gradient_norm").cpu()
            )
            first_step_evidence["all_trainable_gradients_present"] = all(
                first_step_evidence["gradient_present"].values()
            )
            first_step_evidence["global_gradient_norm_finite"] = math.isfinite(
                first_step_evidence["global_gradient_norm"]
            )
            first_step_evidence["semantics"] = (
                "Presence is observed before BF16Optimizer clears gradients. Its finite global norm check covers all present gradients. Zero gradients are valid at initialization, including DFlash2 bilinear selector factors."
            )
        result = {
            "run_id": run_id,
            "architecture": architecture,
            "packing": packing,
            "label": label,
            "optimizer_steps": steps,
            "microsteps": trainer.micro_step,
            "samples": expected_count,
            "durable_acked_samples": len(acked),
            "useful_tokens": useful_tokens,
            "supervised_tokens": supervised_tokens,
            "pipeline_seconds": elapsed,
            "useful_tokens_per_second": useful_tokens / elapsed,
            "samples_per_second": expected_count / elapsed,
            "capture_finished_seconds": capture_end - start,
            "producer_finished_seconds": producer_state["end"] - start,
            "trainer_fit_seconds_from_first_capture": fit_end - start,
            "fit_wall_seconds": fit_end - fit_begin,
            "model_and_runtime_setup_seconds": setup_seconds,
            "checkpoint_seconds": sum(item["seconds"] for item in checkpoints),
            "checkpoint_events": checkpoints,
            "checkpoint_proof": checkpoint_proof,
            "backend": backend_evidence,
            "parameter_inventory": parameter_inventory,
            "optimizer_coverage": optimizer_coverage,
            "first_warmup_step_gradient_evidence": first_step_evidence,
            "parameter_update_evidence": parameter_updates,
            "optimizer_step_evidence": optimizer_steps,
            "loader_perf_counters": loader_counters,
            "loader_perf_note": "wait_producer_s and wait_fetch_s are consumer blocking; fetch_s overlaps training and is not additive with wall time",
            "final_loss": final_loss,
            "max_observed_backlog_refs": channel.max_backlog,
            "prompt_sha256": prompt_hash,
            "initial_draft_boundary_fingerprint": fingerprint,
            "actual_capture_task_order": source.task_ids,
            "actual_request_sha256": source.request_digest.hexdigest(),
            "actual_consumed_task_order": ack_order,
            "capture_calls": [
                {**event, "start": event["start"] - start, "end": event["end"] - start}
                for event in source.calls
            ],
            "publication_events": [
                {**event, "time": event["time"] - start}
                for event in channel.publications
            ],
            "optimizer_ack_events": [
                {
                    "step": event["step"],
                    "time": event["time"] - start,
                    "sample_count": len(event["sample_ids"]),
                    "task_ids": [
                        sample.split(":", 1)[1] for sample in event["sample_ids"]
                    ],
                }
                for event in acknowledgements
            ],
            "capture_calls_after_first_optimizer": sum(
                call["start"] > acknowledgements[0]["time"] for call in source.calls
            ),
            "sustained_live_overlap_established": any(
                call["start"] > acknowledgements[0]["time"] for call in source.calls
            ),
            "checkpoint_policy": {
                "save_interval": args.save_interval,
                "canonical_final_save": True,
            },
            "resolved_draft_config": config.to_dict(),
            "logged": logged,
            "compiler_counters_before": compiler_before,
            "compiler_counters_after": _compiler_counters(),
            "compiler_counter_delta": _counter_delta(
                compiler_before, _compiler_counters()
            ),
            "full_warmup_replay": args.warmup_steps >= args.steps,
        }
        (work / "result.json").write_text(json.dumps(result, indent=2, default=str))
        return result
    finally:
        stop.set()
        if producer_thread is not None and producer_thread.is_alive():
            channel.mark_consumer_failed("benchmark consumer stopped")
            producer_thread.join(timeout=args.request_timeout + 10)
        if trainer is not None and not fit_started:
            trainer._loader.close()
            if trainer._on_fit_finally is not None:
                trainer._on_fit_finally()
        store.discard_external_attempts(reason="benchmark-finished")
        store.close()
        if not args.keep_checkpoints and (work / "output").exists():
            shutil.rmtree(work / "output")
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()


def _summary(results):
    summary = {}
    for architecture in sorted({row["architecture"] for row in results}):
        arms = {}
        for packing in (False, True):
            rows = [
                row
                for row in results
                if row["architecture"] == architecture and row["packing"] == packing
            ]
            if rows:
                arms["packed" if packing else "padded"] = {
                    "runs": len(rows),
                    "median_pipeline_seconds": statistics.median(
                        row["pipeline_seconds"] for row in rows
                    ),
                    "median_useful_tokens_per_second": statistics.median(
                        row["useful_tokens_per_second"] for row in rows
                    ),
                    "median_fit_seconds_with_checkpoint": statistics.median(
                        row["trainer_fit_seconds_from_first_capture"] for row in rows
                    ),
                }
        if len(arms) == 2:
            arms["pipeline_speedup"] = (
                arms["padded"]["median_pipeline_seconds"]
                / arms["packed"]["median_pipeline_seconds"]
            )
            arms["fit_speedup_with_checkpoint"] = (
                arms["padded"]["median_fit_seconds_with_checkpoint"]
                / arms["packed"]["median_fit_seconds_with_checkpoint"]
            )
        summary[architecture] = arms
    return summary


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("online pipeline benchmark requires a CUDA consumer")
    import requests
    import torch.distributed as dist

    requests.get(args.server_url.rstrip("/") + "/health", timeout=10).raise_for_status()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    own_group = not dist.is_initialized()
    if own_group:
        os.environ.update(
            RANK="0",
            WORLD_SIZE="1",
            LOCAL_RANK="0",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT=str(args.dist_port),
        )
        torch.cuda.set_device(0)
        from specforge.distributed import init_distributed

        init_distributed(timeout=10, tp_size=1)
    target_config = AutoConfig.from_pretrained(args.target_model)
    architectures = (
        ("dflash", "dflash2") if args.algorithm == "both" else (args.algorithm,)
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "target_config": target_config.to_dict(),
        "warmup_runs": [],
        "measured_runs": [],
        "timing_contract": "first capture dispatch -> final synchronous durable ack + CUDA sync, including configured periodic checkpoints before the final step; final checkpoint and cleanup excluded from pipeline_seconds but included in trainer_fit_seconds_from_first_capture; all checkpoint events separately timed",
        "scope": "actual target weights and target embeddings/head; freshly initialized full draft; single-rank canonical trainer with actual wrapper/sharding recorded per run",
        "prompt_source": (
            str(args.prompts_path)
            if args.prompts_path
            else "synthetic deterministic token prompts"
        ),
        "prompt_file_sha256": (
            hashlib.sha256(args.prompts_path.read_bytes()).hexdigest()
            if args.prompts_path
            else None
        ),
        "server_startup": "existing server: startup excluded, not measured by this process",
        "capture_cache_policy": "canonical adapter generates a fresh extra_key per request attempt, forcing full prefill",
        "producer": "canonical drive_producer in parallel thread, one worker/concurrency=1, bounded ref backlog",
        "async_ack": False,
    }

    def save():
        report["summary"] = _summary(report["measured_runs"])
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, default=str))
        temporary.replace(args.output)

    try:
        for architecture in architectures:
            for packing in (False, True):
                row = run_pipeline(
                    args,
                    target_config,
                    architecture,
                    packing,
                    args.warmup_steps,
                    "warmup",
                )
                row["cold_first_use_for_mode"] = True
                report["warmup_runs"].append(row)
                save()
                gc.collect()
                torch.cuda.empty_cache()
            expected = None
            for repeat in range(args.repeats):
                for index, packing in enumerate((False, True, True, False)):
                    label = f"repeat{repeat:02d}-arm{index}"
                    row = run_pipeline(
                        args, target_config, architecture, packing, args.steps, label
                    )
                    identity = (
                        row["prompt_sha256"],
                        row["initial_draft_boundary_fingerprint"],
                        row["actual_capture_task_order"],
                        row["actual_consumed_task_order"],
                        row["actual_request_sha256"],
                    )
                    if expected is None:
                        expected = identity
                    elif identity != expected:
                        raise AssertionError(
                            "A/B prompt order or draft initialization differs"
                        )
                    report["measured_runs"].append(row)
                    save()
                    gc.collect()
                    torch.cuda.empty_cache()
                    print(
                        json.dumps(
                            {
                                key: row[key]
                                for key in (
                                    "architecture",
                                    "packing",
                                    "label",
                                    "pipeline_seconds",
                                    "useful_tokens_per_second",
                                    "checkpoint_seconds",
                                    "capture_calls_after_first_optimizer",
                                )
                            }
                        ),
                        flush=True,
                    )
    finally:
        save()
        if own_group and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
