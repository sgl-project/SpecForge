"""Shared helpers for DFlash-family disaggregated online examples.

DFlash and Domino share the same server-capture feature schema:
``input_ids`` + ``loss_mask`` + captured ``hidden_states``.  This module keeps the
common producer/consumer plumbing in one place while entry points still own the
algorithm-specific composite model and strategy kwargs.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from transformers import AutoConfig, AutoTokenizer

from specforge.inference.adapters.server_capture import SGLangServerCaptureAdapter
from specforge.launch import build_disagg_online_consumer, build_disagg_online_producer
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from specforge.tracker import create_tracker
from specforge.utils import get_local_device, print_on_rank0


@dataclass(frozen=True)
class CaptureConfig:
    aux_layer_ids: Tuple[int, ...]
    target_hidden_size: int


@dataclass(frozen=True)
class ConsumerParts:
    draft_model: DFlashDraftModel
    mask_token_id: int
    target_components: TargetEmbeddingsAndHead


def log_event(component: str, message: str) -> None:
    print(f"[{component}] {time.strftime('%Y-%m-%d %H:%M:%S')} {message}", flush=True)


def elapsed(start: float) -> str:
    return f"{time.perf_counter() - start:.3f}s"


def safe_len(value) -> str:
    try:
        return str(len(value))
    except TypeError:
        return "unknown"


def disagg_role() -> str:
    role = os.environ.get("DISAGG_ROLE")
    if role:
        return role
    return "producer" if os.environ.get("RCLI_NODE_RANK", "0") == "0" else "consumer"


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def ref_channel() -> StreamingRefChannel:
    return StreamingRefChannel(os.environ["DISAGG_REF_CHANNEL"])


def mooncake_store(run_id: str) -> MooncakeFeatureStore:
    return MooncakeFeatureStore(
        store_id=run_id,
        setup_kwargs={
            "local_hostname": os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "127.0.0.1"),
            "metadata_server": os.environ["MOONCAKE_METADATA_SERVER"],
            "master_server_addr": os.environ["MOONCAKE_MASTER_SERVER_ADDR"],
            "protocol": os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
            "rdma_devices": os.environ.get("MOONCAKE_RDMA_DEVICES", ""),
            "global_segment_size": int(os.environ.get("DISAGG_CLIENT_SEGMENT_SIZE", 0)),
            "local_buffer_size": int(
                os.environ.get("DISAGG_CLIENT_BUFFER_SIZE", 1 << 30)
            ),
        },
    )


def server_urls() -> list:
    urls = os.environ.get("DISAGG_SERVER_URLS")
    if urls:
        return [u.strip() for u in urls.split(",") if u.strip()]
    return [os.environ["DISAGG_SERVER_URL"]]


def load_capture_config(args) -> CaptureConfig:
    with open(args.draft_config_path) as f:
        draft_cfg = json.load(f)
    return CaptureConfig(
        aux_layer_ids=tuple(draft_cfg["dflash_config"]["target_layer_ids"]),
        target_hidden_size=int(draft_cfg["hidden_size"]),
    )


def build_producer_prompts(args, tokenizer, *, max_prompts: int = 0):
    from datasets import load_dataset
    from specforge.data import build_eagle3_dataset

    total_start = time.perf_counter()
    phase = time.perf_counter()
    log_event("producer-timing", f"load_dataset start path={args.train_data_path}")
    dataset = load_dataset("json", data_files=args.train_data_path)["train"]
    log_event(
        "producer-timing",
        f"load_dataset done rows={safe_len(dataset)} elapsed={elapsed(phase)}",
    )

    cache_scope = "all"
    if max_prompts > 0 and max_prompts < len(dataset):
        phase = time.perf_counter()
        log_event("producer-timing", f"select start max_prompts={max_prompts}")
        dataset = dataset.select(range(max_prompts))
        cache_scope = f"first-{max_prompts}"
        log_event(
            "producer-timing",
            f"select done rows={safe_len(dataset)} elapsed={elapsed(phase)}",
        )

    cache_key_parts = (
        f"{args.train_data_path}-{args.max_length}-{args.chat_template}-"
        f"{args.target_model_path}"
    )
    if cache_scope != "all":
        cache_key_parts = f"{cache_key_parts}-{cache_scope}"
    cache_key = hashlib.md5(cache_key_parts.encode()).hexdigest()

    phase = time.perf_counter()
    log_event(
        "producer-timing",
        "build_eagle3_dataset start "
        f"rows={safe_len(dataset)} cache_key={cache_key} "
        f"num_proc={args.build_dataset_num_proc}",
    )
    dataset = build_eagle3_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=os.path.join(args.cache_dir, "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.build_dataset_num_proc,
    )
    log_event(
        "producer-timing",
        f"build_eagle3_dataset done rows={safe_len(dataset)} elapsed={elapsed(phase)}",
    )

    phase = time.perf_counter()
    log_event(
        "producer-timing",
        f"filter start rows={safe_len(dataset)} block_size={args.block_size}",
    )
    dataset = dataset.filter(lambda x: x["loss_mask"].sum() >= 2 * args.block_size)
    log_event(
        "producer-timing",
        f"filter done rows={safe_len(dataset)} elapsed={elapsed(phase)}",
    )

    prompts = []
    total_rows = safe_len(dataset)
    log_every = max(0, env_int("DISAGG_PROMPT_LOG_EVERY", 10000))
    phase = time.perf_counter()
    last_log = phase
    total_tokens = 0
    log_event(
        "producer-timing",
        f"materialize prompts start rows={total_rows} log_every={log_every}",
    )
    for idx, row in enumerate(dataset, start=1):
        input_ids, loss_mask = row["input_ids"][0], row["loss_mask"][0]
        attn = row.get("attention_mask")
        n = int(attn[0].sum().item()) if attn is not None else input_ids.shape[0]
        total_tokens += int(n)
        prompts.append(
            {
                "payload": {
                    "input_ids": input_ids[:n].tolist(),
                    "loss_mask": loss_mask[:n].tolist(),
                }
            }
        )
        if log_every and (idx == 1 or idx % log_every == 0):
            now = time.perf_counter()
            log_event(
                "producer-timing",
                "materialize prompts progress "
                f"{idx}/{total_rows} total_tokens={total_tokens} "
                f"elapsed={now - phase:.3f}s interval={now - last_log:.3f}s",
            )
            last_log = now

    log_event(
        "producer-timing",
        "materialize prompts done "
        f"prompts={len(prompts)} total_tokens={total_tokens} "
        f"elapsed={elapsed(phase)} total_elapsed={elapsed(total_start)}",
    )
    return prompts


def run_server_capture_producer(
    strategy: str,
    args,
    run_id: str,
    *,
    target_repr=None,
) -> None:
    total_start = time.perf_counter()
    log_event("producer-timing", f"run_producer start run_id={run_id}")

    phase = time.perf_counter()
    log_event("producer-timing", f"tokenizer load start path={args.target_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    log_event("producer-timing", f"tokenizer load done elapsed={elapsed(phase)}")

    max_prompts = env_int("DISAGG_MAX_PROMPTS", 0)
    phase = time.perf_counter()
    prompts = build_producer_prompts(args, tokenizer, max_prompts=max_prompts)
    if max_prompts > 0 and len(prompts) > max_prompts:
        log_event(
            "producer-timing",
            f"final prompt slice start prompts={len(prompts)} max_prompts={max_prompts}",
        )
        prompts = prompts[:max_prompts]
    log_event(
        "producer-timing",
        f"build_producer_prompts returned prompts={len(prompts)} elapsed={elapsed(phase)}",
    )

    phase = time.perf_counter()
    log_event(
        "producer-timing", f"draft config load start path={args.draft_config_path}"
    )
    capture_cfg = load_capture_config(args)
    log_event("producer-timing", f"draft config load done elapsed={elapsed(phase)}")

    phase = time.perf_counter()
    log_event("producer-timing", "mooncake store init start")
    store = mooncake_store(run_id)
    log_event("producer-timing", f"mooncake store init done elapsed={elapsed(phase)}")

    phase = time.perf_counter()
    log_event("producer-timing", "ref channel init start")
    channel = ref_channel()
    log_event("producer-timing", f"ref channel init done elapsed={elapsed(phase)}")

    urls = server_urls()
    phase = time.perf_counter()
    log_event("producer-timing", f"adapter init start urls={urls}")
    adapters = [
        SGLangServerCaptureAdapter(
            url,
            store,
            run_id=run_id,
            strategy=strategy,
            target_model_version=args.target_model_path,
        )
        for url in urls
    ]
    log_event("producer-timing", f"adapter init done elapsed={elapsed(phase)}")
    print(f"[producer] {len(prompts)} prompts -> {len(urls)} server(s): {urls}")

    phase = time.perf_counter()
    log_event("producer-timing", "build_disagg_online_producer start")
    prompt_epochs = max(1, int(args.num_epochs))
    _workers, drive_producer = build_disagg_online_producer(
        strategy=strategy,
        feature_source=adapters if len(adapters) > 1 else adapters[0],
        prompts=prompts,
        feature_store=store,
        channel=channel,
        run_id=run_id,
        target_hidden_size=capture_cfg.target_hidden_size,
        target_repr=target_repr,
        aux_hidden_state_layer_ids=capture_cfg.aux_layer_ids,
        prompt_epochs=prompt_epochs,
    )
    log_event(
        "producer-timing",
        f"build_disagg_online_producer done elapsed={elapsed(phase)}",
    )

    phase = time.perf_counter()
    log_event("producer-timing", "drive_producer start")
    produced = drive_producer()
    log_event("producer-timing", f"drive_producer done elapsed={elapsed(phase)}")
    print(f"[producer] streamed {produced} samples; channel closed", flush=True)
    log_event(
        "producer-timing", f"run_producer done total_elapsed={elapsed(total_start)}"
    )


def resolve_mask_token(args, tokenizer) -> int:
    if args.mask_token_id is not None:
        return args.mask_token_id
    if tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id
    tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
    return tokenizer.mask_token_id


def build_dflash_family_draft(
    args,
    *,
    expected_projector_type: Optional[str] = None,
    required_dflash_fields: Iterable[str] = (),
) -> DFlashDraftModel:
    device = get_local_device()
    if args.draft_config_path:
        draft_config = AutoConfig.from_pretrained(args.draft_config_path)
        print_on_rank0(f"Loaded draft config from {args.draft_config_path}")
        if (
            hasattr(draft_config, "block_size")
            and draft_config.block_size != args.block_size
        ):
            print_on_rank0(
                f"Warning: checkpoint block_size ({draft_config.block_size}) differs from "
                f"command-line arg ({args.block_size}). Using checkpoint value."
            )
    else:
        target_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config = AutoConfig.from_pretrained(args.target_model_path)
        draft_config.num_hidden_layers = args.num_draft_layers
        draft_config.block_size = args.block_size
        draft_config.num_target_layers = target_config.num_hidden_layers
        print_on_rank0("Auto-generated draft config from target model config")

    if not hasattr(draft_config, "dflash_config") or draft_config.dflash_config is None:
        draft_config.dflash_config = {}

    dflash_config = draft_config.dflash_config
    if expected_projector_type is not None:
        projector_type = dflash_config.get("projector_type", None)
        if projector_type != expected_projector_type:
            raise ValueError(
                f"Expected dflash_config.projector_type={expected_projector_type!r}, "
                f"got {projector_type!r}."
            )
    missing = sorted(set(required_dflash_fields) - set(dflash_config))
    if missing:
        raise ValueError(f"Draft config missing dflash_config fields: {missing}")

    draft_config._attn_implementation = args.attention_backend
    print_on_rank0(f"Using attention backend: {args.attention_backend}")

    draft_model = DFlashDraftModel(draft_config).to(device=device, dtype=torch.bfloat16)
    print_on_rank0(
        f"Draft config: block_size={draft_config.block_size}, "
        f"num_hidden_layers={draft_config.num_hidden_layers}, "
        f"num_target_layers={draft_config.num_target_layers}"
    )
    print_on_rank0(
        f"Draft model parameters: {sum(p.numel() for p in draft_model.parameters()):,}"
    )
    return draft_model


def build_consumer_parts(
    args,
    *,
    expected_projector_type: Optional[str] = None,
    required_dflash_fields: Iterable[str] = (),
) -> ConsumerParts:
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    draft_model = build_dflash_family_draft(
        args,
        expected_projector_type=expected_projector_type,
        required_dflash_fields=required_dflash_fields,
    )
    mask_token_id = resolve_mask_token(args, tokenizer)
    draft_model.mask_token_id = mask_token_id
    draft_model.config.dflash_config["mask_token_id"] = mask_token_id
    draft_model.config.dflash_config["target_layer_ids"] = draft_model.target_layer_ids

    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )
    return ConsumerParts(
        draft_model=draft_model,
        mask_token_id=mask_token_id,
        target_components=target_components,
    )


def _optimizer_factory(args, total_steps: int, holder: Optional[Dict] = None):
    def factory(draft_module):
        opt = BF16Optimizer(
            draft_module,
            lr=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            total_steps=total_steps,
        )
        if holder is not None:
            holder["opt"] = opt
        return opt

    return factory


def _make_logger(args, holder: Optional[Dict] = None):
    tracker = create_tracker(args, args.output_dir)

    def logger(metrics, step):
        logdict = {}
        for key, value in metrics.items():
            # Log accuracy under `train/accuracy` to match stock train_dflash.py.
            k = "accuracy" if key == "acc" else key
            try:
                logdict[f"train/{k}"] = float(value)
            except (TypeError, ValueError):
                try:
                    seq = [float(x) for x in value]
                except (TypeError, ValueError):
                    continue
                if seq:
                    logdict[f"train/{k}_mean"] = sum(seq) / len(seq)
        # Log the current learning rate (like stock train_dflash.py -> train/lr).
        opt = (holder or {}).get("opt")
        if opt is not None:
            try:
                logdict["train/lr"] = float(opt.get_learning_rate())
            except Exception:
                pass
        if logdict:
            tracker.log(logdict, step=step)
            summary = {k.split("/")[-1]: round(v, 4) for k, v in logdict.items()}
            print(f"[consumer] step {step} {summary}", flush=True)

    return logger


def fit_online_consumer(
    strategy: str,
    args,
    run_id: str,
    composite_model,
    *,
    strategy_kwargs: Optional[Dict] = None,
) -> None:
    max_steps = env_int("DISAGG_MAX_STEPS", 0) or None
    total_steps = env_int("DISAGG_TOTAL_STEPS", 0) or max_steps or 10_000

    print(
        f"[consumer] training from mooncake://{run_id} "
        f"(producer-looped epochs={args.num_epochs})",
        flush=True,
    )
    # Shared holder so the logger can read the live optimizer's learning rate.
    opt_holder: Dict = {}
    trainer, loader = build_disagg_online_consumer(
        strategy=strategy,
        feature_store=mooncake_store(run_id),
        channel=ref_channel(),
        eagle3_model=composite_model,  # legacy param name = composite draft model
        optimizer_factory=_optimizer_factory(args, total_steps, opt_holder),
        run_id=run_id,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        num_epochs=1,
        max_steps=max_steps,
        total_steps=total_steps,
        save_interval=args.save_interval,
        tp_size=args.tp_size,
        metadata_db_path=os.environ.get("DISAGG_DB") or None,
        logger=_make_logger(args, opt_holder),
        log_interval=env_int("DISAGG_LOG_INTERVAL", 1),
        strategy_kwargs=strategy_kwargs,
        inbox_dir=os.environ.get("DISAGG_INBOX_DIR") or None,
        idle_timeout_s=float(os.environ.get("DISAGG_IDLE_TIMEOUT", 0)) or None,
    )
    try:
        trainer.fit(loader)
    finally:
        if getattr(trainer, "ref_distributor", None) is not None:
            trainer.ref_distributor.stop()
    print(f"[consumer] DONE ({run_id})", flush=True)
