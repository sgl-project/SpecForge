"""Disaggregated ONLINE Domino example over patched SGLang server-capture.

Domino reuses the DFlash capture schema (``hidden_states`` + hard labels), but
trains ``OnlineDominoModel`` with the Domino strategy and lambda-base schedule.
The producer is CPU-only and drives one or more patched SGLang servers; the
consumer is the training pool reading refs from the channel and tensors from
Mooncake.
"""

import hashlib
import json
import os
import sys

import torch

if not torch.cuda.is_available():
    # CPU producer: keep yunchang/flashinfer CUDA probes on the clean fallback.
    sys.modules["flashinfer"] = None

from accelerate.utils import set_seed
from train_domino import parse_args
from transformers import AutoConfig, AutoTokenizer

from specforge.core.domino import OnlineDominoModel
from specforge.distributed import destroy_distributed, init_distributed
from specforge.inference.adapters.server_capture import SGLangServerCaptureAdapter
from specforge.launch import build_disagg_online_consumer, build_disagg_online_producer
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from specforge.tracker import create_tracker

RUN_ID = os.environ.get("DISAGG_STORE_ID", "qwen3-8b-domino-disagg")


def _role() -> str:
    role = os.environ.get("DISAGG_ROLE")
    if role:
        return role
    return "producer" if os.environ.get("RCLI_NODE_RANK", "0") == "0" else "consumer"


def _max(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _ref_channel() -> StreamingRefChannel:
    return StreamingRefChannel(os.environ["DISAGG_REF_CHANNEL"])


def _mooncake_store() -> MooncakeFeatureStore:
    return MooncakeFeatureStore(
        store_id=RUN_ID,
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


def _producer_prompts(args, tokenizer):
    from datasets import load_dataset
    from specforge.data import build_eagle3_dataset

    cache_key = hashlib.md5(
        f"{args.train_data_path}-{args.max_length}-{args.chat_template}-"
        f"{args.target_model_path}".encode()
    ).hexdigest()
    dataset = load_dataset("json", data_files=args.train_data_path)["train"]
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
    dataset = dataset.filter(lambda x: x["loss_mask"].sum() >= 2 * args.block_size)

    prompts = []
    for row in dataset:
        input_ids, loss_mask = row["input_ids"][0], row["loss_mask"][0]
        attn = row.get("attention_mask")
        n = int(attn[0].sum().item()) if attn is not None else input_ids.shape[0]
        prompts.append(
            {
                "payload": {
                    "input_ids": input_ids[:n].tolist(),
                    "loss_mask": loss_mask[:n].tolist(),
                }
            }
        )
    return prompts


def _resolve_mask_token(args, tokenizer) -> int:
    if args.mask_token_id is not None:
        return args.mask_token_id
    if tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id
    tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
    return tokenizer.mask_token_id


def _build_draft_model(args) -> DFlashDraftModel:
    draft_config = AutoConfig.from_pretrained(args.draft_config_path)
    dflash_config = getattr(draft_config, "dflash_config", {}) or {}
    if dflash_config.get("projector_type") != "domino":
        raise ValueError(
            "Domino disagg training requires dflash_config.projector_type='domino'."
        )
    required = {"emb_dim", "gru_hidden_dim", "pure_draft_prefix_len", "shift_label"}
    missing = sorted(required - set(dflash_config))
    if missing:
        raise ValueError(f"Domino config missing dflash_config fields: {missing}")

    draft_config._attn_implementation = args.attention_backend
    device = torch.device("cuda", torch.cuda.current_device())
    draft_model = DFlashDraftModel(draft_config).to(
        device=device, dtype=torch.bfloat16
    )
    print(
        "[consumer] draft "
        f"layers={draft_config.num_hidden_layers} "
        f"target_layers={draft_model.target_layer_ids} "
        f"block_size={draft_model.block_size}",
        flush=True,
    )
    return draft_model


def _server_urls() -> list:
    urls = os.environ.get("DISAGG_SERVER_URLS")
    if urls:
        return [u.strip() for u in urls.split(",") if u.strip()]
    return [os.environ["DISAGG_SERVER_URL"]]


def run_producer(args) -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    prompts = _producer_prompts(args, tokenizer)
    max_prompts = _max("DISAGG_MAX_PROMPTS", 0)
    if max_prompts:
        prompts = prompts[:max_prompts]

    with open(args.draft_config_path) as f:
        draft_cfg = json.load(f)
    aux_layer_ids = tuple(draft_cfg["dflash_config"]["target_layer_ids"])
    target_hidden = int(draft_cfg["hidden_size"])

    store = _mooncake_store()
    channel = _ref_channel()
    urls = _server_urls()
    adapters = [
        SGLangServerCaptureAdapter(
            url,
            store,
            run_id=RUN_ID,
            strategy="domino",
            target_model_version=args.target_model_path,
        )
        for url in urls
    ]
    print(f"[producer] {len(prompts)} prompts -> {len(urls)} server(s): {urls}")

    _workers, drive_producer = build_disagg_online_producer(
        strategy="domino",
        feature_source=adapters if len(adapters) > 1 else adapters[0],
        prompts=prompts,
        feature_store=store,
        channel=channel,
        run_id=RUN_ID,
        target_hidden_size=target_hidden,
        target_repr=None,
        aux_hidden_state_layer_ids=aux_layer_ids,
    )
    produced = drive_producer()
    print(f"[producer] streamed {produced} samples; channel closed", flush=True)


def run_consumer(args) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    draft_model = _build_draft_model(args)
    mask_token_id = _resolve_mask_token(args, tokenizer)
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
    domino_model = OnlineDominoModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        mask_token_id=mask_token_id,
        block_size=draft_model.block_size,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        shift_label=draft_model.shift_label,
    )

    max_steps = _max("DISAGG_MAX_STEPS", 0) or None
    total_steps = _max("DISAGG_TOTAL_STEPS", 0) or max_steps or 10_000

    def optimizer_factory(draft_module):
        return BF16Optimizer(
            draft_module,
            lr=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            total_steps=total_steps,
        )

    tracker = create_tracker(args, args.output_dir)

    def logger(metrics, step):
        logdict = {}
        for key, value in metrics.items():
            try:
                logdict[f"train/{key}"] = float(value)
            except (TypeError, ValueError):
                continue
        if logdict:
            tracker.log(logdict, step=step)
            summary = {k.split("/")[-1]: round(v, 4) for k, v in logdict.items()}
            print(f"[consumer] step {step} {summary}", flush=True)

    print(f"[consumer] training from mooncake://{RUN_ID}", flush=True)
    trainer, loader = build_disagg_online_consumer(
        strategy="domino",
        feature_store=_mooncake_store(),
        channel=_ref_channel(),
        eagle3_model=domino_model,  # legacy param name = composite draft model
        optimizer_factory=optimizer_factory,
        run_id=RUN_ID,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        num_epochs=args.num_epochs,
        max_steps=max_steps,
        total_steps=total_steps,
        save_interval=args.save_interval,
        tp_size=args.tp_size,
        metadata_db_path=os.environ.get("DISAGG_DB") or None,
        logger=logger,
        log_interval=_max("DISAGG_LOG_INTERVAL", 1),
        strategy_kwargs={
            "lambda_start": args.lambda_base_start,
            "decay_ratio": args.lambda_base_decay_ratio,
        },
        inbox_dir=os.environ.get("DISAGG_INBOX_DIR") or None,
        idle_timeout_s=float(os.environ.get("DISAGG_IDLE_TIMEOUT", 0)) or None,
    )
    try:
        trainer.fit(loader)
    finally:
        if getattr(trainer, "ref_distributor", None) is not None:
            trainer.ref_distributor.stop()
    print(f"[consumer] DONE ({RUN_ID})", flush=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    role = _role()
    print(f"[disagg-domino] role={role} run_id={RUN_ID}", flush=True)
    if role == "producer":
        run_producer(args)
        return
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    try:
        run_consumer(args)
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
