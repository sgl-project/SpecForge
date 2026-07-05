"""Disaggregated ONLINE DFlash example: rollout/inference pool and trainer pool
as separate processes that share only a filesystem mount.

Online analogue of ``run_disagg_eagle3.py`` (which wires the *offline* seam). Here
the target model is never precomputed to disk — the **producer** runs it live and
streams captured features to the **consumer**, which trains the DFlash draft:

* **producer** (rollout / inference pool): builds the DFlash target engine and
  drives ``build_disagg_online_producer`` — RolloutWorkers ``put()`` consume-once
  features into a ``SharedDirFeatureStore`` and stream tensor-free ``SampleRef``s
  through a ``StreamingRefChannel``. No draft, no optimizer.
* **consumer** (training pool): builds the DFlash draft (``OnlineDFlashModel``) and
  trains via ``build_disagg_online_consumer``, reading refs off the channel and
  features from the shared store. The target model is never loaded here.

The two pools share a run-scoped SQLite metadata store (``DISAGG_DB``) so commits
and consume-once acks land together, exactly as in a real two-node split.

This mirrors ``scripts/train_dflash.py``'s model assembly (``build_models``, mask
token, ``TargetEmbeddingsAndHead``, ``OnlineDFlashModel``) so the disaggregated
curve is comparable to the colocated ``examples/run_qwen3.6_27b_dflash_online.sh``
run. It runs **staged** (producer drains, then consumer trains): the interleaved
single-process runner forwards the target in a thread, and DFlash's HF capture
(``output_hidden_states=True``) trips transformers' non-thread-safe hook wrapper.

Config comes from the environment so one wrapper can drive both pools; role is
``DISAGG_ROLE`` (``producer``/``consumer``), or derived from ``RCLI_NODE_RANK``:

    DISAGG_ROLE=producer|consumer      # else rank 0 -> producer, else consumer
    DISAGG_STORE_ROOT=/root/disagg36   # shared *data* mount (both pools)
    DISAGG_STORE_ID=qwen36-dflash-disagg   # producer/consumer must match
    DISAGG_DB=/root/disagg36/run.db    # shared durable metadata store
    DISAGG_MAX_PROMPTS=300             # cap the prompt pool (0 = all)
    DISAGG_MAX_STEPS=150               # trainer max optimizer steps (0 = all)
    DISAGG_LOG_INTERVAL=1              # per-step metric logging cadence

W&B logging on the consumer is driven by ``train_dflash``'s tracker args
(``--report-to wandb --wandb-project ... --wandb-name ...``); export
``WANDB_API_KEY`` in the environment (never hard-code it).
"""

import os

from accelerate.utils import set_seed
from train_dflash import build_dataloader, build_models, parse_args
from transformers import AutoTokenizer

from specforge.core.dflash import OnlineDFlashModel
from specforge.distributed import destroy_distributed, init_distributed
from specforge.launch import build_disagg_online_consumer, build_disagg_online_producer
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.runtime.data_plane.disaggregated import SharedDirFeatureStore
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from specforge.tracker import create_tracker

RUN_ID = os.environ.get("DISAGG_STORE_ID", "qwen36-dflash-disagg")


def _role() -> str:
    role = os.environ.get("DISAGG_ROLE")
    if role:
        return role
    return "producer" if os.environ.get("RCLI_NODE_RANK", "0") == "0" else "consumer"


def _store_root() -> str:
    return os.path.join(os.environ["DISAGG_STORE_ROOT"], RUN_ID)


def _max(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _resolve_mask_token(args, tokenizer) -> int:
    if args.mask_token_id is not None:
        return args.mask_token_id
    if tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id
    tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
    return tokenizer.mask_token_id


def _extract_prompts(train_dataloader):
    """Flatten the training dataloader into rollout prompts (input_ids + mask)."""
    prompts = []
    for batch in train_dataloader:
        input_ids = batch["input_ids"]
        loss_mask = batch["loss_mask"]
        attn = batch.get("attention_mask")
        for i in range(input_ids.shape[0]):
            n = int(attn[i].sum().item()) if attn is not None else input_ids.shape[1]
            prompts.append(
                {
                    "payload": {
                        "input_ids": input_ids[i, :n].tolist(),
                        "loss_mask": loss_mask[i, :n].tolist(),
                    }
                }
            )
    return prompts


def _configure_draft(args, draft_model, tokenizer) -> int:
    """Pin the mask token on the draft config (matches train_dflash.main)."""
    mask_token_id = _resolve_mask_token(args, tokenizer)
    draft_model.mask_token_id = mask_token_id
    draft_model.config.dflash_config["mask_token_id"] = mask_token_id
    draft_model.config.dflash_config["target_layer_ids"] = draft_model.target_layer_ids
    return mask_token_id


def run_producer(args) -> None:
    """Inference pool: run the target live, stream captured features out."""
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    target_model, draft_model = build_models(args)  # sets capture layers from draft
    _configure_draft(args, draft_model, tokenizer)

    train_dataloader, _eval = build_dataloader(args, tokenizer)
    prompts = _extract_prompts(train_dataloader)
    max_prompts = _max("DISAGG_MAX_PROMPTS", 0)
    if max_prompts:
        prompts = prompts[:max_prompts]

    store = SharedDirFeatureStore(_store_root(), store_id=RUN_ID)
    channel = StreamingRefChannel(os.path.join(_store_root(), "refs.jsonl"))
    print(f"[producer] {len(prompts)} prompts -> {_store_root()}", flush=True)

    _workers, drive_producer = build_disagg_online_producer(
        strategy="dflash",
        target_model=target_model,
        prompts=prompts,
        feature_store=store,
        channel=channel,
        run_id=RUN_ID,
        target_hidden_size=int(draft_model.config.hidden_size),
        target_repr=None,  # DFlash trains on captured hidden states, no target dist.
        metadata_db_path=os.environ.get("DISAGG_DB") or None,
    )
    produced = drive_producer()  # generates, streams, closes the channel (EOF)
    print(f"[producer] streamed {produced} samples; channel closed", flush=True)


def run_consumer(args) -> None:
    """Training pool: train the DFlash draft off the shared store + ref channel."""
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    _target_unused, draft_model = build_models(args)
    mask_token_id = _configure_draft(args, draft_model, tokenizer)

    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device="cuda",
        trust_remote_code=args.trust_remote_code,
    )
    dflash_model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        mask_token_id=mask_token_id,
        block_size=draft_model.block_size,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        loss_type=args.loss_type,
        dpace_alpha=args.dpace_alpha,
    )

    max_steps = _max("DISAGG_MAX_STEPS", 0) or None

    def optimizer_factory(draft_module):
        return BF16Optimizer(
            draft_module,
            lr=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            total_steps=max_steps or 10_000,
        )

    # Route the trainer's per-step metrics to the configured tracker (W&B).
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

    store = SharedDirFeatureStore(_store_root(), store_id=RUN_ID)
    channel = StreamingRefChannel(os.path.join(_store_root(), "refs.jsonl"))
    print(f"[consumer] training from {_store_root()}", flush=True)

    trainer, loader = build_disagg_online_consumer(
        strategy="dflash",
        feature_store=store,
        channel=channel,
        eagle3_model=dflash_model,  # legacy param name = the trainable draft model
        optimizer_factory=optimizer_factory,
        run_id=RUN_ID,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        num_epochs=args.num_epochs,
        max_steps=max_steps,
        save_interval=args.save_interval,
        tp_size=args.tp_size,
        metadata_db_path=os.environ.get("DISAGG_DB") or None,
        logger=logger,
        log_interval=_max("DISAGG_LOG_INTERVAL", 1),
    )
    trainer.fit(loader)
    print(f"[consumer] DONE ({RUN_ID})", flush=True)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)

    role = _role()
    print(
        f"[disagg-dflash] role={role} run_id={RUN_ID} "
        f"node_rank={os.environ.get('RCLI_NODE_RANK', '0')}",
        flush=True,
    )
    try:
        if role == "producer":
            run_producer(args)
        else:
            run_consumer(args)
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
