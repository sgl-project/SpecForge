"""Disaggregated ONLINE DFlash example — real server-capture zero-copy transport.

Inference and training are fully decoupled processes that share only a Mooncake
object store (RDMA-capable, no shared *data* mount):

* **producer** (inference pool): a live SGLang server — patched with
  ``patches/sglang/v0.5.14/spec-capture.patch`` and launched with
  ``--enable-spec-capture`` — runs the frozen Qwen3.6-27B target and writes the
  captured DFlash context hidden states straight into Mooncake. This process is
  a thin driver: it sends prompts through ``SGLangServerCaptureAdapter`` and
  streams the resulting tensor-free ``SampleRef``s to the consumer. No target
  weights, no draft, no GPU model here.
* **consumer** (training pool): trains the DFlash draft (``OnlineDFlashModel``),
  reading refs off the channel and features from Mooncake — zero re-copy.

This replaces the old in-process producer (which loaded the 27B target locally
and ran staged, since DFlash's HF capture is not thread-safe): the server does
the capture, so a single prefill per prompt feeds training directly. It is the
disaggregated sibling of ``examples/run_qwen3.6_27b_dflash_online.sh``.

Config is environment-driven so one wrapper drives both roles; role is
``DISAGG_ROLE`` (``producer``/``consumer``) or derived from ``RCLI_NODE_RANK``:

    DISAGG_ROLE=producer|consumer      # else rank 0 -> producer, else consumer
    DISAGG_SERVER_URL=http://host:30000   # the patched SGLang server (producer)
    DISAGG_STORE_ID=qwen36-dflash-disagg  # Mooncake key namespace (both roles)
    DISAGG_REF_CHANNEL=/root/disagg/refs.jsonl   # tiny ref manifest (both roles)
    DISAGG_DB=/root/disagg/run.db      # shared durable metadata store
    DISAGG_MAX_PROMPTS=400             # cap the prompt pool (0 = all)
    DISAGG_MAX_STEPS=0                 # trainer max optimizer steps (0 = all)

Mooncake connection uses the standard ``MOONCAKE_*`` env vars (see
``examples/disagg/README.md``); the server sink reads the same ones, so both
sides land on one master. Export ``WANDB_API_KEY`` for consumer W&B logging.
"""

import json
import os

from accelerate.utils import set_seed
from train_dflash import build_dataloader, build_models, parse_args
from transformers import AutoTokenizer

from specforge.core.dflash import OnlineDFlashModel
from specforge.distributed import destroy_distributed, init_distributed
from specforge.inference.adapters.server_capture import SGLangServerCaptureAdapter
from specforge.launch import build_disagg_online_consumer, build_disagg_online_producer
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from specforge.tracker import create_tracker

RUN_ID = os.environ.get("DISAGG_STORE_ID", "qwen36-dflash-disagg")


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
    """Connect to the same Mooncake master the server sink writes to."""
    return MooncakeFeatureStore(
        store_id=RUN_ID,
        setup_kwargs={
            "local_hostname": os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "127.0.0.1"),
            "metadata_server": os.environ["MOONCAKE_METADATA_SERVER"],
            "master_server_addr": os.environ["MOONCAKE_MASTER_SERVER_ADDR"],
            "protocol": os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
            "rdma_devices": os.environ.get("MOONCAKE_RDMA_DEVICES", ""),
        },
    )


def _extract_prompts(train_dataloader):
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


def _resolve_mask_token(args, tokenizer) -> int:
    if args.mask_token_id is not None:
        return args.mask_token_id
    if tokenizer.mask_token_id is not None:
        return tokenizer.mask_token_id
    tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
    return tokenizer.mask_token_id


def run_producer(args) -> None:
    """Thin driver: prompts -> patched server (captures to Mooncake) -> refs."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.target_model_path, trust_remote_code=args.trust_remote_code
    )
    train_dataloader, _eval = build_dataloader(args, tokenizer)
    prompts = _extract_prompts(train_dataloader)
    max_prompts = _max("DISAGG_MAX_PROMPTS", 0)
    if max_prompts:
        prompts = prompts[:max_prompts]

    # DFlash captures the concatenation of these target layers; the shell passes
    # the same list to the server's --spec-capture-aux-layer-ids.
    draft_cfg = json.load(open(args.draft_config_path))
    aux_layer_ids = tuple(draft_cfg["dflash_config"]["target_layer_ids"])
    target_hidden = int(draft_cfg["hidden_size"])

    store = _mooncake_store()
    channel = _ref_channel()
    adapter = SGLangServerCaptureAdapter(
        os.environ["DISAGG_SERVER_URL"],
        store,
        run_id=RUN_ID,
        strategy="dflash",
        target_model_version=args.target_model_path,
    )
    print(f"[producer] {len(prompts)} prompts -> {os.environ['DISAGG_SERVER_URL']}")

    _workers, drive_producer = build_disagg_online_producer(
        strategy="dflash",
        feature_source=adapter,
        prompts=prompts,
        feature_store=store,
        channel=channel,
        run_id=RUN_ID,
        target_hidden_size=target_hidden,
        target_repr=None,  # DFlash trains on captured hidden states, no target dist.
        aux_hidden_state_layer_ids=aux_layer_ids,
        metadata_db_path=os.environ.get("DISAGG_DB") or None,
    )
    produced = drive_producer()
    print(f"[producer] streamed {produced} samples; channel closed", flush=True)


def run_consumer(args) -> None:
    """Training pool: train the DFlash draft off Mooncake + the ref channel."""
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    _target_unused, draft_model = build_models(args)
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

    print(f"[consumer] training from mooncake://{RUN_ID}", flush=True)
    trainer, loader = build_disagg_online_consumer(
        strategy="dflash",
        feature_store=_mooncake_store(),
        channel=_ref_channel(),
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
    role = _role()
    print(f"[disagg-dflash] role={role} run_id={RUN_ID}", flush=True)
    # The producer is a thin HTTP driver (no torch model); only the trainer
    # needs the process group.
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
