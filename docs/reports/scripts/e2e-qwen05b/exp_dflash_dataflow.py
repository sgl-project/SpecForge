"""Experiment launcher: DFlash through the DataFlow runtime (+ offline dump).

Mirrors scripts/train_dflash.py's model assembly (build_models, mask token,
TargetEmbeddingsAndHead, OnlineDFlashModel) but hands orchestration to the
DataFlow launch builders with strategy="dflash". Extended via env vars so the
native train_dflash arg parser stays untouched:

  EXP_MODE=online|offline|dump   (default online)
  EXP_TOPO=colo|disagg           (default colo)
  EXP_HS_PATH=<dir>              dflash offline feature dir (dump target / offline source)
  EXP_STORE_ROOT=<dir>           SharedDirFeatureStore root for disagg
  EXP_MAX_PROMPTS=<n>            cap the prompt list
  EXP_MAX_STEPS=<n>              trainer max_steps

`dump` runs the target capture over the prompts and writes
{input_ids, loss_mask, hidden_states} .ckpt files (OfflineManifestReader
layout, dflash feature keys). Per-step loss prints as `CURVE step= loss=`.
"""

import os

import torch
from accelerate.utils import set_seed
from train_dflash import build_dataloader, build_models, parse_args
from transformers import AutoTokenizer

from specforge.core.dflash import OnlineDFlashModel
from specforge.distributed import destroy_distributed, init_distributed
from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead
from specforge.optimizer import BF16Optimizer

MODE = os.environ.get("EXP_MODE", "online")
ROLE = os.environ.get("EXP_ROLE", "")  # producer|consumer for TOPO=disagg2p
DB_PATH = os.environ.get("EXP_DB", "")
TOPO = os.environ.get("EXP_TOPO", "colo")
HS_PATH = os.environ.get("EXP_HS_PATH", "/root/exp/dumps/dflash-hs")
STORE_ROOT = os.environ.get("EXP_STORE_ROOT", "/root/exp/store")
MAX_PROMPTS = int(os.environ.get("EXP_MAX_PROMPTS", "0"))
MAX_STEPS = int(os.environ.get("EXP_MAX_STEPS", "100"))
PROMPTS_FROM = os.environ.get("EXP_PROMPTS_FROM", "")  # "jsonl:<path>" bypasses
# this tree's dataloader and feeds main-rendered tokens.


def _prompts_from_jsonl(path):
    import json

    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prompts.append(
                {"payload": {"input_ids": r["input_ids"], "loss_mask": r["loss_mask"]}}
            )
    return prompts


DFLASH_KEYS = ("input_ids", "loss_mask", "hidden_states")


def _logger(metrics, step):
    loss = metrics.get("loss")
    try:
        loss = float(loss)
    except (TypeError, ValueError):
        return
    print(f"CURVE step={step} loss={loss:.6f}", flush=True)


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


def _dump_features(target_model, prompts, out_dir, device):
    """Capture dflash features per prompt and write OfflineManifestReader files."""
    os.makedirs(out_dir, exist_ok=True)
    for i, prompt in enumerate(prompts):
        ids = torch.as_tensor(
            prompt["payload"]["input_ids"], dtype=torch.long, device=device
        ).unsqueeze(0)
        lm = torch.as_tensor(
            prompt["payload"]["loss_mask"], dtype=torch.long, device=device
        ).unsqueeze(0)
        attn = torch.ones_like(ids)
        out = target_model.capture(input_ids=ids, attention_mask=attn, loss_mask=lm)
        record = {
            "input_ids": out.input_ids[0].cpu(),
            "loss_mask": out.loss_mask[0].cpu(),
            "hidden_states": out.hidden_states.cpu(),  # [1, seq, W]
        }
        torch.save(record, os.path.join(out_dir, f"sample_{i:08d}.ckpt"))
    print(f"[dump] wrote {len(prompts)} dflash feature files to {out_dir}", flush=True)


def _ingest_dflash(store, hs_path, run_id):
    from specforge.runtime.data_plane.feature_store import load_feature_file
    from specforge.runtime.data_plane.offline_reader import list_feature_files

    refs = []
    for i, path in enumerate(list_feature_files(hs_path)):
        raw = load_feature_file(path)
        tensors = {k: raw[k] for k in DFLASH_KEYS}
        ref = store.put(
            tensors,
            sample_id=f"{run_id}:{i:08d}",
            metadata={
                "run_id": run_id,
                "strategy": "dflash",
                "format": "offline_dflash",
                "target_repr": None,
                "num_tokens": int(tensors["input_ids"].numel()),
                "file_index": i,
            },
        )
        refs.append(ref)
    return refs


def main():
    args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)

    run_id = f"df-{MODE}-{TOPO}"
    device = torch.device("cuda")

    target_model, draft_model = build_models(args)

    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    if args.mask_token_id is not None:
        mask_token_id = args.mask_token_id
    elif tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id
    else:
        tokenizer.add_special_tokens({"mask_token": "<|MASK|>"})
        mask_token_id = tokenizer.mask_token_id
    draft_model.mask_token_id = mask_token_id
    draft_model.config.dflash_config["mask_token_id"] = mask_token_id
    draft_model.config.dflash_config["target_layer_ids"] = draft_model.target_layer_ids

    if PROMPTS_FROM.startswith("jsonl:"):
        prompts = _prompts_from_jsonl(PROMPTS_FROM[len("jsonl:") :])
    else:
        train_dataloader, _eval = build_dataloader(args, tokenizer)
        prompts = _extract_prompts(train_dataloader)
    if MAX_PROMPTS:
        prompts = prompts[:MAX_PROMPTS]
    print(f"[{run_id}] {len(prompts)} prompts", flush=True)

    store_root = os.path.join(STORE_ROOT, run_id)
    if TOPO == "disagg2p" and MODE == "online" and ROLE == "producer":
        # INFERENCE POOL: target only; skip draft-side assembly below.
        from specforge.launch import build_disagg_online_producer
        from specforge.runtime.data_plane.disaggregated import SharedDirFeatureStore
        from specforge.runtime.data_plane.streaming_ref_channel import (
            StreamingRefChannel,
        )

        store = SharedDirFeatureStore(store_root, store_id=run_id)
        channel = StreamingRefChannel(os.path.join(store_root, "refs.jsonl"))
        workers, drive = build_disagg_online_producer(
            strategy="dflash",
            target_model=target_model,
            prompts=prompts,
            feature_store=store,
            channel=channel,
            run_id=run_id,
            target_hidden_size=int(draft_model.config.hidden_size),
            target_repr=None,
            metadata_db_path=DB_PATH or None,
        )
        produced = drive()
        print(f"[{run_id}] 2proc producer put {produced} samples", flush=True)
        destroy_distributed()
        return

    if MODE == "dump":
        _dump_features(target_model, prompts, HS_PATH, device)
        destroy_distributed()
        return

    target_components = TargetEmbeddingsAndHead.from_pretrained(
        args.target_model_path,
        embed_key=args.embedding_key,
        lm_head_key=args.lm_head_key,
        device=device.type,
        trust_remote_code=args.trust_remote_code,
    )
    dflash_model = OnlineDFlashModel(
        draft_model=draft_model,
        target_lm_head=target_components.lm_head,
        target_embed_tokens=target_components.embed_tokens,
        block_size=draft_model.block_size,
        mask_token_id=mask_token_id,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        loss_type=args.loss_type,
        dpace_alpha=args.dpace_alpha,
    )

    def optimizer_factory(draft_module):
        return BF16Optimizer(
            draft_module,
            lr=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            # keep the LR schedule identical to the patched main baseline
            total_steps=int(os.environ.get("EXP_TOTAL_STEPS", "0")) or MAX_STEPS,
        )

    hidden_size = int(draft_model.config.hidden_size)

    if TOPO == "disagg2p" and MODE == "online" and ROLE == "consumer":
        # TRAINER POOL: no target-model forward in this process.
        from specforge.launch import build_disagg_online_consumer
        from specforge.runtime.data_plane.disaggregated import SharedDirFeatureStore
        from specforge.runtime.data_plane.streaming_ref_channel import (
            StreamingRefChannel,
        )

        store = SharedDirFeatureStore(store_root, store_id=run_id)
        channel = StreamingRefChannel(os.path.join(store_root, "refs.jsonl"))
        trainer, loader = build_disagg_online_consumer(
            strategy="dflash",
            feature_store=store,
            channel=channel,
            eagle3_model=dflash_model,
            optimizer_factory=optimizer_factory,
            run_id=run_id,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            num_epochs=1,
            max_steps=MAX_STEPS,
            save_interval=0,
            tp_size=args.tp_size,
            metadata_db_path=DB_PATH or None,
            logger=_logger,
            log_interval=1,
        )
        trainer.fit(loader)
        print(f"[{run_id}] DONE", flush=True)
        destroy_distributed()
        return
    if TOPO == "disagg2p" and MODE == "offline":
        from specforge.runtime.data_plane.disagg_ingest import (
            read_ref_manifest,
            write_ref_manifest,
        )
        from specforge.runtime.data_plane.disaggregated import SharedDirFeatureStore

        manifest = os.path.join(store_root, "manifest.json")
        if ROLE == "producer":
            store = SharedDirFeatureStore(
                store_root, store_id=run_id, retain_on_release=True
            )
            refs = _ingest_dflash(store, HS_PATH, run_id)
            write_ref_manifest(refs, manifest)
            print(f"[{run_id}] 2proc producer ingested {len(refs)}", flush=True)
            destroy_distributed()
            return
        from specforge.launch import build_disagg_offline_runtime

        store = SharedDirFeatureStore(
            store_root, store_id=run_id, retain_on_release=True
        )
        refs = read_ref_manifest(manifest)
        trainer, loader = build_disagg_offline_runtime(
            strategy="dflash",
            feature_store=store,
            refs=refs,
            eagle3_model=dflash_model,
            target_head=None,
            optimizer_factory=optimizer_factory,
            run_id=run_id,
            output_dir=args.output_dir,
            max_len=args.max_length,
            batch_size=args.batch_size,
            accumulation_steps=args.accumulation_steps,
            num_epochs=args.num_epochs,
            max_steps=MAX_STEPS,
            save_interval=0,
            tp_size=args.tp_size,
            logger=_logger,
            log_interval=1,
        )
        trainer.fit(loader)
        print(f"[{run_id}] DONE", flush=True)
        destroy_distributed()
        return

    if MODE == "online":
        if TOPO == "colo":
            from specforge.launch import build_online_runtime

            trainer, loader, workers, controller, drive_rollout = build_online_runtime(
                strategy="dflash",
                target_model=target_model,
                prompts=prompts,
                eagle3_model=dflash_model,
                optimizer_factory=optimizer_factory,
                run_id=run_id,
                output_dir=args.output_dir,
                target_hidden_size=hidden_size,
                target_repr=None,
                batch_size=args.batch_size,
                accumulation_steps=args.accumulation_steps,
                num_epochs=1,
                max_steps=MAX_STEPS,
                save_interval=0,
                tp_size=args.tp_size,
                logger=_logger,
                log_interval=1,
            )
            produced = drive_rollout()
            print(f"[{run_id}] rollout produced {produced} samples", flush=True)
            trainer.fit(loader)
        else:
            # STAGED disagg (produce-all, then train): the interleaved runner
            # forwards the target in a producer THREAD, and DFlash's HF capture
            # (output_hidden_states=True) trips transformers' non-thread-safe
            # hook-recording wrapper while the consumer forwards the draft.
            # Same disagg store + ref channel, sequential phases.
            from specforge.launch import (
                build_disagg_online_consumer,
                build_disagg_online_producer,
            )
            from specforge.runtime.data_plane.disaggregated import SharedDirFeatureStore
            from specforge.runtime.data_plane.streaming_ref_channel import (
                StreamingRefChannel,
            )

            store_root = os.path.join(STORE_ROOT, run_id)
            store = SharedDirFeatureStore(store_root, store_id=run_id)
            channel = StreamingRefChannel(os.path.join(store_root, "refs.jsonl"))
            workers, drive_producer = build_disagg_online_producer(
                strategy="dflash",
                target_model=target_model,
                prompts=prompts,
                feature_store=store,
                channel=channel,
                run_id=run_id,
                target_hidden_size=hidden_size,
                target_repr=None,
                in_flight_high_watermark=1_000_000,  # staged: no backpressure
            )
            produced = drive_producer()
            print(f"[{run_id}] staged producer put {produced} samples", flush=True)
            trainer, loader = build_disagg_online_consumer(
                strategy="dflash",
                feature_store=store,
                channel=channel,
                eagle3_model=dflash_model,
                optimizer_factory=optimizer_factory,
                run_id=run_id,
                output_dir=args.output_dir,
                batch_size=args.batch_size,
                accumulation_steps=args.accumulation_steps,
                num_epochs=1,
                max_steps=MAX_STEPS,
                save_interval=0,
                tp_size=args.tp_size,
                logger=_logger,
                log_interval=1,
            )
            trainer.fit(loader)
    else:  # offline
        if TOPO == "colo":
            from specforge.launch import build_offline_runtime

            trainer, loader = build_offline_runtime(
                strategy="dflash",
                hidden_states_path=HS_PATH,
                eagle3_model=dflash_model,
                target_head=None,
                optimizer_factory=optimizer_factory,
                run_id=run_id,
                output_dir=args.output_dir,
                max_len=args.max_length,
                batch_size=args.batch_size,
                accumulation_steps=args.accumulation_steps,
                num_epochs=args.num_epochs,
                max_steps=MAX_STEPS,
                save_interval=0,
                tp_size=args.tp_size,
                logger=_logger,
                log_interval=1,
            )
            trainer.fit(loader)
        else:
            from specforge.launch import build_disagg_offline_runtime
            from specforge.runtime.data_plane.disaggregated import SharedDirFeatureStore

            store_root = os.path.join(STORE_ROOT, run_id)
            store = SharedDirFeatureStore(
                store_root, store_id=run_id, retain_on_release=True
            )
            refs = _ingest_dflash(store, HS_PATH, run_id)
            print(f"[{run_id}] ingested {len(refs)} offline samples", flush=True)
            trainer, loader = build_disagg_offline_runtime(
                strategy="dflash",
                feature_store=store,
                refs=refs,
                eagle3_model=dflash_model,
                target_head=None,
                optimizer_factory=optimizer_factory,
                run_id=run_id,
                output_dir=args.output_dir,
                max_len=args.max_length,
                batch_size=args.batch_size,
                accumulation_steps=args.accumulation_steps,
                num_epochs=args.num_epochs,
                max_steps=MAX_STEPS,
                save_interval=0,
                tp_size=args.tp_size,
                logger=_logger,
                log_interval=1,
            )
            trainer.fit(loader)

    print(f"[{run_id}] DONE", flush=True)
    destroy_distributed()


if __name__ == "__main__":
    main()
