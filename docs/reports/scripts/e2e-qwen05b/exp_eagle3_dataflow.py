"""Experiment launcher: EAGLE3 through the DataFlow runtime, colo or disagg.

Same wiring as scripts/train_eagle3_dataflow.py, extended via env vars so the
native train_eagle3 arg parser stays untouched:

  EXP_TOPO=colo|disagg      topology (default colo)
  EXP_STORE_ROOT=<dir>      SharedDirFeatureStore root for disagg
  EXP_MAX_PROMPTS=<n>       cap the online prompt list (rollout cost control)

Per-step loss is printed as `CURVE step=<s> loss=<x>` (log_interval=1).
"""

import os

from accelerate.utils import set_seed
from train_eagle3 import (
    build_dataloaders,
    build_draft_model,
    build_target_model,
    parse_args,
)
from train_eagle3_dataflow import _extract_prompts, _target_hidden_and_vocab

from specforge.distributed import destroy_distributed, init_distributed
from specforge.optimizer import BF16Optimizer

TOPO = os.environ.get("EXP_TOPO", "colo")
ROLE = os.environ.get("EXP_ROLE", "")  # producer|consumer for TOPO=disagg2p
DB_PATH = os.environ.get("EXP_DB", "")
STORE_ROOT = os.environ.get("EXP_STORE_ROOT", "/root/exp/store")
MAX_PROMPTS = int(os.environ.get("EXP_MAX_PROMPTS", "0"))
PROMPTS_FROM = os.environ.get("EXP_PROMPTS_FROM", "")  # "dump:<dir>" bypasses
# this tree's jsonl rendering (its batch-composition-dependent stub bug) and
# feeds the exact tokens main rendered.


def _prompts_from_dump(dump_dir):
    import glob

    import torch

    prompts = []
    for p in sorted(glob.glob(os.path.join(dump_dir, "*.ckpt"))):
        r = torch.load(p, weights_only=False)
        prompts.append(
            {
                "payload": {
                    "input_ids": r["input_ids"].tolist(),
                    "loss_mask": r["loss_mask"].tolist(),
                }
            }
        )
    return prompts


def _logger(metrics, step):
    loss = metrics.get("loss")
    try:
        loss = float(loss)
    except (TypeError, ValueError):
        return
    print(f"CURVE step={step} loss={loss:.6f}", flush=True)


def main():
    parser, args = parse_args()
    args.target_batch_size = args.tp_size * args.batch_size
    set_seed(args.seed)
    init_distributed(
        timeout=args.dist_timeout,
        tp_size=args.tp_size,
        sp_ring_size=args.sp_ring_size,
        sp_ulysses_size=args.sp_ulysses_size,
    )

    online = args.train_hidden_states_path is None
    run_id = f"e3-{'online' if online else 'offline'}-{TOPO}"

    draft_config, draft_model, _ckpt, _resume = build_draft_model(args)
    train_dataloader, vocab_mapping_path, _eval = build_dataloaders(args, draft_config)
    draft_model.load_vocab_mapping(vocab_mapping_path)

    from specforge import OnlineEagle3Model

    eagle3_model = OnlineEagle3Model(
        draft_model=draft_model,
        length=args.ttt_length,
        attention_backend=args.attention_backend,
        lk_loss_type=args.lk_loss_type,
        kl_scale=args.kl_scale,
        kl_decay=args.kl_decay,
    ).cuda()

    def optimizer_factory(draft_module):
        return BF16Optimizer(
            draft_module,
            lr=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            total_steps=args.total_steps or 10_000,
        )

    store_root = os.path.join(STORE_ROOT, run_id)
    if TOPO == "disagg2p" and online and ROLE == "producer":
        # INFERENCE POOL: target model only; trainer lives in another process.
        from specforge.launch import build_disagg_online_producer
        from specforge.runtime.data_plane.disaggregated import SharedDirFeatureStore
        from specforge.runtime.data_plane.streaming_ref_channel import (
            StreamingRefChannel,
        )

        target_model, _ = build_target_model(args, draft_config, is_online=True)
        hidden_size, vocab_size = _target_hidden_and_vocab(target_model)
        if PROMPTS_FROM.startswith("dump:"):
            prompts = _prompts_from_dump(PROMPTS_FROM[len("dump:") :])
        else:
            prompts = _extract_prompts(train_dataloader)
        if MAX_PROMPTS:
            prompts = prompts[:MAX_PROMPTS]
        store = SharedDirFeatureStore(store_root, store_id=run_id)
        channel = StreamingRefChannel(os.path.join(store_root, "refs.jsonl"))
        workers, drive = build_disagg_online_producer(
            strategy="eagle3",
            target_model=target_model,
            prompts=prompts,
            feature_store=store,
            channel=channel,
            run_id=run_id,
            target_hidden_size=hidden_size,
            target_vocab_size=vocab_size,
            target_repr="logits",
            metadata_db_path=DB_PATH or None,
        )
        produced = drive()
        print(f"[{run_id}] 2proc producer put {produced} samples", flush=True)
        destroy_distributed()
        return
    if TOPO == "disagg2p" and online and ROLE == "consumer":
        # TRAINER POOL: no target model in this process.
        from specforge.launch import build_disagg_online_consumer
        from specforge.runtime.data_plane.disaggregated import SharedDirFeatureStore
        from specforge.runtime.data_plane.streaming_ref_channel import (
            StreamingRefChannel,
        )

        store = SharedDirFeatureStore(store_root, store_id=run_id)
        channel = StreamingRefChannel(os.path.join(store_root, "refs.jsonl"))
        trainer, loader = build_disagg_online_consumer(
            strategy="eagle3",
            feature_store=store,
            channel=channel,
            eagle3_model=eagle3_model,
            optimizer_factory=optimizer_factory,
            run_id=run_id,
            output_dir=args.output_dir,
            batch_size=args.target_batch_size,
            accumulation_steps=args.draft_accumulation_steps,
            num_epochs=1,
            max_steps=args.max_num_steps,
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
    if TOPO == "disagg2p" and not online:
        from specforge.runtime.data_plane.disagg_ingest import (
            ingest_offline_features,
            read_ref_manifest,
            write_ref_manifest,
        )
        from specforge.runtime.data_plane.disaggregated import SharedDirFeatureStore

        manifest = os.path.join(store_root, "manifest.json")
        if ROLE == "producer":
            # FEATURE POOL: no models at all — load .ckpt dumps into the store.
            store = SharedDirFeatureStore(
                store_root, store_id=run_id, retain_on_release=True
            )
            refs = ingest_offline_features(
                store,
                args.train_hidden_states_path,
                run_id=run_id,
                ttt_length=args.ttt_length,
                max_len=args.max_length,
            )
            write_ref_manifest(refs, manifest)
            print(f"[{run_id}] 2proc producer ingested {len(refs)}", flush=True)
            destroy_distributed()
            return
        # consumer: TRAINER POOL
        from specforge.launch import build_disagg_offline_runtime

        target_head, _ = build_target_model(args, draft_config, is_online=False)
        store = SharedDirFeatureStore(
            store_root, store_id=run_id, retain_on_release=True
        )
        refs = read_ref_manifest(manifest)
        trainer, loader = build_disagg_offline_runtime(
            strategy="eagle3",
            feature_store=store,
            refs=refs,
            eagle3_model=eagle3_model,
            target_head=target_head,
            optimizer_factory=optimizer_factory,
            run_id=run_id,
            output_dir=args.output_dir,
            max_len=args.max_length,
            batch_size=args.target_batch_size,
            accumulation_steps=args.draft_accumulation_steps,
            num_epochs=args.num_epochs,
            max_steps=args.max_num_steps,
            save_interval=0,
            tp_size=args.tp_size,
            logger=_logger,
            log_interval=1,
        )
        trainer.fit(loader)
        print(f"[{run_id}] DONE", flush=True)
        destroy_distributed()
        return

    if online:
        target_model, _ = build_target_model(args, draft_config, is_online=True)
        hidden_size, vocab_size = _target_hidden_and_vocab(target_model)
        if PROMPTS_FROM.startswith("dump:"):
            prompts = _prompts_from_dump(PROMPTS_FROM[len("dump:") :])
        else:
            prompts = _extract_prompts(train_dataloader)
        if MAX_PROMPTS:
            prompts = prompts[:MAX_PROMPTS]
        print(f"[{run_id}] rolling out {len(prompts)} prompts", flush=True)

        if TOPO == "colo":
            from specforge.launch import build_online_runtime

            trainer, loader, workers, controller, drive_rollout = build_online_runtime(
                strategy="eagle3",
                target_model=target_model,
                prompts=prompts,
                eagle3_model=eagle3_model,
                optimizer_factory=optimizer_factory,
                run_id=run_id,
                output_dir=args.output_dir,
                target_hidden_size=hidden_size,
                target_vocab_size=vocab_size,
                target_repr="logits",
                ttt_length=args.ttt_length,
                batch_size=args.target_batch_size,
                accumulation_steps=args.draft_accumulation_steps,
                num_epochs=1,
                max_steps=args.max_num_steps,
                save_interval=0,
                tp_size=args.tp_size,
                sp_ulysses_size=args.sp_ulysses_size,
                sp_ring_size=args.sp_ring_size,
                logger=_logger,
                log_interval=1,
            )
            produced = drive_rollout()
            print(f"[{run_id}] rollout produced {produced} samples", flush=True)
            trainer.fit(loader)
        else:
            from specforge.launch import build_disagg_online_runtime
            from specforge.runtime.data_plane.disaggregated import SharedDirFeatureStore

            store_root = os.path.join(STORE_ROOT, run_id)
            store = SharedDirFeatureStore(store_root, store_id=run_id)
            trainer, loader, run = build_disagg_online_runtime(
                strategy="eagle3",
                target_model=target_model,
                prompts=prompts,
                eagle3_model=eagle3_model,
                optimizer_factory=optimizer_factory,
                feature_store=store,
                run_id=run_id,
                output_dir=args.output_dir,
                target_hidden_size=hidden_size,
                ref_channel_path=os.path.join(store_root, "refs.jsonl"),
                target_vocab_size=vocab_size,
                target_repr="logits",
                batch_size=args.target_batch_size,
                accumulation_steps=args.draft_accumulation_steps,
                num_epochs=1,
                max_steps=args.max_num_steps,
                save_interval=0,
                tp_size=args.tp_size,
                sp_ulysses_size=args.sp_ulysses_size,
                sp_ring_size=args.sp_ring_size,
                logger=_logger,
                log_interval=1,
            )
            steps = run()
            print(f"[{run_id}] disagg run finished at step {steps}", flush=True)
    else:
        target_head, _ = build_target_model(args, draft_config, is_online=False)
        if TOPO == "colo":
            from specforge.launch import build_offline_runtime

            trainer, loader = build_offline_runtime(
                strategy="eagle3",
                hidden_states_path=args.train_hidden_states_path,
                eagle3_model=eagle3_model,
                target_head=target_head,
                optimizer_factory=optimizer_factory,
                run_id=run_id,
                output_dir=args.output_dir,
                ttt_length=args.ttt_length,
                max_len=args.max_length,
                batch_size=args.target_batch_size,
                accumulation_steps=args.draft_accumulation_steps,
                num_epochs=args.num_epochs,
                max_steps=args.max_num_steps,
                save_interval=0,
                tp_size=args.tp_size,
                sp_ulysses_size=args.sp_ulysses_size,
                sp_ring_size=args.sp_ring_size,
                logger=_logger,
                log_interval=1,
            )
            trainer.fit(loader)
        else:
            from specforge.launch import build_disagg_offline_runtime
            from specforge.runtime.data_plane.disagg_ingest import (
                ingest_offline_features,
            )
            from specforge.runtime.data_plane.disaggregated import SharedDirFeatureStore

            store_root = os.path.join(STORE_ROOT, run_id)
            store = SharedDirFeatureStore(
                store_root, store_id=run_id, retain_on_release=True
            )
            refs = ingest_offline_features(
                store,
                args.train_hidden_states_path,
                run_id=run_id,
                ttt_length=args.ttt_length,
                max_len=args.max_length,
            )
            print(f"[{run_id}] ingested {len(refs)} offline samples", flush=True)
            trainer, loader = build_disagg_offline_runtime(
                strategy="eagle3",
                feature_store=store,
                refs=refs,
                eagle3_model=eagle3_model,
                target_head=target_head,
                optimizer_factory=optimizer_factory,
                run_id=run_id,
                output_dir=args.output_dir,
                max_len=args.max_length,
                batch_size=args.target_batch_size,
                accumulation_steps=args.draft_accumulation_steps,
                num_epochs=args.num_epochs,
                max_steps=args.max_num_steps,
                save_interval=0,
                tp_size=args.tp_size,
                sp_ulysses_size=args.sp_ulysses_size,
                sp_ring_size=args.sp_ring_size,
                logger=_logger,
                log_interval=1,
            )
            trainer.fit(loader)

    print(f"[{run_id}] DONE", flush=True)
    destroy_distributed()


if __name__ == "__main__":
    main()
