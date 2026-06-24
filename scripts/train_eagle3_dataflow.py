"""Thin launcher: offline EAGLE3 training through the SpecForge DataFlow runtime.

This script is a *launcher* (M3): it builds models + optimizer, hands them to the
runtime, and runs ``TrainerController.fit``. No training logic lives here — the
loop, loss, projection, checkpoint, and eval all live in ``specforge.runtime``.

Reuses the existing model/data builders from ``scripts.train_eagle3`` so model
construction stays DRY; only the *orchestration* moves behind the runtime.

Example (offline):
    torchrun --standalone --nproc_per_node 1 scripts/train_eagle3_dataflow.py \
        --target-model-path <hf-model> --draft-model-config configs/llama3-8B-eagle3.json \
        --train-data-path <prompts.jsonl> --train-hidden-states-path <features_dir> \
        --output-dir ./output --max-num-steps 20
"""

from accelerate.utils import set_seed

# reuse existing builders so model construction is not duplicated
from train_eagle3 import (
    build_dataloaders,
    build_draft_model,
    build_target_model,
    parse_args,
)

from specforge.distributed import destroy_distributed, init_distributed
from specforge.optimizer import BF16Optimizer
from specforge.runtime.launch import build_offline_eagle3_runtime


def main():
    parser, args = parse_args()
    # parse_args() does not derive target_batch_size (train_eagle3.main computes
    # it inline before building dataloaders); the offline runtime builder and
    # build_dataloaders both read it, so derive it here too.
    args.target_batch_size = args.tp_size * args.batch_size

    # TODO(dataflow-launcher parity with scripts/train_eagle3.py): this launcher
    # covers core EAGLE3 training (offline + online: loss / projection / FSDP /
    # TP / grad-accum / checkpoint), validated old-vs-new. The following
    # train_eagle3.py features are NOT yet wired here and still require the
    # legacy script:
    #   - VLM / multimodal targets (--is-vlm, QwenVLOnlineEagle3Model)
    #   - USP sequence parallelism (--attention-backend usp -> process_data_usp;
    #     this path uses OfflineEagle3Dataset.process_data, no per-rank seq shard)
    #   - eval loop (--eval-data-path / --eval-hidden-states-path)
    #   - resume from checkpoint (--resume)
    #   - experiment trackers (--report-to wandb / swanlab / tensorboard)
    #   - online multi-epoch re-rollout (online runs a single consume-once pass)
    set_seed(args.seed)
    init_distributed(
        timeout=args.dist_timeout,
        tp_size=args.tp_size,
        sp_ring_size=args.sp_ring_size,
        sp_ulysses_size=args.sp_ulysses_size,
    )
    if args.train_hidden_states_path is None:
        raise SystemExit(
            "train_eagle3_dataflow currently wires the OFFLINE path; pass "
            "--train-hidden-states-path. (Online wiring composes RolloutWorker + "
            "SGLangAdapter over the same control/data plane.)"
        )

    draft_config, draft_model, _ckpt, _resume = build_draft_model(args)
    target_head, _ = build_target_model(args, draft_config, is_online=False)
    # vocab mapping is produced from the prompt dataset exactly as today
    _train, vocab_mapping_path, _eval = build_dataloaders(args, draft_config)
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

    # built AFTER FSDP-wrap (inside the runtime) over the wrapped inner draft
    def optimizer_factory(draft_module):
        return BF16Optimizer(
            draft_module,
            lr=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            total_steps=args.total_steps or 10_000,
        )

    trainer, loader = build_offline_eagle3_runtime(
        hidden_states_path=args.train_hidden_states_path,
        eagle3_model=eagle3_model,
        target_head=target_head,
        optimizer_factory=optimizer_factory,
        run_id="eagle3-offline",
        output_dir=args.output_dir,
        ttt_length=args.ttt_length,
        max_len=args.max_length,
        batch_size=args.target_batch_size,
        accumulation_steps=args.draft_accumulation_steps,
        num_epochs=args.num_epochs,
        max_steps=args.max_num_steps,
        save_interval=args.save_interval,
        tp_size=args.tp_size,
        sp_ulysses_size=args.sp_ulysses_size,
        sp_ring_size=args.sp_ring_size,
        logger=lambda m, s: print(f"step {s}: {m}", flush=True),
    )
    trainer.fit(loader)
    destroy_distributed()


if __name__ == "__main__":
    main()
