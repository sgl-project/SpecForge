"""Disaggregated offline EAGLE3 example: producer and consumer on different pools.

This is the *assemble* example for the M6 disaggregation seam. It runs the SAME
offline EAGLE3 training as ``scripts/train_eagle3_dataflow.py`` (reusing its model
builders, so results align with the colocated run), but splits the work across two
pools that share only a filesystem mount:

* **producer** (rollout/feature pool): ``ingest_offline_features`` loads the
  precomputed ``.ckpt`` features and ``put()``s them into a
  ``SharedDirFeatureStore`` on the shared mount, then publishes a tensor-free ref
  manifest. No model, no trainer.
* **consumer** (training pool): waits for the manifest, builds the EAGLE3 model
  exactly as the offline launcher does, and trains via
  ``build_disagg_eagle3_runtime`` reading features from the shared store.

The control plane carries only ``SampleRef`` metadata across the boundary; the
feature tensors travel through the shared store. Disaggregation changes *where*
features live, not their values, so the training curve matches the colocated
offline baseline.

Role is taken from ``DISAGG_ROLE`` (``producer``/``consumer``); if unset it is
derived from ``RCLI_NODE_RANK`` (0 -> producer, else consumer). Shared paths +
auth come from the environment so one wrapper can drive both nodes:

    DISAGG_STORE_ROOT=/workspace/disagg_store   # shared mount, both pools
    DISAGG_MANIFEST=/workspace/disagg_store/refs.json
    DISAGG_STORE_ID=eagle3-disagg               # producer/consumer must match
    DISAGG_AUTH_TOKEN=<secret>                   # optional (B9 auth)
"""

import os
import time

from accelerate.utils import set_seed

from specforge.distributed import destroy_distributed, init_distributed
from specforge.optimizer import BF16Optimizer
from specforge.runtime.data_plane.disagg_ingest import (
    ingest_offline_features,
    read_ref_manifest,
    write_ref_manifest,
)
from specforge.runtime.data_plane.disaggregated import AuthPolicy, SharedDirFeatureStore
from specforge.runtime.launch import build_disagg_eagle3_runtime

# reuse the existing builders so model construction matches the offline path
from train_eagle3 import (
    build_dataloaders,
    build_draft_model,
    build_target_model,
    parse_args,
    sanity_check,
)

RUN_ID = "eagle3-disagg"


def _role() -> str:
    role = os.environ.get("DISAGG_ROLE")
    if role:
        return role
    return "producer" if os.environ.get("RCLI_NODE_RANK", "0") == "0" else "consumer"


def _store(args) -> SharedDirFeatureStore:
    token = os.environ.get("DISAGG_AUTH_TOKEN") or None
    return SharedDirFeatureStore(
        os.environ["DISAGG_STORE_ROOT"],
        store_id=os.environ.get("DISAGG_STORE_ID", RUN_ID),
        auth=AuthPolicy(token),
        credential=token,
    )


def run_producer(args) -> None:
    manifest = os.environ["DISAGG_MANIFEST"]
    store = _store(args)
    refs = ingest_offline_features(
        store,
        args.train_hidden_states_path,
        run_id=RUN_ID,
        ttt_length=args.ttt_length,
        max_len=args.max_length,
    )
    write_ref_manifest(refs, manifest)
    open(manifest + ".done", "w").close()  # liveness marker the consumer waits on
    print(
        f"[producer] ingested {len(refs)} samples into {store.root}; "
        f"manifest -> {manifest}",
        flush=True,
    )


def run_consumer(args) -> None:
    manifest = os.environ["DISAGG_MANIFEST"]
    init_distributed(
        timeout=args.dist_timeout,
        tp_size=args.tp_size,
        sp_ring_size=args.sp_ring_size,
        sp_ulysses_size=args.sp_ulysses_size,
    )
    sanity_check(args)  # derives target_batch_size/dp_size the builders read (needs dist)
    # wait for the producer to publish the manifest (shared mount)
    deadline = time.monotonic() + 1800
    while not os.path.exists(manifest + ".done"):
        if time.monotonic() > deadline:
            raise SystemExit(f"[consumer] timed out waiting for {manifest}.done")
        time.sleep(2)

    draft_config, draft_model, _ckpt, _resume = build_draft_model(args)
    target_head, _ = build_target_model(args, draft_config, is_online=False)
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

    def optimizer_factory(draft_module):
        return BF16Optimizer(
            draft_module,
            lr=args.learning_rate,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            total_steps=args.total_steps or 10_000,
        )

    store = _store(args)
    refs = read_ref_manifest(manifest)
    print(f"[consumer] training from {len(refs)} disagg refs in {store.root}", flush=True)

    trainer, loader = build_disagg_eagle3_runtime(
        feature_store=store,
        refs=refs,
        eagle3_model=eagle3_model,
        target_head=target_head,
        optimizer_factory=optimizer_factory,
        run_id=RUN_ID,
        output_dir=args.output_dir,
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


def main() -> None:
    parser, args = parse_args()
    set_seed(args.seed)
    if args.train_hidden_states_path is None:
        raise SystemExit("disagg example wires the OFFLINE path; pass --train-hidden-states-path")
    role = _role()
    print(f"[disagg] role={role} node_rank={os.environ.get('RCLI_NODE_RANK', '0')}", flush=True)
    if role == "producer":
        run_producer(args)
    else:
        run_consumer(args)


if __name__ == "__main__":
    main()
