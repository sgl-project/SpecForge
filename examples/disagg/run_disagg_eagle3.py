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

Backend is selected by ``DISAGG_BACKEND`` (default ``shared_dir``):

* ``shared_dir`` — ``SharedDirFeatureStore`` over a shared POSIX mount.
* ``mooncake`` — ``MooncakeFeatureStore``, the M6 fast path: a network object
  store (RDMA/TCP), so the feature tensors need **no shared data mount**. Caveat:
  a Mooncake object lives in the *producer's* memory segment, so the producer
  process must stay alive until the consumer has read everything — this example
  holds the producer open until the consumer writes ``<manifest>.consumed`` (or
  ``DISAGG_PRODUCER_HOLD_S`` seconds elapse). The small ref manifest + sentinels
  still travel through ``DISAGG_MANIFEST``.

Role is taken from ``DISAGG_ROLE`` (``producer``/``consumer``/``colocated``); if
unset it is derived from ``RCLI_NODE_RANK`` (0 -> producer, else consumer).
Config comes from the environment so one wrapper can drive both nodes:

    DISAGG_MANIFEST=/workspace/disagg_store/refs.json  # small shared control plane
    DISAGG_STORE_ID=eagle3-disagg               # producer/consumer must match
    DISAGG_AUTH_TOKEN=<secret>                   # optional (B9 auth)
    # backend=shared_dir (default):
    DISAGG_STORE_ROOT=/workspace/disagg_store    # shared *data* mount
    # backend=mooncake:
    DISAGG_BACKEND=mooncake
    MOONCAKE_LOCAL_HOSTNAME=<this-node-ip>
    MOONCAKE_METADATA_SERVER=<metadata server url>
    MOONCAKE_MASTER_SERVER_ADDR=<master host:port>
    MOONCAKE_PROTOCOL=tcp                         # or "rdma"
"""

import os
import time

from accelerate.utils import set_seed

# reuse the existing builders so model construction matches the offline path
from train_eagle3 import (
    build_dataloaders,
    build_draft_model,
    build_target_model,
    parse_args,
    sanity_check,
)

from specforge.distributed import destroy_distributed, init_distributed
from specforge.optimizer import BF16Optimizer
from specforge.runtime.data_plane.disagg_ingest import (
    ingest_offline_features,
    read_ref_manifest,
    write_ref_manifest,
)
from specforge.runtime.data_plane.disaggregated import AuthPolicy, SharedDirFeatureStore
from specforge.runtime.data_plane.feature_store import FeatureStore
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.launch import (
    build_disagg_eagle3_runtime,
    build_offline_eagle3_runtime,
)

RUN_ID = "eagle3-disagg"


def _role() -> str:
    role = os.environ.get("DISAGG_ROLE")
    if role:
        return role
    return "producer" if os.environ.get("RCLI_NODE_RANK", "0") == "0" else "consumer"


def _backend() -> str:
    return os.environ.get("DISAGG_BACKEND", "shared_dir")


def _store(args, *, retain_on_release: bool = False) -> FeatureStore:
    token = os.environ.get("DISAGG_AUTH_TOKEN") or None
    store_id = os.environ.get("DISAGG_STORE_ID", RUN_ID)
    if _backend() == "mooncake":
        # Fast path: producer put()s and consumer get()s by key across nodes over
        # the Mooncake object store -- no shared *data* mount. store_id namespaces
        # the keys, so producer and consumer must agree on it (as with shared_dir).
        return MooncakeFeatureStore(
            store_id=store_id,
            setup_kwargs={
                "local_hostname": os.environ["MOONCAKE_LOCAL_HOSTNAME"],
                "metadata_server": os.environ["MOONCAKE_METADATA_SERVER"],
                "master_server_addr": os.environ["MOONCAKE_MASTER_SERVER_ADDR"],
                "protocol": os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
            },
            auth=AuthPolicy(token),
            credential=token,
            retain_on_release=retain_on_release,
        )
    return SharedDirFeatureStore(
        os.environ["DISAGG_STORE_ROOT"],
        store_id=store_id,
        auth=AuthPolicy(token),
        credential=token,
        retain_on_release=retain_on_release,
    )


def _hold_producer_until_consumed(manifest: str) -> None:
    """Keep the producer (and its Mooncake memory segment) alive until the
    consumer signals completion, since a Mooncake object lives in the producing
    process's segment. shared_dir does not need this (files persist on the mount).
    """
    consumed = manifest + ".consumed"
    hold_s = float(os.environ.get("DISAGG_PRODUCER_HOLD_S", "3600"))
    deadline = time.monotonic() + hold_s
    print(
        f"[producer] mooncake backend: holding segment until {consumed} "
        f"(<= {hold_s:.0f}s)",
        flush=True,
    )
    while not os.path.exists(consumed):
        if time.monotonic() > deadline:
            print(
                "[producer] hold timed out before consumer signalled; exiting "
                "(consumer may lose features)",
                flush=True,
            )
            return
        time.sleep(2)
    print("[producer] consumer signalled done; releasing segment", flush=True)


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
    location = getattr(store, "root", f"mooncake://{store.store_id}")
    print(
        f"[producer] ingested {len(refs)} samples into {location}; "
        f"manifest -> {manifest}",
        flush=True,
    )
    if _backend() == "mooncake":
        _hold_producer_until_consumed(manifest)


def _build_model_and_optimizer(args):
    """Identical EAGLE3 model/optimizer build for both consumer and colocated.

    Sharing this keeps the disaggregated and colocated runs apples-to-apples
    (same draft, same target_head, same optimizer) so any metric difference can
    only come from the feature transport, not the model.
    """
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

    return eagle3_model, target_head, optimizer_factory


def _log_interval() -> int:
    return int(os.environ.get("DISAGG_LOG_INTERVAL", "25"))


def run_colocated(args) -> None:
    """Baseline: same model + assembly via build_offline (LocalFeatureStore).

    For a head-to-head accept-length/loss comparison against the disaggregated
    consumer on identical features/seed.
    """
    init_distributed(
        timeout=args.dist_timeout,
        tp_size=args.tp_size,
        sp_ring_size=args.sp_ring_size,
        sp_ulysses_size=args.sp_ulysses_size,
    )
    sanity_check(args)
    eagle3_model, target_head, optimizer_factory = _build_model_and_optimizer(args)
    print(f"[colocated] training from {args.train_hidden_states_path}", flush=True)
    trainer, loader = build_offline_eagle3_runtime(
        hidden_states_path=args.train_hidden_states_path,
        eagle3_model=eagle3_model,
        target_head=target_head,
        optimizer_factory=optimizer_factory,
        run_id="eagle3-colocated",
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
        log_interval=_log_interval(),
    )
    trainer.fit(loader)
    destroy_distributed()


def run_consumer(args) -> None:
    manifest = os.environ["DISAGG_MANIFEST"]
    init_distributed(
        timeout=args.dist_timeout,
        tp_size=args.tp_size,
        sp_ring_size=args.sp_ring_size,
        sp_ulysses_size=args.sp_ulysses_size,
    )
    sanity_check(
        args
    )  # derives target_batch_size/dp_size the builders read (needs dist)
    # wait for the producer to publish the manifest (shared mount)
    deadline = time.monotonic() + 1800
    while not os.path.exists(manifest + ".done"):
        if time.monotonic() > deadline:
            raise SystemExit(f"[consumer] timed out waiting for {manifest}.done")
        time.sleep(2)

    eagle3_model, target_head, optimizer_factory = _build_model_and_optimizer(args)

    # offline ref set is re-iterated across epochs -> retain on release (read-only)
    store = _store(args, retain_on_release=True)
    refs = read_ref_manifest(manifest)
    location = getattr(store, "root", f"mooncake://{store.store_id}")
    print(
        f"[consumer] training from {len(refs)} disagg refs in {location}", flush=True
    )

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
        log_interval=_log_interval(),
    )
    trainer.fit(loader)
    destroy_distributed()
    if _backend() == "mooncake":
        # release the producer holding its Mooncake segment open (see docstring)
        open(manifest + ".consumed", "w").close()


def main() -> None:
    parser, args = parse_args()
    set_seed(args.seed)
    if args.train_hidden_states_path is None:
        raise SystemExit(
            "disagg example wires the OFFLINE path; pass --train-hidden-states-path"
        )
    role = _role()
    print(
        f"[disagg] role={role} node_rank={os.environ.get('RCLI_NODE_RANK', '0')}",
        flush=True,
    )
    if role == "producer":
        run_producer(args)
    elif role == "colocated":
        run_colocated(args)  # baseline for head-to-head comparison
    else:
        run_consumer(args)


if __name__ == "__main__":
    main()
