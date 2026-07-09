"""Disaggregated ONLINE DFlash over patched SGLang server-capture."""

import os
import sys

import torch

if not torch.cuda.is_available():
    # CPU producer: keep yunchang/flashinfer CUDA probes on the clean fallback.
    sys.modules["flashinfer"] = None

from _dflash_family_disagg import (
    build_consumer_parts,
    disagg_role,
    fit_online_consumer,
    run_server_capture_producer,
)
from accelerate.utils import set_seed
from train_dflash import parse_args

from specforge.core.dflash import OnlineDFlashModel
from specforge.distributed import destroy_distributed, init_distributed

STRATEGY = "dflash"
RUN_ID = os.environ.get("DISAGG_STORE_ID", "qwen36-dflash-disagg")


def run_producer(args) -> None:
    run_server_capture_producer(STRATEGY, args, RUN_ID)


def run_consumer(args) -> None:
    parts = build_consumer_parts(args)
    dflash_model = OnlineDFlashModel(
        draft_model=parts.draft_model,
        target_lm_head=parts.target_components.lm_head,
        target_embed_tokens=parts.target_components.embed_tokens,
        mask_token_id=parts.mask_token_id,
        block_size=parts.draft_model.block_size,
        attention_backend=args.attention_backend,
        num_anchors=args.num_anchors,
        loss_decay_gamma=args.loss_decay_gamma,
        loss_type=args.loss_type,
        dpace_alpha=args.dpace_alpha,
    )
    fit_online_consumer(STRATEGY, args, RUN_ID, dflash_model)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    role = disagg_role()
    print(f"[disagg-dflash] role={role} run_id={RUN_ID}", flush=True)
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
