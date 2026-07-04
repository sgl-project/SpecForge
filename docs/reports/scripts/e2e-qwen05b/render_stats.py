"""Per-jsonl-line render stats for THIS tree's eagle3 data pipeline.

Maps dataset rows back to jsonl lines by invoking this tree's own
``safe_conversations_generator`` per line (it silently skips malformed lines,
so row index != line index), then records (token_len, mask_sum) per line from
the same dataloader the training scripts build.

Env: EXP_STATS_OUT (json path), EXP_SRC_JSONL. Run inside <tree>/scripts.
"""

import json
import os
import tempfile

from accelerate.utils import set_seed
from train_eagle3 import build_dataloaders, build_draft_model, parse_args

import specforge.utils as sf_utils
from specforge.distributed import destroy_distributed, init_distributed

SRC = os.environ.get("EXP_SRC_JSONL", "/root/exp/data/sharegpt_train.jsonl")
OUT = os.environ["EXP_STATS_OUT"]


def line_mapping(src):
    """dataset_row -> jsonl line index, via this tree's own generator."""
    mapping = []
    fd, tmp = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    try:
        with open(src) as f:
            for li, line in enumerate(f):
                if not line.strip():
                    continue
                with open(tmp, "w") as t:
                    t.write(line)
                if len(list(sf_utils.safe_conversations_generator(tmp))) == 1:
                    mapping.append(li)
    finally:
        os.remove(tmp)
    return mapping


def main():
    parser, args = parse_args()
    args.target_batch_size = args.tp_size * args.batch_size
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)

    mapping = line_mapping(SRC)
    print(f"[stats] generator keeps {len(mapping)} lines", flush=True)

    draft_config, _d, _c, _r = build_draft_model(args)
    train_dataloader, _v, _e = build_dataloaders(args, draft_config)

    stats = {}
    for i, batch in enumerate(train_dataloader):
        if i >= len(mapping):
            break
        n = int(batch["attention_mask"][0].sum().item())
        mask_sum = int(batch["loss_mask"][0][:n].sum().item())
        stats[mapping[i]] = [n, mask_sum]
    with open(OUT, "w") as f:
        json.dump(stats, f)
    print(f"[stats] wrote {len(stats)} rows -> {OUT}", flush=True)
    destroy_distributed()


if __name__ == "__main__":
    main()
