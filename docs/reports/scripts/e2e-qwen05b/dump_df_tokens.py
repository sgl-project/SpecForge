"""Dump MAIN's dflash dataloader tokens to jsonl (input_ids + loss_mask per row).

Run inside the MAIN tree scripts/ dir. Env: EXP_TOKENS_OUT.
"""

import json
import os

from accelerate.utils import set_seed
from train_dflash import build_dataloader, parse_args
from transformers import AutoTokenizer

from specforge.distributed import destroy_distributed, init_distributed

OUT = os.environ.get("EXP_TOKENS_OUT", "/root/exp/data/df_tokens.jsonl")


def main():
    args = parse_args()
    set_seed(args.seed)
    init_distributed(timeout=args.dist_timeout, tp_size=args.tp_size)
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_path)
    train_dataloader, _eval = build_dataloader(args, tokenizer)
    n = 0
    with open(OUT, "w") as f:
        for batch in train_dataloader:
            ids = batch["input_ids"]
            lm = batch["loss_mask"]
            attn = batch.get("attention_mask")
            for i in range(ids.shape[0]):
                k = int(attn[i].sum().item()) if attn is not None else ids.shape[1]
                f.write(
                    json.dumps(
                        {
                            "input_ids": ids[i, :k].tolist(),
                            "loss_mask": lm[i, :k].tolist(),
                        }
                    )
                    + "\n"
                )
                n += 1
    print(f"[df-tokens] wrote {n} rows -> {OUT}", flush=True)
    destroy_distributed()


if __name__ == "__main__":
    main()
