#!/usr/bin/env python3
"""Trim a preformatted DSV4 corpus to a token budget at TURN granularity.

For each row, the rendered text is tokenized once (fast tokenizer, offset
mapping) and cut at the last complete assistant turn — the last
``<｜end▁of▁sentence｜>`` — whose final token index is within --max-tokens.
Rows with no complete assistant turn inside the budget are dropped.

Why not let SpecForge's max_length truncation handle it: mid-turn truncation
supervises half a DSML tool call and strips the EOS off the final turn;
turn-boundary trimming keeps every supervised span complete. Token counts are
measured with the real tokenizer, never estimated from characters (specforge
skill, trap 10).

Output rows add: tokens (post-trim), turns_kept, turns_total, trimmed.
"""
import argparse
import bisect
import collections
import json
import os
import re
import sys
from multiprocessing import Pool

TOK_DIR = os.environ.get(
    "SF_TOKENIZER", "/personal/SpecForge/exports/deepseek-v4-flash-dspark-step1885"
)
EOS = "<｜end▁of▁sentence｜>"

_tok = None
def tok():
    global _tok
    if _tok is None:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True)
    return _tok


def trim_row(args):
    line, max_tokens = args
    r = json.loads(line)
    text = r["text"]
    eos_ends = [m.end() for m in re.finditer(re.escape(EOS), text)]
    enc = tok()(text, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    n = len(offs)
    r["turns_total"] = len(eos_ends)
    if n <= max_tokens:
        r["tokens"] = n
        r["turns_kept"] = len(eos_ends)
        r["trimmed"] = False
        return json.dumps(r, ensure_ascii=False), "kept"
    starts = [a for a, _ in offs]
    kept = None
    for i, ce in enumerate(eos_ends):
        # first token starting at/after the EOS end == tokens consumed so far
        ntok = bisect.bisect_left(starts, ce)
        if ntok <= max_tokens:
            kept = (i + 1, ce, ntok)
        else:
            break
    if kept is None:
        return None, "dropped_no_turn_fits"
    turns, cut_char, ntok = kept
    r["text"] = text[:cut_char]
    r["tokens"] = ntok
    r["turns_kept"] = turns
    r["trimmed"] = True
    return json.dumps(r, ensure_ascii=False), "trimmed"


def line_offsets(path):
    """Byte offset of each line; reuses/creates the `.offsets` cache that
    scripts/view_dataset.py also maintains."""
    import array
    cp = path + ".offsets"
    if os.path.exists(cp) and os.path.getmtime(cp) >= os.path.getmtime(path):
        a = array.array("q")
        with open(cp, "rb") as f:
            a.frombytes(f.read())
        return a.tolist()
    offs, pos = [], 0
    with open(path, "rb") as f:
        for line in f:
            offs.append(pos)
            pos += len(line)
    with open(cp, "wb") as f:
        f.write(array.array("q", offs).tobytes())
    return offs


def process_chunk(job):
    src, start_byte, count, max_tokens, out_path = job
    st = collections.Counter()
    with open(src, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        f.seek(start_byte)
        for _ in range(count):
            line = f.readline()
            if not line:
                break
            try:
                blob, status = trim_row((line, max_tokens))
            except Exception as e:
                st[f"err_{type(e).__name__}"] += 1
                continue
            st[status] += 1
            if blob is not None:
                try:
                    out.write(blob + "\n")
                except UnicodeEncodeError:
                    out.write(json.dumps(json.loads(blob), ensure_ascii=True) + "\n")
                    st["surrogate_escaped"] += 1
    return dict(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    offs = line_offsets(args.src)
    total = len(offs)
    per = (total + args.workers - 1) // args.workers
    parts_dir = args.out + ".parts"
    os.makedirs(parts_dir, exist_ok=True)
    jobs = [
        (args.src, offs[w * per], per, args.max_tokens,
         os.path.join(parts_dir, f"part-{w:03d}.jsonl"))
        for w in range(args.workers) if w * per < total
    ]
    print(f"{total} rows, {len(jobs)} workers", flush=True)
    with Pool(len(jobs)) as p:
        stats = p.map(process_chunk, jobs)
    agg = collections.Counter()
    for s in stats:
        agg.update(s)
    with open(args.out, "w", encoding="utf-8") as fout:
        for job in jobs:
            with open(job[4], encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
            os.remove(job[4])
    os.rmdir(parts_dir)
    for k in sorted(agg):
        print(f"  {k} = {agg[k]}")
    out_rows = agg["kept"] + agg["trimmed"]
    print(f"DONE {out_rows}/{total} rows -> {args.out}")


if __name__ == "__main__":
    main()
