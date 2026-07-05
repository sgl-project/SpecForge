"""Write the final filtered jsonl: lines BOTH trees render identically & trainably."""

import json

OURS = json.load(open("/root/exp/data/stats_ours.json"))
MAIN = json.load(open("/root/exp/data/stats_main.json"))
SRC = "/root/exp/data/sharegpt_train.jsonl"
DST = "/root/exp/data/sharegpt_filtered.jsonl"
KEEP = 150

good = []
for li, (n, m) in sorted(((int(k), v) for k, v in OURS.items())):
    other = MAIN.get(str(li))
    if other is None:
        continue
    if n > 16 and m >= 8 and other[0] == n and other[1] == m:
        good.append(li)
    if len(good) >= KEEP:
        break

lines = open(SRC).readlines()
with open(DST, "w") as f:
    for li in good:
        f.write(lines[li])
print(f"[intersect] kept {len(good)} identically-rendered lines -> {DST}")
