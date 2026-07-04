#!/bin/bash
# Single-node setup for the Qwen3.6-27B DFlash disagg run. Code comes from the
# fork (NFS /workspace quota is exhausted), everything else is container-local.
set -euxo pipefail
E=/root/exp36
mkdir -p $E/logs $E/out $E/store36 $E/data
cd $E

# code from the merged fork branch (PR #593 on PR #645 base)
if [ ! -d $E/sf ]; then
  git clone --depth 1 -b qwen36-dflash-on-645 https://github.com/maocheng23/SpecForge.git $E/sf
fi

# deps the sglang image lacks (--no-deps keeps the image's torch pin)
python -m pip install --break-system-packages --no-deps accelerate yunchang qwen-vl-utils 2>&1 | tail -1 || true

# per-step CURVE prints + EXP_MAX_STEPS/EXP_TOTAL_STEPS knobs in train_dflash
python $E/sf/docs/reports/scripts/e2e-qwen05b/patch_main.py $E/sf 2>&1 | tail -2 || echo "patch skipped"

# target weights (~54GB, container-local)
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.6-27B")
print("model ready")
EOF

# Nemotron-v2 data (deterministic; sample-capped for a bounded demo run)
if [ ! -f $E/data/nemotron_v2_train.jsonl ]; then
  cd $E/sf
  python scripts/prepare_nemotron_post_training_v2.py \
    --sample-size 4000 --eval-ratio 0.01 --seed 42 \
    --train-output-path $E/data/nemotron_v2_train.jsonl \
    --eval-output-path $E/data/nemotron_v2_eval.jsonl
fi
wc -l $E/data/nemotron_v2_train.jsonl
echo SETUP36-1N-DONE

# make the DataFlow exp launcher importable next to train_dflash
cp /root/exp36/sf/docs/reports/scripts/e2e-qwen05b/exp_dflash_dataflow.py /root/exp36/sf/scripts/
cp /root/exp36/sf/docs/reports/scripts/e2e-qwen05b/parse_curves.py /root/exp36/sf/scripts/
echo LAUNCHER-COPIED
