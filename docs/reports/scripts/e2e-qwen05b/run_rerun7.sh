#!/bin/bash
# Targeted rerun: ours consumes MAIN-rendered tokens everywhere.
#  - dump main's dflash tokens -> df_tokens.jsonl
#  - regen dflash-hs dump from those tokens (ours capture)
#  - rerun ours-e3-online x2 (prompts from e3-hs dump) + all 5 dflash runs
# Keeps round-5 logs for: main-e3-online, main-e3-offline, ours-e3-offline-*.
set -uxo pipefail
export FLASHINFER_DISABLE_VERSION_CHECK=1
EXP=/root/exp
LOGS=$EXP/logs

E3_ARGS="--target-model-path Qwen/Qwen2.5-0.5B-Instruct \
  --draft-model-config $EXP/configs/qwen2.5-0.5b-eagle3.json \
  --train-data-path $EXP/data/sharegpt_final.jsonl \
  --chat-template qwen --max-length 512 --batch-size 1 \
  --learning-rate 1e-4 --ttt-length 7 --seed 0 --num-epochs 1 \
  --max-num-steps 100 --total-steps 10000 --target-model-backend hf"

DF_ARGS="--target-model-path Qwen/Qwen2.5-0.5B-Instruct \
  --draft-config-path $EXP/configs/qwen2.5-0.5b-dflash.json \
  --train-data-path $EXP/data/sharegpt_final.jsonl \
  --chat-template qwen --max-length 512 --batch-size 1 \
  --learning-rate 6e-4 --seed 0 --num-epochs 1 \
  --mask-token-id 151669 --target-model-backend hf --log-interval 1 \
  --save-interval 1000000"

rm -f $LOGS/status.txt
rm -rf $EXP/store/* $EXP/dumps/dflash-hs
rm -f $LOGS/ours-e3-online-*.log $LOGS/*df*.log

# main's dflash token render (CPU-ish, GPU0)
cd $EXP/sf-main/scripts
EXP_TOKENS_OUT=$EXP/data/df_tokens.jsonl PYTHONPATH=$EXP/sf-main \
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29440 \
  --nnodes 1 --nproc_per_node 1 dump_df_tokens.py $DF_ARGS \
  --output-dir $EXP/out/df-tokens > $LOGS/df-tokens.log 2>&1
wc -l $EXP/data/df_tokens.jsonl

# dflash offline features from main-rendered tokens (ours capture)
cd $EXP/sf-ours/scripts
EXP_MODE=dump EXP_HS_PATH=$EXP/dumps/dflash-hs EXP_MAX_PROMPTS=130 \
EXP_PROMPTS_FROM=jsonl:$EXP/data/df_tokens.jsonl PYTHONPATH=$EXP/sf-ours \
CUDA_VISIBLE_DEVICES=0 torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29441 \
  --nnodes 1 --nproc_per_node 1 exp_dflash_dataflow.py $DF_ARGS \
  --output-dir $EXP/out/df-dump > $LOGS/dump-df.log 2>&1
ls $EXP/dumps/dflash-hs | wc -l

tr_run () { # name gpu port workdir extra_env script args...
  local name=$1 gpu=$2 port=$3 wd=$4 env_str=$5 script=$6; shift 6
  ( cd $wd && env $env_str CUDA_VISIBLE_DEVICES=$gpu \
      torchrun --rdzv-backend c10d --rdzv-endpoint localhost:$port \
      --nnodes 1 --nproc_per_node 1 $script "$@" \
      > $LOGS/$name.log 2>&1 ; echo "EXIT=$? $name" >> $LOGS/status.txt ) &
}

# ours eagle3 online, prompts = main-rendered dump tokens
tr_run ours-e3-online-colo 1 29501 $EXP/sf-ours/scripts \
  "PYTHONPATH=$EXP/sf-ours EXP_TOPO=colo EXP_MAX_PROMPTS=130 EXP_PROMPTS_FROM=dump:$EXP/dumps/e3-hs" exp_eagle3_dataflow.py \
  $E3_ARGS --output-dir $EXP/out/ours-e3-online-colo

tr_run ours-e3-online-disagg 2 29502 $EXP/sf-ours/scripts \
  "PYTHONPATH=$EXP/sf-ours EXP_TOPO=disagg EXP_MAX_PROMPTS=130 EXP_PROMPTS_FROM=dump:$EXP/dumps/e3-hs EXP_STORE_ROOT=$EXP/store" exp_eagle3_dataflow.py \
  $E3_ARGS --output-dir $EXP/out/ours-e3-online-disagg

# dflash: main baseline + ours x4, all on main-rendered tokens
tr_run main-df-online 0 29510 $EXP/sf-main \
  "PYTHONPATH=$EXP/sf-main EXP_MAX_STEPS=100 EXP_TOTAL_STEPS=1000" scripts/train_dflash.py \
  $DF_ARGS --output-dir $EXP/out/main-df-online

tr_run ours-df-online-colo 3 29511 $EXP/sf-ours/scripts \
  "PYTHONPATH=$EXP/sf-ours EXP_MODE=online EXP_TOPO=colo EXP_MAX_PROMPTS=130 EXP_MAX_STEPS=100 EXP_TOTAL_STEPS=1000 EXP_PROMPTS_FROM=jsonl:$EXP/data/df_tokens.jsonl" exp_dflash_dataflow.py \
  $DF_ARGS --output-dir $EXP/out/ours-df-online-colo

tr_run ours-df-online-disagg 4 29512 $EXP/sf-ours/scripts \
  "PYTHONPATH=$EXP/sf-ours EXP_MODE=online EXP_TOPO=disagg EXP_MAX_PROMPTS=130 EXP_MAX_STEPS=100 EXP_TOTAL_STEPS=1000 EXP_PROMPTS_FROM=jsonl:$EXP/data/df_tokens.jsonl EXP_STORE_ROOT=$EXP/store" exp_dflash_dataflow.py \
  $DF_ARGS --output-dir $EXP/out/ours-df-online-disagg

tr_run ours-df-offline-colo 5 29513 $EXP/sf-ours/scripts \
  "PYTHONPATH=$EXP/sf-ours EXP_MODE=offline EXP_TOPO=colo EXP_HS_PATH=$EXP/dumps/dflash-hs EXP_MAX_STEPS=100 EXP_TOTAL_STEPS=1000 EXP_PROMPTS_FROM=jsonl:$EXP/data/df_tokens.jsonl" exp_dflash_dataflow.py \
  $DF_ARGS --output-dir $EXP/out/ours-df-offline-colo

tr_run ours-df-offline-disagg 6 29514 $EXP/sf-ours/scripts \
  "PYTHONPATH=$EXP/sf-ours EXP_MODE=offline EXP_TOPO=disagg EXP_HS_PATH=$EXP/dumps/dflash-hs EXP_MAX_STEPS=100 EXP_TOTAL_STEPS=1000 EXP_PROMPTS_FROM=jsonl:$EXP/data/df_tokens.jsonl EXP_STORE_ROOT=$EXP/store" exp_dflash_dataflow.py \
  $DF_ARGS --output-dir $EXP/out/ours-df-offline-disagg

wait
echo "RERUN7-DONE" >> $LOGS/status.txt
cat $LOGS/status.txt
