#!/bin/bash
# TRUE disaggregated placement test: inference (producer) and trainer (consumer)
# as SEPARATE PROCESSES on DIFFERENT GPUs, sharing only a directory store, a
# ref-channel file, and a SQLite metadata store. Consumer curves land in
# $LOGS/ours-<fam>-2proc.log for comparison against the colocated runs.
set -uxo pipefail
export FLASHINFER_DISABLE_VERSION_CHECK=1
EXP=/root/exp
LOGS=$EXP/logs
UNTIED=/root/exp/models/qwen2.5-0.5b-untied
S=$EXP/store2p
rm -rf $S; mkdir -p $S/e3-online-disagg2p $S/df-online-disagg2p $S/e3-offline-disagg2p $S/df-offline-disagg2p
touch $S/e3-online-disagg2p/refs.jsonl $S/df-online-disagg2p/refs.jsonl
rm -f $LOGS/status2p.txt $LOGS/*-2proc*.log

E3_ON="--target-model-path Qwen/Qwen2.5-0.5B-Instruct \
  --draft-model-config $EXP/configs/qwen2.5-0.5b-eagle3.json \
  --train-data-path $EXP/data/sharegpt_final.jsonl \
  --chat-template qwen --max-length 512 --batch-size 1 \
  --learning-rate 1e-4 --ttt-length 7 --seed 0 --num-epochs 1 \
  --max-num-steps 100 --total-steps 10000 --target-model-backend hf"
E3_OFF="$E3_ON --train-hidden-states-path $EXP/dumps/e3-hs"
E3_OFF=${E3_OFF/--target-model-path Qwen\/Qwen2.5-0.5B-Instruct/--target-model-path $UNTIED}

DF_ARGS="--target-model-path Qwen/Qwen2.5-0.5B-Instruct \
  --draft-config-path $EXP/configs/qwen2.5-0.5b-dflash.json \
  --train-data-path $EXP/data/sharegpt_final.jsonl \
  --chat-template qwen --max-length 512 --batch-size 1 \
  --learning-rate 6e-4 --seed 0 --num-epochs 1 \
  --mask-token-id 151669 --target-model-backend hf --log-interval 1 \
  --save-interval 1000000"

go () { # logname gpu port env_str script args...
  local name=$1 gpu=$2 port=$3 env_str=$4 script=$5; shift 5
  ( cd $EXP/sf-ours/scripts && env $env_str CUDA_VISIBLE_DEVICES=$gpu \
      timeout 2400 torchrun --rdzv-backend c10d --rdzv-endpoint localhost:$port \
      --nnodes 1 --nproc_per_node 1 $script "$@" \
      > $LOGS/$name.log 2>&1 ; echo "EXIT=$? $name" >> $LOGS/status2p.txt ) &
}

# --- eagle3 ONLINE 2proc: producer GPU0 || consumer GPU1 (concurrent) ---
go e3-online-2proc-producer 0 29601 \
  "PYTHONPATH=$EXP/sf-ours EXP_TOPO=disagg2p EXP_ROLE=producer EXP_STORE_ROOT=$S EXP_DB=$S/e3-online.db EXP_MAX_PROMPTS=130 EXP_PROMPTS_FROM=dump:$EXP/dumps/e3-hs" \
  exp_eagle3_dataflow.py $E3_ON --output-dir $EXP/out/e3-2p-prod
go ours-e3-online-2proc 1 29602 \
  "PYTHONPATH=$EXP/sf-ours EXP_TOPO=disagg2p EXP_ROLE=consumer EXP_STORE_ROOT=$S EXP_DB=$S/e3-online.db" \
  exp_eagle3_dataflow.py $E3_ON --output-dir $EXP/out/e3-2p-cons

# --- dflash ONLINE 2proc: producer GPU2 || consumer GPU3 (concurrent) ---
go df-online-2proc-producer 2 29603 \
  "PYTHONPATH=$EXP/sf-ours EXP_MODE=online EXP_TOPO=disagg2p EXP_ROLE=producer EXP_STORE_ROOT=$S EXP_DB=$S/df-online.db EXP_MAX_PROMPTS=130 EXP_MAX_STEPS=100 EXP_TOTAL_STEPS=1000 EXP_PROMPTS_FROM=jsonl:$EXP/data/df_tokens.jsonl" \
  exp_dflash_dataflow.py $DF_ARGS --output-dir $EXP/out/df-2p-prod
go ours-df-online-2proc 3 29604 \
  "PYTHONPATH=$EXP/sf-ours EXP_MODE=online EXP_TOPO=disagg2p EXP_ROLE=consumer EXP_STORE_ROOT=$S EXP_DB=$S/df-online.db EXP_MAX_STEPS=100 EXP_TOTAL_STEPS=1000 EXP_PROMPTS_FROM=jsonl:$EXP/data/df_tokens.jsonl" \
  exp_dflash_dataflow.py $DF_ARGS --output-dir $EXP/out/df-2p-cons

# --- eagle3 OFFLINE 2proc: producer then consumer (separate processes, GPU4) ---
( cd $EXP/sf-ours/scripts && \
  env PYTHONPATH=$EXP/sf-ours EXP_TOPO=disagg2p EXP_ROLE=producer EXP_STORE_ROOT=$S CUDA_VISIBLE_DEVICES=4 \
    timeout 2400 torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29605 --nnodes 1 --nproc_per_node 1 \
    exp_eagle3_dataflow.py $E3_OFF --output-dir $EXP/out/e3off-2p-prod > $LOGS/e3-offline-2proc-producer.log 2>&1
  echo "EXIT=$? e3-offline-2proc-producer" >> $LOGS/status2p.txt
  env PYTHONPATH=$EXP/sf-ours EXP_TOPO=disagg2p EXP_ROLE=consumer EXP_STORE_ROOT=$S CUDA_VISIBLE_DEVICES=4 \
    timeout 2400 torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29606 --nnodes 1 --nproc_per_node 1 \
    exp_eagle3_dataflow.py $E3_OFF --output-dir $EXP/out/e3off-2p-cons > $LOGS/ours-e3-offline-2proc.log 2>&1
  echo "EXIT=$? ours-e3-offline-2proc" >> $LOGS/status2p.txt ) &

# --- dflash OFFLINE 2proc: producer then consumer (separate processes, GPU5) ---
( cd $EXP/sf-ours/scripts && \
  env PYTHONPATH=$EXP/sf-ours EXP_MODE=offline EXP_TOPO=disagg2p EXP_ROLE=producer EXP_STORE_ROOT=$S EXP_HS_PATH=$EXP/dumps/dflash-hs EXP_MAX_STEPS=100 EXP_TOTAL_STEPS=1000 EXP_PROMPTS_FROM=jsonl:$EXP/data/df_tokens.jsonl CUDA_VISIBLE_DEVICES=5 \
    timeout 2400 torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29607 --nnodes 1 --nproc_per_node 1 \
    exp_dflash_dataflow.py $DF_ARGS --output-dir $EXP/out/dfoff-2p-prod > $LOGS/df-offline-2proc-producer.log 2>&1
  echo "EXIT=$? df-offline-2proc-producer" >> $LOGS/status2p.txt
  env PYTHONPATH=$EXP/sf-ours EXP_MODE=offline EXP_TOPO=disagg2p EXP_ROLE=consumer EXP_STORE_ROOT=$S EXP_HS_PATH=$EXP/dumps/dflash-hs EXP_MAX_STEPS=100 EXP_TOTAL_STEPS=1000 EXP_PROMPTS_FROM=jsonl:$EXP/data/df_tokens.jsonl CUDA_VISIBLE_DEVICES=5 \
    timeout 2400 torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29608 --nnodes 1 --nproc_per_node 1 \
    exp_dflash_dataflow.py $DF_ARGS --output-dir $EXP/out/dfoff-2p-cons > $LOGS/ours-df-offline-2proc.log 2>&1
  echo "EXIT=$? ours-df-offline-2proc" >> $LOGS/status2p.txt ) &

wait
echo "2PROC-DONE" >> $LOGS/status2p.txt
cat $LOGS/status2p.txt
