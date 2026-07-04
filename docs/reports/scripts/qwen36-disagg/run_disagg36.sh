#!/bin/bash
# Qwen3.6-27B DFlash, DISAGGREGATED on one 8-GPU node:
#   producer (27B target inference)  -> GPUs 0-3
#   consumer (DFlash draft trainer)  -> GPUs 4-7
# sharing a SharedDirFeatureStore + StreamingRefChannel + SQLite metadata store.
# Producer and consumer are separate processes launched concurrently.
set -uxo pipefail
E=/root/exp36
LOGS=$E/logs
S=$E/store36
RUN=df-online-disagg2p        # store_id the launcher derives from run_id
rm -rf $S; mkdir -p $S/$RUN; touch $S/$RUN/refs.jsonl
rm -f $LOGS/qwen36-*.log $LOGS/status36.txt

export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCHINDUCTOR_CACHE_DIR=$E/cache/compiled_kernels

# recipe args (PR #593), max-length trimmed to 2048 + prompt/step caps for a
# bounded demo run; everything else matches examples/run_qwen3.6_27b_dflash_online.sh
ARGS="--target-model-path Qwen/Qwen3.6-27B \
  --target-model-backend hf --trust-remote-code \
  --draft-config-path $E/sf/configs/qwen3.6-27b-dflash.json \
  --embedding-key model.language_model.embed_tokens.weight \
  --lm-head-key lm_head.weight --mask-token-id 248070 \
  --train-data-path $E/data/nemotron_v2_train.jsonl \
  --chat-template qwen3.5 --max-length 2048 --batch-size 1 \
  --learning-rate 6e-4 --warmup-ratio 0.04 --max-grad-norm 1.0 \
  --block-size 16 --num-anchors 512 --loss-decay-gamma 7.0 \
  --attention-backend flex_attention --seed 42 \
  --log-interval 1 --save-interval 1000000"

cd $E/sf/scripts

# --- producer: inference pool, GPUs 0-3 ---
( env PYTHONPATH=$E/sf CUDA_VISIBLE_DEVICES=0,1,2,3 \
    EXP_MODE=online EXP_TOPO=disagg2p EXP_ROLE=producer \
    EXP_STORE_ROOT=$S EXP_DB=$S/$RUN.db EXP_MAX_PROMPTS=300 \
    torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29701 --nnodes 1 --nproc_per_node 1 \
    exp_dflash_dataflow.py $ARGS --output-dir $E/out/qwen36-prod \
    > $LOGS/qwen36-producer.log 2>&1 ; echo "EXIT=$? producer" >> $LOGS/status36.txt ) &

# --- consumer: trainer pool, GPUs 4-7 ---
( env PYTHONPATH=$E/sf CUDA_VISIBLE_DEVICES=4,5,6,7 \
    EXP_MODE=online EXP_TOPO=disagg2p EXP_ROLE=consumer \
    EXP_STORE_ROOT=$S EXP_DB=$S/$RUN.db EXP_MAX_STEPS=150 EXP_TOTAL_STEPS=1000 \
    torchrun --rdzv-backend c10d --rdzv-endpoint localhost:29702 --nnodes 1 --nproc_per_node 1 \
    exp_dflash_dataflow.py $ARGS --output-dir $E/out/qwen36-cons \
    > $LOGS/qwen36-consumer.log 2>&1 ; echo "EXIT=$? consumer" >> $LOGS/status36.txt ) &

wait
echo "DISAGG36-DONE" >> $LOGS/status36.txt
cat $LOGS/status36.txt
