#!/bin/bash
set -eux

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# train eagle3 for gemma3-1b-it
NUM_GPUS=${1:-8}
TP_SIZE=${2:-4}

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=1800000
export NCCL_BUFFSIZE=2097152
export NCCL_DEBUG=INFO
export GLOO_SOCKET_IFNAME=eth0

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /opt/ml/model/eagle3/base_models/film-pii-reduction-gemma-1b-gptq-v0.1 \
    --draft-model-config $ROOT_DIR/configs/gemma3-1b-eagle3.json \
    --train-data-path /opt/ml/model/eagle3/data/film_pii/film_pii_9_gemma_1B_gptq/merged_converted.jsonl \
    --is-preformatted \
    --build-dataset-num-proc 40 \
    --output-dir /opt/ml/model/eagle3/trained_models/film_pii/film-pii-reduction-gemma-1b-gptq-v0.1 \
    --num-epochs 2 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template gemma \
    --tp-size 1 \
    --attention-backend flex_attention \
    --target-model-backend hf \
    --log-interval 10 \
    --cache-dir $ROOT_DIR/cache \
    --dist-timeout 120
