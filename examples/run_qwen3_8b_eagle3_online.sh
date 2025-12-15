#!/bin/bash
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount_base.sh ofs-llab-volume /nfs/ofs-llab-volume b1e837d84f284c3392c4e16065fad32e nmgllab
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh ofs-fengyu /nfs/ofs-fengyu 2ec0280f2f304e19bf2397eb9facf787 nmgpu
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh ofs-alohajiang /nfs/ofs-alohajiang 56bfb0a2b4454b4cb5d305a8507dc1e0 hnapu
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh ofs-llm-data /nfs/ofs-llm-ssd  b00b083f426245d1a6abbc2f0164124a nmgmlmodeltrain
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh ofs-llab-cold /nfs/ofs-llab-cold a7ce0339d04d4b008040e3a28e3fc44e nmgllab
WORK_PATH="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1

export PATH="/home/luban/miniconda3/bin:$PATH"
export PYTHONPATH=$WORK_PATH:$PYTHONPATH
export TMPDIR=/tmp-data
sudo chmod -R 777 /tmp-data

pip install accelerate rich yunchang --trusted-host didiyum.sys.xiaojukeji.com -i http://didiyum.sys.xiaojukeji.com/didiyum/pip/simple/
pip install sglang[all]==0.5.5  --trusted-host didiyum.sys.xiaojukeji.com -i http://didiyum.sys.xiaojukeji.com/didiyum/pip/simple/
#sed -i.bak '644s/log_info_on_rank0(logger, "Chunked prefix cache is turned on.")/pass/' /home/luban/miniconda3/lib/python3.12/site-packages/sglang/srt/model_executor/model_runner.py
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels

# support tp8 train eagle3 for Qwen3-4B/8B/32B up to tp_size = 8
NUM_GPUS=${1:-1}
TP_SIZE=${2:-1}
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}

torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path /nfs/volume-1615-2/models/Qwen3-8B \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-eagle3-sharegpt \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --target-model-backend sglang
