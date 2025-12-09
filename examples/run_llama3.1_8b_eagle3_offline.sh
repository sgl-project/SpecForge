sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount_base.sh ofs-llab-volume /nfs/ofs-llab-volume b1e837d84f284c3392c4e16065fad32e nmgllab
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh ofs-fengyu /nfs/ofs-fengyu 2ec0280f2f304e19bf2397eb9facf787 nmgpu
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh ofs-alohajiang /nfs/ofs-alohajiang 56bfb0a2b4454b4cb5d305a8507dc1e0 hnapu
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh ofs-llm-data /nfs/ofs-llm-ssd  b00b083f426245d1a6abbc2f0164124a nmgmlmodeltrain
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh ofs-llab-cold /nfs/ofs-llab-cold a7ce0339d04d4b008040e3a28e3fc44e nmgllab
WORK_PATH="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1
export PATH="/home/luban/miniconda3/bin:$PATH"
pip install accelerate rich yunchang --trusted-host didiyum.sys.xiaojukeji.com -i http://didiyum.sys.xiaojukeji.com/didiyum/pip/simple/
pip install sglang[all]==0.5.5  --trusted-host didiyum.sys.xiaojukeji.com -i http://didiyum.sys.xiaojukeji.com/didiyum/pip/simple/
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname $SCRIPT_DIR)
echo ROOT_DIR:$ROOT_DIR
cd $ROOT_DIR
NUM_GPUS=8
TP_SIZE=1
BUILD_DATASET_NUM_PROC=${BUILD_DATASET_NUM_PROC:-64}
export PYTHONPATH=$ROOT_DIR:$PYTHONPATH
export TMPDIR=/tmp-data
sudo chmod -R 777 /tmp-data
# generate hidden states
#torchrun \
#    --standalone \
#    --nproc_per_node $NUM_GPUS \
#    scripts/prepare_hidden_states.py \
#    --target-model-path  /nfs/volume-1615-2/models/Qwen3-8B \
#    --enable-aux-hidden-states \
#    --data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
#    --output-path $ROOT_DIR/cache/hidden_states/Qwen3-8B \
#    --chat-template qwen \
#    --max-length 4096 \
#    --tp-size $TP_SIZE \
#    --batch-size 32 \
#    --mem-fraction-static 0.8\

# train eagle3 offline
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3.py \
    --target-model-path  /nfs/volume-1615-2/models/Qwen3-8B \
    --draft-model-config $ROOT_DIR/configs/qwen3-8b-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt_train.jsonl \
    --train-hidden-states-path $ROOT_DIR/cache/hidden_states/Qwen3-8B \
    --build-dataset-num-proc $BUILD_DATASET_NUM_PROC \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-eagle3-sharegpt_ulysses \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen \
    --cache-dir $ROOT_DIR/cache \
    --embedding-key model.embed_tokens.weight \
    --tp-size $TP_SIZE \
    --target-model-backend sglang \
    --total-steps 10000 \
    --log-interval 5 \
    --report tensorboard \
    --attention-backend usp \
    --sp-ulysses-size 4 \


