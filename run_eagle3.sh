sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount_base.sh ofs-llab-volume /nfs/ofs-llab-volume b1e837d84f284c3392c4e16065fad32e nmgllab
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh ofs-fengyu /nfs/ofs-fengyu 2ec0280f2f304e19bf2397eb9facf787 nmgpu
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh ofs-alohajiang /nfs/ofs-alohajiang 56bfb0a2b4454b4cb5d305a8507dc1e0 hnapu
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh ofs-llm-data /nfs/ofs-llm-ssd  b00b083f426245d1a6abbc2f0164124a nmgmlmodeltrain
sudo bash /mnt/common/jianshu/ofs/release/current/script/ofs_mount.sh ofs-llab-cold /nfs/ofs-llab-cold a7ce0339d04d4b008040e3a28e3fc44e nmgllab
WORK_PATH="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1

which python
g++ --version
export PATH="/home/luban/miniconda3/bin:$PATH"
export PYTHONPATH=$WORK_PATH:$PYTHONPATH
export TMPDIR=/tmp-data
sudo chmod -R 777 /tmp-data

pip install accelerate rich yunchang --trusted-host didiyum.sys.xiaojukeji.com -i http://didiyum.sys.xiaojukeji.com/didiyum/pip/simple/
pip install sglang[all]==0.5.4  --trusted-host didiyum.sys.xiaojukeji.com -i http://didiyum.sys.xiaojukeji.com/didiyum/pip/simple/
sed -i.bak '644s/log_info_on_rank0(logger, "Chunked prefix cache is turned on.")/pass/' /home/luban/miniconda3/lib/python3.12/site-packages/sglang/srt/model_executor/model_runner.py
export FLASHINFER_DISABLE_VERSION_CHECK=1
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1
cp patch/deepseek_v2.py /home/luban/miniconda3/lib/python3.12/site-packages/sglang/srt/models/.
# 单机, 可指定机器数
N_NODE=1
N_PER_NODE=8
ddp_config_option=" --nproc_per_node ${N_PER_NODE} \
            --nnodes 1 \
            --node_rank 0 \
            --master_addr 127.0.0.1 \
            --master_port 65531 "

# 多机
if [ ! -z $DISTRIBUTED_NODE_COUNT ]; then
    N_NODE=$DISTRIBUTED_NODE_COUNT
    N_PER_NODE=$RESOURCE_NUM_GPU
    ddp_config_option=" --nproc_per_node $RESOURCE_NUM_GPU \
              --nnodes $DISTRIBUTED_NODE_COUNT \
              --node_rank $DISTRIBUTED_NODE_RANK \
              --master_addr $DISTRIBUTED_MASTER_HOSTS \
              --master_port $DISTRIBUTED_PYTORCH_PORT "

    export NCCL_DEBUG=INFO
    export NCCL_SOCKET_IFNAME=eth0
    export NCCL_IB_GID_INDEX=3
    export NCCL_IB_DISABLE=0
    # export NCCL_IB_HCA=${hca_list::-1}
    export NCCL_IB_HCA="mlx5_0:1,mlx5_3:1,mlx5_4:1,mlx5_5:1"
    export NCCL_NET_GDR_LEVEL=2
    export NCCL_IB_QPS_PER_CONNECTION=4
    # export NCCL_IB_TC=160
    export NCCL_IB_TC=98        # 内部98  cat /sys/class/infiniband/mlx5_5/tc/1/traffic_class
    export NCCL_IB_TIMEOUT=22
    export NCCL_PXN_DISABLE=0

    export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH

    export GLOO_SOCKET_IFNAME=eth0
fi

echo $ddp_config_option
#sudo apt-get install parallel rsync -y
# 核心：按目录拆分，每个目录用1个rsync进程复制，避免单文件并发冲突
#find /nfs/ofs-llab-cold/model/deepseek-ai/DeepSeek-V3.1 -type d -not -path "*/\.cache/*" -not -empty | parallel -j16 'rsync -av --progress --exclude=".*" {}/* /tmp-data/$(echo {} | sed "s|^/nfs/ofs-llab-cold/model/deepseek-ai/DeepSeek-V3.1/||" | xargs dirname)/'

#rsync -av --progress --num-threads=4 /nfs/ofs-llab-cold/model/deepseek-ai/DeepSeek-V3.1/ /tmp-data/.
method=$1
data_verison=$2
template=deepseek3

model_path=/nfs/ofs-llab-cold/model/deepseek-ai/DeepSeek-V3.1
hidden_state_path=/nfs/ofs-llab-cold/users/daiyajun/models/Eagle3_ds_hidden_state_$data_verison/
train_data_path=/nfs/ofs-llm-ssd/user/daiyajun/dtaxi/data/MTP/processed_data/$data_verison.json
eval_data_path=/nfs/ofs-llm-ssd/user/daiyajun/dtaxi/data/MTP/processed_data/val_$data_verison.json
eval_hidden_state_path=/nfs/ofs-llab-cold/users/daiyajun/models/Eagle3_ds_hidden_state_val_eval_$data_verison.json
save_model_path=/nfs/ofs-llm-ssd/models/fengyu/output/ups_Eagle3_ds_$data_verison
export SGLANG_TORCH_PROFILER_DIR=$save_model_path

# max_len=16400 #20480
max_len=10240
if [[ $method == "generate" ]]; then
    torchrun \
        --nproc_per_node=8 \
        scripts/prepare_hidden_states.py \
        --enable-aux-hidden-states \
        --tp-size 8 \
        --batch-size 1 \
        --mem-frac 0.75 \
        --max-length $max_len \
        --data-path $train_data_path \
        --output-path $hidden_state_path \
        --model-path $model_path \
        --chat-template $template
else
    torchrun \
        $ddp_config_option \
        scripts/train_eagle3_offline.py \
        --target-model-path $model_path \
        --draft-model-config ./configs/deepseek-r1-671b-eagle3.json \
        --train-data-path $train_data_path \
        --train-hidden-states-path $hidden_state_path \
        --output-dir $save_model_path \
        --num-epochs 10 \
        --tp-size 1 \
        --draft-global-batch-size 32 \
        --draft-micro-batch-size 1 \
        --learning-rate 5e-5 \
        --draft-attention-backend usp \
        --sp-ulysses-size 2 \
        --sp-ring-size 2 \
        --max-length $max_len \
        --chat-template $template \
        --cache-dir ./cache \
        --dist-timeout=10 \
        --log-steps 1 \
        --ttt-length 5 \
        --report tensorboard \
        --eval-data-path $eval_data_path \
        --eval-hidden-states-path $eval_hidden_state_path \
        --profile \
        --profile-start-step  10 \
        --profile-num-steps 3
fi


