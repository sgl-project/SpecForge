which python
g++ --version
export PATH="/home/luban/miniconda3/bin:$PATH"
WORK_PATH="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
export PYTHONPATH=WORK_PATH:$PYTHONPATH
export TMPDIR=/tmp-data
sudo chmod -R 777 /tmp-data
echo DISTRIBUTED_MASTER_HOSTS:$DISTRIBUTED_MASTER_HOSTS
echo DISTRIBUTED_NODE_RANK:$DISTRIBUTED_NODE_RANK
N_NODE=$DISTRIBUTED_NODE_COUNT
N_PER_NODE=$RESOURCE_NUM_GPU
pip install sglang[all]==0.5.4  --trusted-host didiyum.sys.xiaojukeji.com -i http://didiyum.sys.xiaojukeji.com/didiyum/pip/simple/
sed -i.bak '644s/log_info_on_rank0(logger, "Chunked prefix cache is turned on.")/pass/' /home/luban/miniconda3/lib/python3.12/site-packages/sglang/srt/model_executor/model_runner.py
export FLASHINFER_DISABLE_VERSION_CHECK=1
pip list
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1
cp patch/deepseek_v2.py /home/luban/miniconda3/lib/python3.12/site-packages/sglang/srt/models/.
torchrun \
    --nproc_per_node  $RESOURCE_NUM_GPU \
    --nnodes $DISTRIBUTED_NODE_COUNT \
    --node_rank $DISTRIBUTED_NODE_RANK \
    --master_addr $DISTRIBUTED_MASTER_HOSTS \
    --master_port $DISTRIBUTED_PYTORCH_PORT \
    scripts/prepare_hidden_states.py \
    --enable-aux-hidden-states \
    --tp-size 16 \
    --batch-size 1 \
    --data-path /nfs/ofs-llm-ssd/user/daiyajun/dtaxi/data/MTP/processed_data/online_v31_personal_demand_data.json \
    --output-path /nfs/ofs-fengyu/data/hidden_state/ \
    --target-model-path /nfs/ofs-llab-cold/model/deepseek-ai/DeepSeek-V3.1 \
    --chat-template deepseek-v3 \
    --max-length 16400
