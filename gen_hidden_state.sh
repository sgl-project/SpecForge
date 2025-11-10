which python
g++ --version
export PATH="/home/luban/miniconda3/bin:$PATH"
ls /tmp-data
export TMPDIR=/tmp-data
sudo chmod -R 777 /tmp-data
#sudo apt-get install parallel rsync -y
# 核心：按目录拆分，每个目录用1个rsync进程复制，避免单文件并发冲突
#find /nfs/ofs-llab-cold/model/deepseek-ai/DeepSeek-V3.1 -type d -not -path "*/\.cache/*" -not -empty | parallel -j16 'rsync -av --progress --exclude=".*" {}/* /tmp-data/$(echo {} | sed "s|^/nfs/ofs-llab-cold/model/deepseek-ai/DeepSeek-V3.1/||" | xargs dirname)/'

#rsync -av --progress --num-threads=4 /nfs/ofs-llab-cold/model/deepseek-ai/DeepSeek-V3.1/ /tmp-data/.
torchrun \
    --nproc_per_node=8 \
    scripts/prepare_hidden_states.py \
    --enable-aux-hidden-states \
    --tp-size 8 \
    --batch-size 1 \
    --mem-frac 0.75 \
    --max-length 2048 \
    --data-path /nfs/ofs-fengyu/data/dtaxi_v31_dt1106.jsonl \
    --output-path /nfs/ofs-fengyu/data/hidden_state4/ \
    --model-path /nfs/ofs-llab-cold/model/deepseek-ai/DeepSeek-V3.1 \
    --chat-template deepseek3
