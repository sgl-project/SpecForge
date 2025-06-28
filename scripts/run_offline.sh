SHAREGPT_DATAPATHS=$HOME/huggingface_cache/Qwen2.5-7B-Instruct/sgl_eagle_sharegpt/
ULTRACHAT_DATAPATHS=$HOME/huggingface_cache/Qwen2.5-7B-Instruct/sgl_eagle_ultrachat/
MODEL_PATH=$HOME/huggingface_cache/Qwen2.5-7B-Instruct

cd ~/sgl-spec/sgl_eagle
if [ -d $SHAREGPT_DATAPATHS ]; then
    echo "SHAREGPT_DATAPATHS exists"
else
    echo "SHAREGPT_DATAPATHS does not exist, generating data..."
    python data/offline_eagle_data.py \
        --outdir $SHAREGPT_DATAPATHS \
        --model_path $MODEL_PATH \
        --dataset sharegpt
fi
if [ -d $ULTRACHAT_DATAPATHS ]; then
    echo "ULTRACHAT_DATAPATHS exists"
else
    echo "ULTRACHAT_DATAPATHS does not exist, generating data..."
    python data/offline_eagle_data.py \
        --outdir $ULTRACHAT_DATAPATHS \
        --model_path $MODEL_PATH \
        --dataset ultrachat
fi

torchrun --nproc_per_node=8 --master_port=29500 train_offline.py --epochs 10