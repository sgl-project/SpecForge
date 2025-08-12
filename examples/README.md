## Training with Flex Attention

Flex attention saves 10x memory and also makes training faster. It is currently in experimental stage. To enable flex attention, you need to pass `--attention-backend flex_attention` to the training script. To allow sharing of compiled kernels, you need to set `TORCHINDUCTOR_CACHE_DIR` to the cache directory.

> <b> Note: Make sure you install torch 2.8.0!</b>

Example training script:
```bash
TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels \
torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    $ROOT_DIR/scripts/train_eagle3_online.py \
    --target-model-path meta-llama/Llama-3.1-8B-Instruct \
    --draft-model-config $ROOT_DIR/configs/llama3-8B-eagle3.json \
    --train-data-path $ROOT_DIR/cache/dataset/sharegpt.jsonl \
    --output-dir $ROOT_DIR/outputs/llama3-8b-eagle3 \
    --num-epochs 1 \
    --batch-size 4 \
    --learning-rate 1e-4 \
    --max-length 2048 \
    --chat-template llama3 \
    --cache-dir $ROOT_DIR/cache
    --attention-backend flex_attention
```
