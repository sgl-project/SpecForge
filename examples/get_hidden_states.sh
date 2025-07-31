torchrun --nproc_per_node=8 \
    scripts/prepare_hidden_states.py \
    --model-path moonshotai/Kimi-K2-Instruct \
    --enable-aux-hidden-states \
    --data-path cache/dataset/sharegpt.jsonl \
    --chat-template kimi_k2 \
    --max-length 2048 \
    --tp-size 8 \
    --batch-size 1 \
    --mem-frac=0.95 \
    --num-samples 2000