#!/bin/bash
torchrun \
    --standalone \
    --nproc_per_node 1 \
    /data1/nfs15/nfs/zhanglei335/git-project/SpecForge/scripts/train_eagle3_online.py \
    --target-model-path /data1/nfs15/nfs/bigdata/zhanglei/ml/inference/model-demo/hf/Qwen/Qwen2.5-VL-7B-Instruct \
    --draft-model-config /data1/nfs15/nfs/zhanglei335/git-project/SpecForge/configs/qwen2-5-vl-eagle3.json \
    --train-data-path /data1/nfs15/nfs/bigdata/zhanglei/ml/datasets/FreedomIntelligence/ALLaVA-4V/allava4v.jsonl \
    --output-dir /aistudio/workspace/mlsys-data/models/mlsys/specforge \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 1e-4 \
    --max-length 4096 \
    --chat-template qwen2-vl \
    --is-vlm \
    --tp-size 1 