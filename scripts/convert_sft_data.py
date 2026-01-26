#!/usr/bin/env python
"""
将 SFT 训练数据转换为 SpecForge EAGLE3 训练格式。

原始格式:
{
    "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
    "images": ["path1.jpg", "path2.jpg"]
}

目标格式:
{
    "conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
    "image": ["path1.jpg", "path2.jpg"]  # 或单个路径
}
"""

import json
import random
import argparse
import os
from tqdm import tqdm


def convert_item(item):
    """转换单条数据"""
    converted = {
        "conversations": item["messages"],
        "image": item.get("images", []),  # 统一使用 list 格式
    }
    
    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert SFT data to SpecForge format")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of samples to extract")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--validate-images", action="store_true", help="Validate image paths exist")
    # 新增筛选参数
    parser.add_argument("--single-image", action="store_true", help="Only keep single-image samples")
    parser.add_argument("--single-turn", action="store_true", help="Only keep single-turn conversations")
    parser.add_argument("--min-response-len", type=int, default=None, help="Minimum response length")
    parser.add_argument("--max-response-len", type=int, default=None, help="Maximum response length")
    args = parser.parse_args()
    
    print(f"Reading from: {args.input}")
    
    # 读取所有数据
    data = []
    invalid_count = 0
    filtered_counts = {"single_image": 0, "single_turn": 0, "response_len": 0}
    
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading data"):
            item = json.loads(line)
            
            # 可选：验证图片路径
            if args.validate_images:
                images = item.get("images", [])
                valid = all(os.path.exists(img) for img in images)
                if not valid:
                    invalid_count += 1
                    continue
            
            # 筛选：单图
            if args.single_image:
                images = item.get("images", [])
                if len(images) != 1:
                    filtered_counts["single_image"] += 1
                    continue
            
            # 筛选：单轮对话
            if args.single_turn:
                messages = item.get("messages", [])
                if len(messages) != 2:
                    filtered_counts["single_turn"] += 1
                    continue
            
            # 筛选：response 长度
            if args.min_response_len is not None or args.max_response_len is not None:
                messages = item.get("messages", [])
                # 找到 assistant 的回复
                response = ""
                for msg in messages:
                    if msg.get("role") == "assistant":
                        response = msg.get("content", "")
                        break
                response_len = len(response)
                
                if args.min_response_len and response_len < args.min_response_len:
                    filtered_counts["response_len"] += 1
                    continue
                if args.max_response_len and response_len > args.max_response_len:
                    filtered_counts["response_len"] += 1
                    continue
            
            data.append(item)
    
    print(f"Total valid samples: {len(data)}")
    if args.validate_images:
        print(f"Skipped (invalid images): {invalid_count}")
    if args.single_image:
        print(f"Filtered (not single image): {filtered_counts['single_image']}")
    if args.single_turn:
        print(f"Filtered (not single turn): {filtered_counts['single_turn']}")
    if args.min_response_len or args.max_response_len:
        print(f"Filtered (response length): {filtered_counts['response_len']}")
    
    # 采样
    if args.sample_size and args.sample_size < len(data):
        print(f"Sampling {args.sample_size} items (seed={args.seed})")
        random.seed(args.seed)
        data = random.sample(data, args.sample_size)
    
    # 统计
    single_image = sum(1 for item in data if len(item.get("images", [])) == 1)
    multi_image = sum(1 for item in data if len(item.get("images", [])) > 1)
    print(f"Single image samples: {single_image}")
    print(f"Multi image samples: {multi_image}")
    
    # 转换并保存
    print(f"Writing to: {args.output}")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc="Converting"):
            converted = convert_item(item)
            f.write(json.dumps(converted, ensure_ascii=False) + '\n')
    
    print("Done!")


if __name__ == "__main__":
    main()
