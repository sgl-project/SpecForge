#!/usr/bin/env python3
"""
使用 vLLM 测试 EAGLE3 模型
"""
import time
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.v1.metrics.reader import Counter, Vector


def parse_args():
    parser = argparse.ArgumentParser(description="测试 EAGLE3 模型效果")
    parser.add_argument("--model-dir", type=str, default="/home/pairshoe/ljl/train_eagle3/models/Qwen/Qwen3.5-4B", help="目标模型路径")
    parser.add_argument("--eagle-dir", type=str, default="/home/pairshoe/ljl/train_eagle3/SpecForge/outputs/qwen3.5-4b-eagle3-20k/epoch_2_step_12000", help="EAGLE3 draft model 路径")
    parser.add_argument("--num-prompts", type=int, default=5, help="测试样本数量")
    parser.add_argument("--output-len", type=int, default=256, help="最大生成长度")
    parser.add_argument("--num-spec-tokens", type=int, default=4, help="推测 token 数量")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="GPU 显存使用率")
    parser.add_argument("--max-model-len", type=int, default=8192, help="最大模型长度")
    parser.add_argument("--print-output", action="store_true", help="打印生成结果")
    parser.add_argument("--temp", type=float, default=0.0, help="采样温度")
    parser.add_argument("--enable-chunked-prefill", action="store_true", help="启用 chunked prefill")
    parser.add_argument("--enforce-eager", action="store_true", help="强制使用 eager 模式")
    return parser.parse_args()


def main(args):
    print("="*60)
    print("EAGLE3 模型测试 (vLLM)")
    print("="*60)
    print(f"目标模型: {args.model_dir}")
    print(f"EAGLE3 模型: {args.eagle_dir}")
    print(f"推测 token 数: {args.num_spec_tokens}")
    print(f"测试样本数: {args.num_prompts}")
    print("="*60)

    # 测试 prompts
    test_prompts = [
        "请解释什么是机器学习。",
        "写一个 Python 函数计算斐波那契数列的第 n 项。",
        "What is the capital of France?",
        "请介绍一下深度学习的基本概念。",
        "Compare Python and Java programming languages.",
    ][:args.num_prompts]

    # 配置 EAGLE3
    speculative_config = {
        "method": "eagle3",
        "model": args.eagle_dir,
        "num_speculative_tokens": args.num_spec_tokens,
    }

    print("\n初始化 vLLM 模型...")
    llm = LLM(
        model=args.model_dir,
        trust_remote_code=True,
        tensor_parallel_size=args.tp,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=speculative_config,
        disable_log_stats=False,
        max_model_len=args.max_model_len,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # 准备 prompts - 使用 token ids
    prompt_ids = [
        tokenizer.encode(p, add_special_tokens=False)
        for p in test_prompts
    ]

    sampling_params = SamplingParams(
        temperature=args.temp,
        max_tokens=args.output_len,
    )

    print(f"\n开始生成 {len(test_prompts)} 个样本...\n")

    start_time = time.time()
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=x) for x in prompt_ids],
        sampling_params=sampling_params,
    )
    end_time = time.time()

    total_time = end_time - start_time

    # 打印结果
    if args.print_output:
        for i, output in enumerate(outputs):
            print("-" * 50)
            print(f"Prompt {i+1}: {test_prompts[i]}")
            print(f"Generated: {output.outputs[0].text}")
            print("-" * 50)

    # 获取指标
    metrics = llm.get_metrics()

    total_tokens = 0
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = [0] * args.num_spec_tokens

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    # 计算 total tokens
    for output in outputs:
        total_tokens += len(output.outputs[0].token_ids)

    # 计算统计
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    accept_rate = num_accepted_tokens / num_draft_tokens if num_draft_tokens > 0 else 0

    print("\n" + "="*60)
    print("测试结果")
    print("="*60)
    print(f"总生成 tokens: {total_tokens}")
    print(f"总用时: {total_time:.2f}s")
    print(f"生成速度: {total_tokens / total_time:.2f} tokens/sec")
    print("-"*60)
    print(f"Draft 次数: {num_drafts}")
    print(f"Draft tokens: {num_draft_tokens}")
    print(f"接受 tokens: {num_accepted_tokens}")
    print(f"接受率: {accept_rate:.2%}")
    print(f"平均接受长度: {acceptance_length:.2f}")
    print(f"理论加速比: ~{acceptance_length:.2f}x")
    print("="*60)

    # 打印每个位置的接受率
    print("\n各位置接受率:")
    for i in range(len(acceptance_counts)):
        acceptance_rate = acceptance_counts[i] / num_drafts if num_drafts > 0 else 0
        print(f"  token {i}: {acceptance_rate:.2%}")

    return acceptance_length


if __name__ == "__main__":
    args = parse_args()
    acceptance_length = main(args)
