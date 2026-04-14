#!/usr/bin/env python3
"""
使用 vLLM 测试 EAGLE3 模型
"""
import argparse
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter


def parse_args():
    parser = argparse.ArgumentParser(description="测试 EAGLE3 模型效果")
    parser.add_argument("--model-dir", type=str, required=True, help="目标模型路径")
    parser.add_argument("--eagle-dir", type=str, required=True, help="EAGLE3 draft model 路径")
    parser.add_argument("--num-prompts", type=int, default=20, help="测试样本数量")
    parser.add_argument("--output-len", type=int, default=256, help="最大生成长度")
    parser.add_argument("--num-spec-tokens", type=int, default=4, help="推测 token 数量")
    parser.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="GPU 显存使用率")
    parser.add_argument("--max-model-len", type=int, default=8192, help="最大模型长度")
    parser.add_argument("--print-output", action="store_true", help="打印生成结果")
    parser.add_argument("--temp", type=float, default=0.0, help="采样温度")
    return parser.parse_args()


def main():
    args = parse_args()

    print("="*60)
    print("EAGLE3 模型测试 (vLLM)")
    print("="*60)
    print(f"目标模型: {args.model_dir}")
    print(f"EAGLE3 模型: {args.eagle_dir}")
    print(f"推测 token 数: {args.num_spec_tokens}")
    print(f"测试样本数: {args.num_prompts}")
    print("="*60)

    # 准备测试 prompts
    test_prompts = [
        "请解释什么是机器学习。",
        "写一个 Python 函数计算斐波那契数列的第 n 项。",
        "What is the capital of France?",
        "请介绍一下深度学习的基本概念。",
        "Write a short story about a robot learning to paint.",
        "解释一下量子计算的原理。",
        "What are the main differences between Python and Java?",
        "请推荐一些学习编程的路径。",
        "Describe the process of photosynthesis.",
        "如何优化神经网络训练过程？",
        "What is the meaning of life according to different philosophies?",
        "请解释 RESTful API 的设计原则。",
        "Compare and contrast React and Vue.js.",
        "什么是区块链技术？它有哪些应用场景？",
        "Explain the concept of recursion in programming.",
        "请分析人工智能的发展趋势。",
        "What are some best practices for code review?",
        "描述一下云计算的三种服务模式。",
        "How does a compiler work?",
        "请谈谈你对软件工程的理解。",
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
        gpu_memory_utilization=args.gpu_memory_utilization,
        speculative_config=speculative_config,
        disable_log_stats=False,
        max_model_len=args.max_model_len,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # 准备 prompts
    llm_prompts = [
        tokenizer.encode(p, add_special_tokens=False)
        for p in test_prompts
    ]

    sampling_params = SamplingParams(
        temperature=args.temp,
        max_tokens=args.output_len,
    )

    print(f"\n开始生成 {len(test_prompts)} 个样本...\n")

    import time
    start_time = time.time()

    outputs = llm.generate(
        llm_prompts,
        sampling_params=sampling_params,
    )

    end_time = time.time()
    total_time = end_time - start_time

    # 打印生成结果
    if args.print_output:
        for i, output in enumerate(outputs):
            print("-" * 50)
            print(f"Prompt {i+1}: {test_prompts[i][:100]}...")
            print(f"Generated: {output.outputs[0].text[:200]}...")
            print("-" * 50)

    # 收集指标
    metrics = llm.get_metrics()

    total_num_output_tokens = sum(
        len(output.outputs[0].token_ids) for output in outputs
    )

    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0

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

    # 计算统计
    acceptance_length = 1 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1
    accept_rate = num_accepted_tokens / num_draft_tokens if num_draft_tokens > 0 else 0

    print("\n" + "="*60)
    print("测试结果")
    print("="*60)
    print(f"总生成 tokens: {total_num_output_tokens}")
    print(f"总用时: {total_time:.2f}s")
    print(f"平均生成速度: {total_num_output_tokens / total_time:.2f} tokens/sec")
    print("-"*60)
    print(f"Draft 次数: {num_drafts}")
    print(f"Draft tokens 数: {num_draft_tokens}")
    print(f"接受 tokens 数: {num_accepted_tokens}")
    print(f"接受率: {accept_rate:.2%}")
    print(f"平均接受长度: {acceptance_length:.2f}")
    print("="*60)

    # 理论加速比
    # 如果接受长度是 2，意味着平均每次 draft 验证 2 个 token，加速约 2x
    theoretical_speedup = acceptance_length
    print(f"理论加速比: ~{theoretical_speedup:.2f}x")
    print("="*60)


if __name__ == "__main__":
    main()
