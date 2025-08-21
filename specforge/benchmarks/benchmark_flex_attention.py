import argparse
import gc
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch._dynamo as dynamo
from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache

from specforge.modeling.draft.llama3_eagle import (
    LlamaAttention,
    LlamaFlexAttention,
    prepare_decoder_attention_mask,
)
from specforge.utils import padding

dynamo.config.recompile_limit = 64

config_dict = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 16384,
    "rms_norm_eps": 1e-05,
    "vocab_size": 32000,
    "hidden_act": "silu",
    "num_hidden_layers": 1,
}

config = LlamaConfig(**config_dict)

TTT_LENGTH = 7
BATCH_SIZE = 4
HIDDEN_SIZE = config.hidden_size * 2


def run_attention(
    seq_len: int,
    hidden_states_list: list[torch.Tensor],
    attention_backend: str = "sdpa",
    enable_profile: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = hidden_states_list[0].shape[0]
    # Initialize cache and attention function based on backend
    if attention_backend == "sdpa":
        cache_hidden = [[], []]
        past_key_values = None
        attn_func = LlamaAttention(config).to(device).to(torch.bfloat16)
    elif attention_backend == "flex_attention":
        cache_hidden = None
        past_key_values = DynamicCache()
        attn_func = LlamaFlexAttention(config).to(device).to(torch.bfloat16)
    else:
        raise ValueError(f"Unknown attention backend: {attention_backend}")

    # Simulate inputs - move to device
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
    input_embeds = torch.randn(batch_size, seq_len, config.hidden_size).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    decoder_attention_mask = prepare_decoder_attention_mask(
        attention_mask=attention_mask,
        input_shape=(batch_size, seq_len),
        inputs_embeds=input_embeds,
        past_key_values_length=0,
    )

    loss_list = []

    if attention_backend == "flex_attention" and enable_profile:
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                f"./profiler_logs/{attention_backend}"
            ),
            record_shapes=False,
            profile_memory=False,
            with_stack=True,
            with_modules=False,
        )
        profiler.start()
    for idx in range(TTT_LENGTH):
        is_last = idx == TTT_LENGTH - 1
        hidden_states = hidden_states_list[idx]
        # Call attention function with appropriate parameters
        if attention_backend == "sdpa":
            output = attn_func(
                hidden_states=hidden_states,
                attention_mask=decoder_attention_mask,
                position_ids=position_ids,
                cache_hidden=cache_hidden,
                output_attentions=False,
                use_cache=True,
            )
        else:  # flex_attention
            output = attn_func(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=False,
                use_cache=True,
            )

        # Compute a simple loss for benchmarking
        loss = output[0].sum()
        loss_list.append(loss)

        if attention_backend == "sdpa" and not is_last:
            # Step 5.7: we need to update the loss mask
            ind = torch.arange(seq_len, device=decoder_attention_mask.device)
            ind0 = ind[idx:]
            ind1 = ind[: seq_len - idx]
            decoder_attention_mask[:, :, ind0, ind1] = torch.finfo(
                decoder_attention_mask.dtype
            ).min

    # Compute mean loss and backward pass
    if loss_list:
        mean_loss = sum(loss_list) / len(loss_list)
        mean_loss.backward()

    if attention_backend == "flex_attention" and enable_profile:
        profiler.stop()


def benchmark_function(
    attention_backend: str,
    seq_lengths: list,
    enable_profile: bool = False,
    enable_warmup: bool = True,
):
    """Benchmark a function for speed and GPU memory usage per sequence length."""
    print(f"\n=== Benchmarking {attention_backend} ===")

    results_per_seq_len = []

    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Warm up runs for this sequence length
        if enable_warmup:
            print("Warming up...")
            for _ in range(2):
                hidden_states = [
                    torch.randn(
                        BATCH_SIZE,
                        seq_len,
                        HIDDEN_SIZE,
                        requires_grad=True,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    for _ in range(TTT_LENGTH)
                ]
                run_attention(seq_len, hidden_states, attention_backend)
            # Clear cache again after warmup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

        # Record initial memory
        initial_memory = 0
        if torch.cuda.is_available():
            initial_memory = torch.cuda.memory_allocated()

        hidden_states = [
            torch.randn(
                BATCH_SIZE,
                seq_len,
                HIDDEN_SIZE,
                requires_grad=True,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for _ in range(TTT_LENGTH)
        ]
        start_time = time.time()
        run_attention(
            seq_len,
            hidden_states,
            attention_backend,
            enable_profile and seq_len == seq_lengths[0],
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()

        # Record memory usage
        peak_memory = 0
        current_memory = 0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()
            current_memory = torch.cuda.memory_allocated()

        results_per_seq_len.append(
            {
                "seq_len": seq_len,
                "time": end_time - start_time,
                "peak_memory": peak_memory,
                "memory_increase": current_memory - initial_memory,
            }
        )

        print(f"  Time: {end_time - start_time:.3f}s")
        print(f"  Peak memory: {peak_memory / 1024**3:.3f} GB")

    return results_per_seq_len


def plot_results(eagle_results, flex_results, seq_lengths):
    """Plot speed and memory comparison between Eagle and Flex attention."""

    # Extract data for plotting
    eagle_times = [r["time"] for r in eagle_results]
    flex_times = [r["time"] for r in flex_results]
    eagle_memory = [r["peak_memory"] / 1024**3 for r in eagle_results]  # Convert to GB
    flex_memory = [r["peak_memory"] / 1024**3 for r in flex_results]  # Convert to GB

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Speed comparison plot
    ax1.plot(
        seq_lengths, eagle_times, "b-o", label="Eagle (SDPA)", linewidth=2, markersize=8
    )
    ax1.plot(
        seq_lengths,
        flex_times,
        "r-s",
        label="Flex Attention",
        linewidth=2,
        markersize=8,
    )
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Time (seconds)")
    ax1.set_title("Speed Comparison: Eagle vs Flex Attention")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("linear")
    ax1.set_yscale("log")

    # Memory comparison plot
    ax2.plot(
        seq_lengths,
        eagle_memory,
        "b-o",
        label="Eagle (SDPA)",
        linewidth=2,
        markersize=8,
    )
    ax2.plot(
        seq_lengths,
        flex_memory,
        "r-s",
        label="Flex Attention",
        linewidth=2,
        markersize=8,
    )
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Peak Memory (GB)")
    ax2.set_title("Memory Usage Comparison: Eagle vs Flex Attention")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Set y-axis ticks every 10GB
    max_memory = max(max(eagle_memory), max(flex_memory))
    ax2.set_yticks(np.arange(0, max_memory + 10, 10))

    plt.tight_layout()
    plt.savefig("attention_benchmark_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Print summary statistics
    print(f"\n=== Performance Summary ===")
    print(f"Sequence lengths tested: {seq_lengths}")
    print(f"\nSpeed ratios (Eagle/Flex):")
    for i, seq_len in enumerate(seq_lengths):
        ratio = eagle_times[i] / flex_times[i] if flex_times[i] > 0 else float("inf")
        print(f"  {seq_len:4d}: {ratio:.2f}x")

    print(f"\nMemory ratios (Eagle/Flex):")
    for i, seq_len in enumerate(seq_lengths):
        ratio = eagle_memory[i] / flex_memory[i] if flex_memory[i] > 0 else float("inf")
        print(f"  {seq_len:4d}: {ratio:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark attention mechanisms")
    parser.add_argument(
        "--enable-profile", action="store_true", help="Enable profiling"
    )
    args = parser.parse_args()

    print("PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA available:", torch.cuda.is_available())
        print("GPU:", torch.cuda.get_device_name())
        print(
            "GPU memory:",
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            "GB",
        )
    else:
        print("CUDA not available - running on CPU")

    # Define sequence lengths to test
    seq_lengths = [128 * i for i in range(1, 28, 4)]
    # Add extra long context
    seq_lengths.extend([16384, 32768])

    print(f"Testing sequence lengths: {seq_lengths}")

    # Run benchmarks
    print("\n" + "=" * 50)
    # Truncate seqlen after 2560 since naive eagle goes OOM
    eagle_seq_lengths = [seq_len for seq_len in seq_lengths if seq_len <= 2560]
    eagle_results = benchmark_function("sdpa", eagle_seq_lengths)
    print("\n" + "=" * 50)
    flex_results = benchmark_function(
        "flex_attention", seq_lengths, enable_profile=args.enable_profile
    )
    # Pad the memory usage on eagle to max memory 80GB when data not available
    max_time = max(result["time"] for result in flex_results)
    for result in flex_results:
        if result["seq_len"] not in eagle_seq_lengths:
            eagle_results.append(
                {
                    "seq_len": result["seq_len"],
                    "time": max_time,
                    "peak_memory": 80 * 1024**3,
                    "memory_increase": 0,  # Not used in plotting
                }
            )

    # Plot results
    plot_results(eagle_results, flex_results, seq_lengths)
