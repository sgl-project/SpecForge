import argparse
import time

import torch

from specforge.core.loss import LogSoftmaxLoss, _compute_loss

TTT_LENGTH = 7


def benchmark_loss_method(
    loss_method: str,
    test_configs: list,
):
    """Benchmark a loss computation method for speed and GPU memory usage."""
    print(f"\n=== Benchmarking {loss_method} Loss ===")

    results = []

    for config in test_configs:
        B, T, V = config
        print(f"\nTesting config: B={B}, T={T}, V={V}")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Create tensors outside timing measurement
        target = torch.softmax(
            torch.randn(B, T, V, device="cuda", dtype=torch.float32), dim=-1
        )
        position_mask = torch.ones((B, T, 1), dtype=torch.bool, device="cuda")

        # Pre-allocate logits tensors for each TTT step
        logits_list = []
        for i in range(TTT_LENGTH):
            logits = torch.randn(
                B, T, V, device="cuda", requires_grad=True, dtype=torch.float32
            )
            logits_list.append(logits)

        torch.cuda.synchronize()  # Ensure all operations are complete
        start_time = time.time()

        plosses = []
        for i in range(TTT_LENGTH):
            logits = logits_list[i]
            if loss_method == "triton":
                loss = LogSoftmaxLoss.apply(logits, target, position_mask)
            else:
                loss = _compute_loss(logits, target, position_mask)
            plosses.append(loss)

        ploss_weight = [0.8**i for i in range(len(plosses))]
        ploss = (
            sum([ploss_weight[i] * plosses[i] for i in range(len(plosses))])
            / TTT_LENGTH
        )
        ploss.backward()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()
        total_time = end_time - start_time
        # Record memory usage
        peak_memory = 0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated()

        results.append(
            {
                "B": B,
                "T": T,
                "V": V,
                "time_total": total_time,
                "peak_memory": peak_memory,
            }
        )

        print(f"  Total time (forward + backward): {total_time*1000:.3f}ms")
        print(f"  Peak memory: {peak_memory / 1024**3:.3f} GB")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark loss computation methods")
    parser.add_argument(
        "--num-runs", type=int, default=5, help="Number of runs for averaging"
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

    # Define test configurations (B, T, V)
    test_configs = [
        (1, 1024, 32000),
        (1, 1024, 64000),
        (1, 4096, 32000),
        (1, 4096, 64000),
        (1, 8192, 32000),
        (1, 8192, 64000),
        (1, 16384, 32000),
    ]

    print(f"Testing configurations: {test_configs}")

    # Run benchmarks
    print("\n" + "=" * 60)
    pytorch_results = benchmark_loss_method("pytorch", test_configs)

    print("\n" + "=" * 60)
    triton_results = benchmark_loss_method("triton", test_configs)

    # Print results summary
    print(f"\n=== Performance Summary ===")
    print(f"Configurations tested: {len(test_configs)}")

    # Print detailed results table
    print(
        f"\n{'Config (B,T,V)':<15} {'PyTorch (ms)':<15} {'Triton (ms)':<15} {'Speedup':<10} {'PyTorch Mem (GB)':<18} {'Triton Mem (GB)':<15} {'Memory Save':<12}"
    )
    print("-" * 115)

    for i, config in enumerate(test_configs):
        B, T, V = config
        config_str = f"({B},{T},{V})"

        pytorch_result = next(
            (r for r in pytorch_results if r["B"] == B and r["T"] == T and r["V"] == V),
            None,
        )
        triton_result = next(
            (r for r in triton_results if r["B"] == B and r["T"] == T and r["V"] == V),
            None,
        )

        if pytorch_result and triton_result:
            pytorch_time_str = f"{pytorch_result['time_total']*1000:.2f}"
            pytorch_mem_str = f"{pytorch_result['peak_memory']/1024**3:.2f}"

            triton_time_str = f"{triton_result['time_total']*1000:.2f}"
            triton_mem_str = f"{triton_result['peak_memory']/1024**3:.2f}"

            if triton_result["time_total"] > 0:
                speedup = pytorch_result["time_total"] / triton_result["time_total"]
                speedup_str = f"{speedup:.2f}x"
            else:
                speedup_str = "N/A"

            # Calculate memory savings percentage
            if pytorch_result["peak_memory"] > 0:
                memory_save_pct = (
                    (pytorch_result["peak_memory"] - triton_result["peak_memory"])
                    / pytorch_result["peak_memory"]
                ) * 100
                memory_save_str = f"{memory_save_pct:.1f}%"
            else:
                memory_save_str = "N/A"

            print(
                f"{config_str:<15} {pytorch_time_str:<15} {triton_time_str:<15} {speedup_str:<10} {pytorch_mem_str:<18} {triton_mem_str:<15} {memory_save_str:<12}"
            )


if __name__ == "__main__":
    main()
