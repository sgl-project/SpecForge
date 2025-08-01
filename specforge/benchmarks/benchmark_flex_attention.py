import torch
import time
import gc
from transformers import LlamaConfig
from specforge.modeling.draft.llama3_eagle import (
    LlamaAttention, 
    prepare_decoder_attention_mask,
)
from specforge.modeling.draft.llama3_flex_attention import LlamaFlexAttention
from transformers.cache_utils import Cache, DynamicCache


config_dict = {
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-05,
    "vocab_size": 32000,
    "hidden_act": "silu",
}

config = LlamaConfig(**config_dict)



def run_eagle_llama_attention(seq_len: int, include_backward: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    attention = LlamaAttention(config).to(device)
    if include_backward:
        attention.train()  # Enable training mode for gradients
    else:
        attention.eval()

    batch_size = 1
    hidden_size = config.hidden_size * 2

    # Simulate inputs - move to device
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)

    attention_mask = torch.ones(batch_size, seq_len).to(device)
    # First 128 is padding
    attention_mask[:, 128:] = False
    input_embeds = torch.randn(batch_size, seq_len, config.hidden_size).to(device)
    decoder_attention_mask = prepare_decoder_attention_mask(
        attention_mask=attention_mask,
        input_shape=(batch_size, seq_len),
        inputs_embeds=input_embeds,
        past_key_values_length=0,
    )

    for idx in range(7):
        is_last = idx == 6
        
        # Clear gradients if doing backward pass
        if include_backward:
            attention.zero_grad()
        
        # Recreate cache for each iteration to avoid graph reuse
        cache_hidden = [[], []]  # [cache_k, cache_v] - fresh for each iteration
        
        # Remove torch.no_grad() when including backward
        context_manager = torch.no_grad() if not include_backward else torch.enable_grad()
        
        with context_manager:
            hidden_states = torch.randn(batch_size, seq_len, hidden_size, 
                                      requires_grad=include_backward).to(device)
            output = attention(
                hidden_states=hidden_states,
                attention_mask=decoder_attention_mask,
                position_ids=position_ids,
                cache_hidden=cache_hidden,
                output_attentions=False,
                use_cache=True
            )
            
            # Compute loss and backward pass if requested
            if include_backward:
                # Create a simple loss (sum of outputs)
                loss = output[0].sum()
                loss.backward()
                
        if not is_last:
            # Step 5.7: we need to update the loss mask
            ind = torch.arange(seq_len, device=decoder_attention_mask.device)
            ind0 = ind[idx:]
            ind1 = ind[: seq_len - idx]
            decoder_attention_mask[:, :, ind0, ind1] = torch.finfo(decoder_attention_mask.dtype).min

def run_flex_attention(seq_len: int, include_backward: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    llama_flex_attention = LlamaFlexAttention(config).to(device)
    if include_backward:
        llama_flex_attention.train()  # Enable training mode for gradients
    else:
        llama_flex_attention.eval()
        
    batch_size = 1
    hidden_size = config.hidden_size * 2
    # Simulate inputs - move to device
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(device)
    attention_mask = torch.ones(batch_size, seq_len).to(device)
    # First 128 is padding
    attention_mask[:, 128:] = False
    
    for i in range(7):
        # Clear gradients if doing backward pass
        if include_backward:
            llama_flex_attention.zero_grad()
        
        # Recreate cache for each iteration to avoid graph reuse
        past_key_values = DynamicCache()  # Fresh cache for each iteration
            
        context_manager = torch.no_grad() if not include_backward else torch.enable_grad()
        
        with context_manager:
            hidden_states = torch.randn(batch_size, seq_len, hidden_size, 
                                      requires_grad=include_backward).to(device)
            flex_output = llama_flex_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
            )
            
            # Compute loss and backward pass if requested
            if include_backward:
                # Create a simple loss (sum of outputs)
                loss = flex_output[0].sum()
                loss.backward()


def benchmark_function(func, func_name: str, *args, **kwargs):
    """Benchmark a function for speed and GPU memory usage."""
    print(f"\n=== Benchmarking {func_name} ===")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Warm up runs
    print("Warming up...")
    for _ in range(3):
        func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Clear cache again after warmup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark runs
    num_runs = 10
    times = []
    
    print(f"Running {num_runs} benchmark iterations...")
    
    # Record initial memory
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
    
    for i in range(num_runs):
        start_time = time.time()
        func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    
    print(f"Average time: {avg_time:.4f}s")
    print(f"Min time: {min_time:.4f}s")
    print(f"Max time: {max_time:.4f}s")
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated()
        current_memory = torch.cuda.memory_allocated()
        print(f"Peak GPU memory: {peak_memory / 1024**3:.3f} GB")
        print(f"Current GPU memory: {current_memory / 1024**3:.3f} GB")
        print(f"Memory increase: {(current_memory - initial_memory) / 1024**3:.3f} GB")
    
    return {
        'avg_time': avg_time,
        'min_time': min_time,
        'max_time': max_time,
        'peak_memory': peak_memory if torch.cuda.is_available() else 0,
        'memory_increase': (current_memory - initial_memory) if torch.cuda.is_available() else 0
    }


if __name__ == "__main__":
    print("PyTorch version:", torch.__version__)
    if torch.cuda.is_available():
        print("CUDA available:", torch.cuda.is_available())
        print("GPU:", torch.cuda.get_device_name())
        print("GPU memory:", torch.cuda.get_device_properties(0).total_memory / 1024**3, "GB")
    else:
        print("CUDA not available - running on CPU")
    
    # Set sequence length
    seq_len = 2048
    print(f"\nBenchmarking with sequence length: {seq_len}")
    
    # Benchmark both forward-only and forward+backward
    for include_backward in [True]:
        mode_str = "Forward + Backward" if include_backward else "Forward Only"
        print(f"\n{'='*60}")
        print(f"BENCHMARKING MODE: {mode_str}")
        print(f"{'='*60}")
        
        # Benchmark flex attention
        flex_results = benchmark_function(
            run_flex_attention, 
            f"FlexAttention ({mode_str})", 
            seq_len=seq_len, 
            include_backward=include_backward
        )
        
        # Force garbage collection between benchmarks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Benchmark eagle attention
        eagle_results = benchmark_function(
            run_eagle_llama_attention, 
            f"EagleAttention ({mode_str})", 
            seq_len=seq_len, 
            include_backward=include_backward
        )
        
        # Print comparison
        print("\n" + "="*50)
        print(f"COMPARISON SUMMARY - {mode_str}")
        print("="*50)
        
        speed_ratio = eagle_results['avg_time'] / flex_results['avg_time']
        print(f"Speed comparison (Eagle vs Flex):")
        print(f"  Eagle: {eagle_results['avg_time']:.4f}s")
        print(f"  Flex:  {flex_results['avg_time']:.4f}s")
        print(f"  Ratio: {speed_ratio:.2f}x ({'Eagle is faster' if speed_ratio < 1 else 'Flex is faster'})")
        
        if torch.cuda.is_available():
            memory_ratio = eagle_results['peak_memory'] / flex_results['peak_memory']
            print(f"\nMemory comparison (Eagle vs Flex):")
            print(f"  Eagle peak: {eagle_results['peak_memory'] / 1024**3:.3f} GB")
            print(f"  Flex peak:  {flex_results['peak_memory'] / 1024**3:.3f} GB")
            print(f"  Ratio: {memory_ratio:.2f}x ({'Eagle uses less' if memory_ratio < 1 else 'Flex uses less'})")