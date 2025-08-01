import unittest
import torch
import torch.nn.functional as F
from unittest.mock import patch
from transformers import LlamaConfig
import numpy as np

from specforge.modeling.draft.llama3_eagle import (
    LlamaAttention,
    FLASH_ATTENTION_AVAILABLE,
    FlashAttnVarlenFunc,
)


class TestFlashAttentionIntegration(unittest.TestCase):
    """Test Flash Attention integration vs standard attention implementation"""

    def setUp(self):
        """Set up test configurations and models"""
        config_dict = {
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-5,
            "rope_scaling": None,
            "vocab_size": 32000,
            "draft_vocab_size": 16000,
        }
        self.config = LlamaConfig(**config_dict)
        
        # Test parameters
        self.batch_size = 2
        self.seq_len = 256  # Large enough to trigger Flash Attention
        self.hidden_size = self.config.hidden_size
        
        # Set device and dtype
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32  # Use fp32 for models, convert to fp16 internally for Flash Attention
        
        # Skip test if Flash Attention is not available or CUDA is not available
        if not FLASH_ATTENTION_AVAILABLE:
            self.skipTest("Flash Attention is not available")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available, required for Flash Attention")

    def create_attention_inputs(self):
        """Create consistent input tensors for attention testing"""
        torch.manual_seed(42)  # For reproducibility
        
        # Create input hidden states (concatenated input_emb and hidden_states)
        hidden_states = torch.randn(
            self.batch_size, self.seq_len, self.hidden_size * 2,
            device=self.device, dtype=self.dtype, requires_grad=True
        )
        
        # Create cache_hidden for enabling cache-based attention
        cache_hidden = [[], []]  # Empty cache to start
        
        # Create attention mask (causal mask)
        attention_mask = torch.full(
            (self.batch_size, 1, self.seq_len, self.seq_len),
            float('-inf'), device=self.device, dtype=self.dtype
        )
        attention_mask = torch.triu(attention_mask, diagonal=1)
        
        # Create position ids
        position_ids = torch.arange(self.seq_len, device=self.device).unsqueeze(0).repeat(self.batch_size, 1)
        
        return hidden_states, cache_hidden, attention_mask, position_ids

    def test_flash_attention_vs_standard_forward(self):
        """Test forward pass comparison between Flash Attention and standard attention"""
        print("\n=== Testing Forward Pass ===")
        
        # Create two identical attention layers
        attn_flash = LlamaAttention(self.config).to(self.device, dtype=self.dtype)
        attn_standard = LlamaAttention(self.config).to(self.device, dtype=self.dtype)
        
        # Copy weights to ensure identical parameters
        attn_standard.load_state_dict(attn_flash.state_dict())
        
        # Create inputs
        hidden_states, cache_hidden_flash, attention_mask, position_ids = self.create_attention_inputs()
        cache_hidden_standard = [[], []]  # Separate cache for standard attention
        
        # Force Flash Attention to be used for first model
        with patch('specforge.modeling.draft.llama3_eagle.FLASH_ATTENTION_AVAILABLE', True):
            # Ensure conditions for Flash Attention are met
            self.assertGreaterEqual(self.seq_len, 128)
            self.assertGreaterEqual(self.batch_size * self.seq_len, 512)
            
            output_flash = attn_flash(
                hidden_states=hidden_states.clone(),
                cache_hidden=cache_hidden_flash,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        
        # Force standard attention for second model
        with patch('specforge.modeling.draft.llama3_eagle.FLASH_ATTENTION_AVAILABLE', False):
            output_standard = attn_standard(
                hidden_states=hidden_states.clone(),
                cache_hidden=cache_hidden_standard,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        
        # Compare outputs
        print(f"Flash Attention output shape: {output_flash.shape}")
        print(f"Standard output shape: {output_standard.shape}")
        
        # Check shapes are identical
        self.assertEqual(output_flash.shape, output_standard.shape)
        
        # Calculate comprehensive numerical differences
        abs_diff = torch.abs(output_flash - output_standard)
        rel_diff = abs_diff / (torch.abs(output_standard) + 1e-8)
        
        # Detailed statistics
        print(f"\n=== COMPREHENSIVE OUTPUT COMPARISON ===")
        print(f"Flash Attention output stats:")
        print(f"  Min: {torch.min(output_flash).item():.6f}, Max: {torch.max(output_flash).item():.6f}")
        print(f"  Mean: {torch.mean(output_flash).item():.6f}, Std: {torch.std(output_flash).item():.6f}")
        
        print(f"Standard Attention output stats:")
        print(f"  Min: {torch.min(output_standard).item():.6f}, Max: {torch.max(output_standard).item():.6f}")
        print(f"  Mean: {torch.mean(output_standard).item():.6f}, Std: {torch.std(output_standard).item():.6f}")
        
        print(f"\nAbsolute Difference Stats:")
        print(f"  Min: {torch.min(abs_diff).item():.6f}, Max: {torch.max(abs_diff).item():.6f}")
        print(f"  Mean: {torch.mean(abs_diff).item():.6f}, Std: {torch.std(abs_diff).item():.6f}")
        print(f"  Median: {torch.median(abs_diff).item():.6f}")
        print(f"  95th percentile: {torch.quantile(abs_diff, 0.95).item():.6f}")
        print(f"  99th percentile: {torch.quantile(abs_diff, 0.99).item():.6f}")
        
        print(f"\nRelative Difference Stats:")
        print(f"  Min: {torch.min(rel_diff).item():.6f}, Max: {torch.max(rel_diff).item():.6f}")
        print(f"  Mean: {torch.mean(rel_diff).item():.6f}, Std: {torch.std(rel_diff).item():.6f}")
        print(f"  Median: {torch.median(rel_diff).item():.6f}")
        
        # Count differences by magnitude
        total_elements = abs_diff.numel()
        diff_1e6 = torch.sum(abs_diff > 1e-6).item()
        diff_1e5 = torch.sum(abs_diff > 1e-5).item()
        diff_1e4 = torch.sum(abs_diff > 1e-4).item()
        diff_1e3 = torch.sum(abs_diff > 1e-3).item()
        diff_1e2 = torch.sum(abs_diff > 1e-2).item()
        
        print(f"\nDifference Distribution (out of {total_elements} elements):")
        print(f"  > 1e-6: {diff_1e6} ({100*diff_1e6/total_elements:.2f}%)")
        print(f"  > 1e-5: {diff_1e5} ({100*diff_1e5/total_elements:.2f}%)")
        print(f"  > 1e-4: {diff_1e4} ({100*diff_1e4/total_elements:.2f}%)")
        print(f"  > 1e-3: {diff_1e3} ({100*diff_1e3/total_elements:.2f}%)")
        print(f"  > 1e-2: {diff_1e2} ({100*diff_1e2/total_elements:.2f}%)")
        
        # Sample values from different regions
        print(f"\n=== SAMPLE VALUES COMPARISON ===")
        batch_samples = [0, min(1, self.batch_size-1)]
        seq_samples = [0, self.seq_len//4, self.seq_len//2, -1]
        hidden_samples = [0, self.hidden_size//4, self.hidden_size//2, -1]
        
        for b in batch_samples:
            for s in seq_samples:
                for h in hidden_samples:
                    if s == -1: s = self.seq_len - 1
                    if h == -1: h = self.hidden_size - 1
                    flash_val = output_flash[b, s, h].item()
                    std_val = output_standard[b, s, h].item()
                    diff_val = abs(flash_val - std_val)
                    print(f"  [{b},{s},{h}]: Flash={flash_val:.6f}, Std={std_val:.6f}, Diff={diff_val:.6f}")
        
        max_abs_diff = torch.max(abs_diff).item()
        mean_abs_diff = torch.mean(abs_diff).item()
        max_rel_diff = torch.max(rel_diff).item()
        
        # The outputs should be close (fp16 has lower precision than fp32)
        self.assertLess(max_abs_diff, 1e-2, "Flash Attention output differs too much from standard attention")
        self.assertLess(mean_abs_diff, 1e-3, "Mean difference too large between implementations")

    def test_flash_attention_vs_standard_backward(self):
        """Test backward pass comparison between Flash Attention and standard attention"""
        print("\n=== Testing Backward Pass ===")
        
        # Create two identical attention layers
        attn_flash = LlamaAttention(self.config).to(self.device, dtype=self.dtype)
        attn_standard = LlamaAttention(self.config).to(self.device, dtype=self.dtype)
        
        # Copy weights to ensure identical parameters
        attn_standard.load_state_dict(attn_flash.state_dict())
        
        # Create inputs with gradient tracking
        hidden_states_flash, cache_hidden_flash, attention_mask, position_ids = self.create_attention_inputs()
        hidden_states_standard = hidden_states_flash.clone().detach().requires_grad_(True)
        cache_hidden_standard = [[], []]
        
        # Forward pass with Flash Attention
        with patch('specforge.modeling.draft.llama3_eagle.FLASH_ATTENTION_AVAILABLE', True):
            output_flash = attn_flash(
                hidden_states=hidden_states_flash,
                cache_hidden=cache_hidden_flash,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        
        # Forward pass with standard attention
        with patch('specforge.modeling.draft.llama3_eagle.FLASH_ATTENTION_AVAILABLE', False):
            output_standard = attn_standard(
                hidden_states=hidden_states_standard,
                cache_hidden=cache_hidden_standard,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
        
        # Create identical loss for backward pass
        loss_flash = torch.sum(output_flash ** 2)
        loss_standard = torch.sum(output_standard ** 2)
        
        # Backward pass
        loss_flash.backward()
        loss_standard.backward()
        
        # Compare gradients of input hidden states
        grad_flash = hidden_states_flash.grad
        grad_standard = hidden_states_standard.grad
        
        print(f"Flash Attention gradient shape: {grad_flash.shape}")
        print(f"Standard gradient shape: {grad_standard.shape}")
        
        # Check gradient shapes
        self.assertEqual(grad_flash.shape, grad_standard.shape)
        
        # Calculate comprehensive gradient differences
        grad_abs_diff = torch.abs(grad_flash - grad_standard)
        grad_rel_diff = grad_abs_diff / (torch.abs(grad_standard) + 1e-8)
        
        # Detailed gradient statistics
        print(f"\n=== COMPREHENSIVE GRADIENT COMPARISON ===")
        print(f"Flash Attention gradient stats:")
        print(f"  Min: {torch.min(grad_flash).item():.6f}, Max: {torch.max(grad_flash).item():.6f}")
        print(f"  Mean: {torch.mean(grad_flash).item():.6f}, Std: {torch.std(grad_flash).item():.6f}")
        
        print(f"Standard Attention gradient stats:")
        print(f"  Min: {torch.min(grad_standard).item():.6f}, Max: {torch.max(grad_standard).item():.6f}")
        print(f"  Mean: {torch.mean(grad_standard).item():.6f}, Std: {torch.std(grad_standard).item():.6f}")
        
        print(f"\nGradient Absolute Difference Stats:")
        print(f"  Min: {torch.min(grad_abs_diff).item():.6f}, Max: {torch.max(grad_abs_diff).item():.6f}")
        print(f"  Mean: {torch.mean(grad_abs_diff).item():.6f}, Std: {torch.std(grad_abs_diff).item():.6f}")
        print(f"  Median: {torch.median(grad_abs_diff).item():.6f}")
        print(f"  95th percentile: {torch.quantile(grad_abs_diff, 0.95).item():.6f}")
        print(f"  99th percentile: {torch.quantile(grad_abs_diff, 0.99).item():.6f}")
        
        print(f"\nGradient Relative Difference Stats:")
        print(f"  Min: {torch.min(grad_rel_diff).item():.6f}, Max: {torch.max(grad_rel_diff).item():.6f}")
        print(f"  Mean: {torch.mean(grad_rel_diff).item():.6f}, Std: {torch.std(grad_rel_diff).item():.6f}")
        print(f"  Median: {torch.median(grad_rel_diff).item():.6f}")
        
        # Count gradient differences by magnitude
        total_grad_elements = grad_abs_diff.numel()
        grad_diff_1e6 = torch.sum(grad_abs_diff > 1e-6).item()
        grad_diff_1e5 = torch.sum(grad_abs_diff > 1e-5).item()
        grad_diff_1e4 = torch.sum(grad_abs_diff > 1e-4).item()
        grad_diff_1e3 = torch.sum(grad_abs_diff > 1e-3).item()
        grad_diff_1e2 = torch.sum(grad_abs_diff > 1e-2).item()
        
        print(f"\nGradient Difference Distribution (out of {total_grad_elements} elements):")
        print(f"  > 1e-6: {grad_diff_1e6} ({100*grad_diff_1e6/total_grad_elements:.2f}%)")
        print(f"  > 1e-5: {grad_diff_1e5} ({100*grad_diff_1e5/total_grad_elements:.2f}%)")
        print(f"  > 1e-4: {grad_diff_1e4} ({100*grad_diff_1e4/total_grad_elements:.2f}%)")
        print(f"  > 1e-3: {grad_diff_1e3} ({100*grad_diff_1e3/total_grad_elements:.2f}%)")
        print(f"  > 1e-2: {grad_diff_1e2} ({100*grad_diff_1e2/total_grad_elements:.2f}%)")
        
        # Sample gradient values from different regions  
        print(f"\n=== GRADIENT SAMPLE VALUES COMPARISON ===")
        batch_samples = [0, min(1, self.batch_size-1)]
        seq_samples = [0, self.seq_len//4, self.seq_len//2, -1]
        hidden_samples = [0, self.hidden_size//4, self.hidden_size//2, -1]
        
        for b in batch_samples:
            for s in seq_samples:
                for h in hidden_samples:
                    if s == -1: s = self.seq_len - 1
                    if h == -1: h = self.hidden_size*2 - 1  # Note: grad has hidden_size*2
                    flash_grad_val = grad_flash[b, s, h].item()
                    std_grad_val = grad_standard[b, s, h].item()
                    diff_grad_val = abs(flash_grad_val - std_grad_val)
                    print(f"  Grad[{b},{s},{h}]: Flash={flash_grad_val:.6f}, Std={std_grad_val:.6f}, Diff={diff_grad_val:.6f}")
        
        grad_max_abs_diff = torch.max(grad_abs_diff).item()
        grad_mean_abs_diff = torch.mean(grad_abs_diff).item()
        grad_max_rel_diff = torch.max(grad_rel_diff).item()
        
        # Compare parameter gradients comprehensively
        print(f"\n=== PARAMETER GRADIENTS COMPARISON ===")
        flash_param_grads = {name: param.grad for name, param in attn_flash.named_parameters() if param.grad is not None}
        standard_param_grads = {name: param.grad for name, param in attn_standard.named_parameters() if param.grad is not None}
        
        for name in flash_param_grads.keys():
            if name in standard_param_grads:
                flash_grad = flash_param_grads[name]
                std_grad = standard_param_grads[name]
                param_grad_diff = torch.abs(flash_grad - std_grad)
                param_rel_diff = param_grad_diff / (torch.abs(std_grad) + 1e-8)
                
                print(f"\nParameter '{name}' gradient comparison:")
                print(f"  Shape: {flash_grad.shape}")
                print(f"  Flash grad - Min: {torch.min(flash_grad).item():.6f}, Max: {torch.max(flash_grad).item():.6f}")
                print(f"  Flash grad - Mean: {torch.mean(flash_grad).item():.6f}, Std: {torch.std(flash_grad).item():.6f}")
                print(f"  Std grad   - Min: {torch.min(std_grad).item():.6f}, Max: {torch.max(std_grad).item():.6f}")
                print(f"  Std grad   - Mean: {torch.mean(std_grad).item():.6f}, Std: {torch.std(std_grad).item():.6f}")
                print(f"  Abs diff   - Min: {torch.min(param_grad_diff).item():.6f}, Max: {torch.max(param_grad_diff).item():.6f}")
                print(f"  Abs diff   - Mean: {torch.mean(param_grad_diff).item():.6f}, Median: {torch.median(param_grad_diff).item():.6f}")
                print(f"  Rel diff   - Mean: {torch.mean(param_rel_diff).item():.6f}, Max: {torch.max(param_rel_diff).item():.6f}")
                
                # Count significant differences
                param_total = param_grad_diff.numel()
                param_diff_1e4 = torch.sum(param_grad_diff > 1e-4).item()
                param_diff_1e3 = torch.sum(param_grad_diff > 1e-3).item()
                param_diff_1e2 = torch.sum(param_grad_diff > 1e-2).item()
                print(f"  Differences > 1e-4: {param_diff_1e4}/{param_total} ({100*param_diff_1e4/param_total:.2f}%)")
                print(f"  Differences > 1e-3: {param_diff_1e3}/{param_total} ({100*param_diff_1e3/param_total:.2f}%)")
                print(f"  Differences > 1e-2: {param_diff_1e2}/{param_total} ({100*param_diff_1e2/param_total:.2f}%)")
                
                param_max_diff = torch.max(param_grad_diff).item()
                self.assertLess(param_max_diff, 1e-1, f"Parameter {name} gradients differ too much")
        
        # The gradients should be close (fp16 has lower precision and different computation paths)
        self.assertLess(grad_max_abs_diff, 1e-1, "Flash Attention gradients differ too much from standard attention")
        self.assertLess(grad_mean_abs_diff, 1e-2, "Mean gradient difference too large between implementations")

    def test_flash_attention_conditions(self):
        """Test the conditions under which Flash Attention is used"""
        print("\n=== Testing Flash Attention Conditions ===")
        
        attn = LlamaAttention(self.config).to(self.device, dtype=self.dtype)
        
        # Test with small sequence length (should use standard attention)
        small_seq_len = 64  # Below threshold
        hidden_states_small = torch.randn(
            self.batch_size, small_seq_len, self.hidden_size * 2,
            device=self.device, dtype=self.dtype
        )
        cache_hidden_small = [[], []]
        attention_mask_small = torch.full(
            (self.batch_size, 1, small_seq_len, small_seq_len),
            float('-inf'), device=self.device, dtype=self.dtype
        )
        attention_mask_small = torch.triu(attention_mask_small, diagonal=1)
        position_ids_small = torch.arange(small_seq_len, device=self.device).unsqueeze(0).repeat(self.batch_size, 1)
        
        # This should work with standard attention (no Flash Attention due to small size)
        with patch('specforge.modeling.draft.llama3_eagle.FLASH_ATTENTION_AVAILABLE', True):
            output_small = attn(
                hidden_states=hidden_states_small,
                cache_hidden=cache_hidden_small,
                attention_mask=attention_mask_small,
                position_ids=position_ids_small,
            )
        
        expected_shape = (self.batch_size, small_seq_len, self.hidden_size)
        self.assertEqual(output_small.shape, expected_shape)
        print(f"Small sequence test passed: {output_small.shape}")

    def test_empty_sequence_handling(self):
        """Test handling of empty sequences"""
        print("\n=== Testing Empty Sequence Handling ===")
        
        attn = LlamaAttention(self.config).to(self.device, dtype=self.dtype)
        
        # Test with zero sequence length
        empty_seq_len = 0
        hidden_states_empty = torch.randn(
            self.batch_size, empty_seq_len, self.hidden_size * 2,
            device=self.device, dtype=self.dtype
        )
        cache_hidden_empty = [[], []]
        attention_mask_empty = torch.empty(
            (self.batch_size, 1, empty_seq_len, empty_seq_len),
            device=self.device, dtype=self.dtype
        )
        position_ids_empty = torch.empty(
            (self.batch_size, empty_seq_len), 
            device=self.device, dtype=torch.long
        )
        
        # This should handle empty sequences gracefully
        output_empty = attn(
            hidden_states=hidden_states_empty,
            cache_hidden=cache_hidden_empty,
            attention_mask=attention_mask_empty,
            position_ids=position_ids_empty,
        )
        
        expected_shape = (self.batch_size, empty_seq_len, self.hidden_size)
        self.assertEqual(output_empty.shape, expected_shape)
        print(f"Empty sequence test passed: {output_empty.shape}")


if __name__ == "__main__":
    # Set up test suite
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestFlashAttentionIntegration))
    
    # Run tests with high verbosity
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100:.1f}%") 