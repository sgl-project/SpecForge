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
        
        # Calculate numerical differences
        abs_diff = torch.abs(output_flash - output_standard)
        max_abs_diff = torch.max(abs_diff).item()
        mean_abs_diff = torch.mean(abs_diff).item()
        rel_diff = abs_diff / (torch.abs(output_standard) + 1e-8)
        max_rel_diff = torch.max(rel_diff).item()
        
        print(f"Max absolute difference: {max_abs_diff:.6f}")
        print(f"Mean absolute difference: {mean_abs_diff:.6f}")
        print(f"Max relative difference: {max_rel_diff:.6f}")
        
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
        
        # Calculate gradient differences
        grad_abs_diff = torch.abs(grad_flash - grad_standard)
        grad_max_abs_diff = torch.max(grad_abs_diff).item()
        grad_mean_abs_diff = torch.mean(grad_abs_diff).item()
        grad_rel_diff = grad_abs_diff / (torch.abs(grad_standard) + 1e-8)
        grad_max_rel_diff = torch.max(grad_rel_diff).item()
        
        print(f"Gradient max absolute difference: {grad_max_abs_diff:.6f}")
        print(f"Gradient mean absolute difference: {grad_mean_abs_diff:.6f}")
        print(f"Gradient max relative difference: {grad_max_rel_diff:.6f}")
        
        # Compare parameter gradients
        flash_param_grads = {name: param.grad for name, param in attn_flash.named_parameters() if param.grad is not None}
        standard_param_grads = {name: param.grad for name, param in attn_standard.named_parameters() if param.grad is not None}
        
        for name in flash_param_grads.keys():
            if name in standard_param_grads:
                param_grad_diff = torch.abs(flash_param_grads[name] - standard_param_grads[name])
                param_max_diff = torch.max(param_grad_diff).item()
                print(f"Parameter {name} gradient max diff: {param_max_diff:.6f}")
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