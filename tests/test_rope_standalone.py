"""
TDD Test: RoPE Fallback Implementation (Standalone)
Date: 2025-12-27
Iteration: 1-2

Standalone tests that don't require loading the full model.
These tests verify the core RoPE function works correctly.
"""

import pytest
import torch
import sys
import os

# Add the project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'turbodiffusion'))


# Copy the function directly for standalone testing
def apply_rotary_emb_torch(x, cos, sin, interleaved=True, inplace=False):
    """
    Pure PyTorch implementation of Rotary Position Embedding.
    Compatible with flash_attn's apply_rotary_emb interface.
    
    Args:
        x: Input tensor of shape [B, L, H, D]
        cos: Cosine values for rotation [L, D/2]
        sin: Sine values for rotation [L, D/2]
        interleaved: If True, use interleaved rotation (flash_attn style)
        inplace: Ignored, kept for API compatibility
    
    Returns:
        Rotated tensor of same shape as input
    """
    batch, seq_len, n_heads, head_dim = x.shape
    
    if interleaved:
        # Interleaved format: pairs of dimensions are rotated together
        x_reshaped = x.reshape(batch, seq_len, n_heads, head_dim // 2, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        
        # Reshape cos/sin for broadcasting: [L, D/2] -> [L, 1, D/2]
        cos = cos.view(seq_len, 1, head_dim // 2)
        sin = sin.view(seq_len, 1, head_dim // 2)
        
        # Apply rotation: [cos, -sin; sin, cos] @ [x1, x2]
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        
        # Interleave back to original format
        output = torch.stack([o1, o2], dim=-1).reshape(batch, seq_len, n_heads, head_dim)
    else:
        # Non-interleaved: first half and second half of dimensions
        d = head_dim // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        
        cos = cos.view(seq_len, 1, d)
        sin = sin.view(seq_len, 1, d)
        
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        output = torch.cat([o1, o2], dim=-1)
    
    return output


class TestRoPEStandalone:
    """Standalone test cases for pure PyTorch RoPE implementation."""
    
    def test_rope_output_shape(self):
        """验证输出形状正确 [B, L, H, D]"""
        batch, seq_len, n_heads, head_dim = 2, 16, 8, 64
        x = torch.randn(batch, seq_len, n_heads, head_dim)
        cos = torch.randn(seq_len, head_dim // 2)
        sin = torch.randn(seq_len, head_dim // 2)
        
        output = apply_rotary_emb_torch(x, cos, sin, interleaved=True)
        
        assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
        print(f"✅ Shape test passed: {output.shape}")
    
    def test_rope_dtype_float32(self):
        """验证 float32 数据类型保持不变"""
        x = torch.randn(2, 16, 8, 64, dtype=torch.float32)
        cos = torch.randn(16, 32, dtype=torch.float32)
        sin = torch.randn(16, 32, dtype=torch.float32)
        
        output = apply_rotary_emb_torch(x, cos, sin)
        
        assert output.dtype == x.dtype, f"Expected dtype {x.dtype}, got {output.dtype}"
        print(f"✅ float32 dtype test passed")
    
    def test_rope_dtype_float16(self):
        """验证 float16 数据类型保持不变"""
        x = torch.randn(2, 16, 8, 64, dtype=torch.float16)
        cos = torch.randn(16, 32, dtype=torch.float16)
        sin = torch.randn(16, 32, dtype=torch.float16)
        
        output = apply_rotary_emb_torch(x, cos, sin)
        
        assert output.dtype == x.dtype, f"Expected dtype {x.dtype}, got {output.dtype}"
        print(f"✅ float16 dtype test passed")
    
    def test_rope_dtype_bfloat16(self):
        """验证 bfloat16 数据类型保持不变"""
        x = torch.randn(2, 16, 8, 64, dtype=torch.bfloat16)
        cos = torch.randn(16, 32, dtype=torch.bfloat16)
        sin = torch.randn(16, 32, dtype=torch.bfloat16)
        
        output = apply_rotary_emb_torch(x, cos, sin)
        
        assert output.dtype == x.dtype, f"Expected dtype {x.dtype}, got {output.dtype}"
        print(f"✅ bfloat16 dtype test passed")
    
    def test_rope_interleaved_no_nan(self):
        """验证 interleaved 模式不产生 NaN"""
        x = torch.randn(1, 4, 2, 8)
        cos = torch.ones(4, 4)
        sin = torch.zeros(4, 4)
        
        output = apply_rotary_emb_torch(x, cos, sin, interleaved=True)
        
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
        print(f"✅ Interleaved mode NaN check passed")
    
    def test_rope_non_interleaved_no_nan(self):
        """验证 non-interleaved 模式不产生 NaN"""
        x = torch.randn(1, 4, 2, 8)
        cos = torch.ones(4, 4)
        sin = torch.zeros(4, 4)
        
        output = apply_rotary_emb_torch(x, cos, sin, interleaved=False)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any(), "Output contains NaN"
        print(f"✅ Non-interleaved mode NaN check passed")
    
    def test_rope_identity_rotation(self):
        """验证当 sin=0, cos=1 时输出接近输入"""
        x = torch.randn(2, 8, 4, 16)
        cos = torch.ones(8, 8)
        sin = torch.zeros(8, 8)
        
        output = apply_rotary_emb_torch(x, cos, sin, interleaved=True)
        
        # 当 sin=0, cos=1 时，应该是恒等变换
        assert torch.allclose(output, x, atol=1e-5), "Identity rotation failed"
        print(f"✅ Identity rotation test passed")
    
    def test_rope_different_batch_sizes(self):
        """验证不同 batch size 都能正常工作"""
        for batch in [1, 2, 4, 8]:
            x = torch.randn(batch, 16, 8, 64)
            cos = torch.randn(16, 32)
            sin = torch.randn(16, 32)
            
            output = apply_rotary_emb_torch(x, cos, sin)
            
            assert output.shape == x.shape
        print(f"✅ Different batch sizes test passed")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_rope_cuda(self):
        """验证 CUDA 张量也能正常工作"""
        x = torch.randn(2, 16, 8, 64, device='cuda')
        cos = torch.randn(16, 32, device='cuda')
        sin = torch.randn(16, 32, device='cuda')
        
        output = apply_rotary_emb_torch(x, cos, sin)
        
        assert output.device.type == 'cuda'
        assert output.shape == x.shape
        print(f"✅ CUDA tensor test passed")


if __name__ == "__main__":
    # 可以直接运行此文件进行测试
    test = TestRoPEStandalone()
    test.test_rope_output_shape()
    test.test_rope_dtype_float32()
    test.test_rope_dtype_float16()
    test.test_rope_dtype_bfloat16()
    test.test_rope_interleaved_no_nan()
    test.test_rope_non_interleaved_no_nan()
    test.test_rope_identity_rotation()
    test.test_rope_different_batch_sizes()
    if torch.cuda.is_available():
        test.test_rope_cuda()
    print("\n🎉 All standalone tests passed!")
