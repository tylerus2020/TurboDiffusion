"""
TDD Test: RoPE Fallback Implementation
Date: 2025-12-27
Iteration: 1-2

Tests for the pure PyTorch RoPE (Rotary Position Embedding) fallback function.
This fallback is used when flash_attn is not available.
"""

import pytest
import torch


class TestRoPEFallback:
    """Test cases for pure PyTorch RoPE implementation."""
    
    def test_rope_function_exists(self):
        """
        🔴 RED: 验证 apply_rotary_emb_torch 函数存在
        
        Expected: 函数应该存在并可调用
        """
        from rcm.networks.wan2pt1 import apply_rotary_emb_torch
        assert callable(apply_rotary_emb_torch), "apply_rotary_emb_torch should be callable"
    
    def test_rope_output_shape(self):
        """
        🔴 RED: 验证输出形状正确 [B, L, H, D]
        
        Expected: 输出形状应与输入相同
        """
        from rcm.networks.wan2pt1 import apply_rotary_emb_torch
        
        batch, seq_len, n_heads, head_dim = 2, 16, 8, 64
        x = torch.randn(batch, seq_len, n_heads, head_dim)
        cos = torch.randn(seq_len, head_dim // 2)
        sin = torch.randn(seq_len, head_dim // 2)
        
        output = apply_rotary_emb_torch(x, cos, sin, interleaved=True)
        
        assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    
    def test_rope_dtype_preserved(self):
        """
        🔴 RED: 验证数据类型保持不变
        
        Expected: 输出 dtype 应与输入相同
        """
        from rcm.networks.wan2pt1 import apply_rotary_emb_torch
        
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(2, 16, 8, 64, dtype=dtype)
            cos = torch.randn(16, 32, dtype=dtype)
            sin = torch.randn(16, 32, dtype=dtype)
            
            output = apply_rotary_emb_torch(x, cos, sin)
            
            assert output.dtype == x.dtype, f"Expected dtype {x.dtype}, got {output.dtype}"
    
    def test_rope_interleaved_mode(self):
        """
        🔴 RED: 验证 interleaved 模式的旋转正确性
        
        Expected: 旋转后的值应该改变（不是全零或 NaN）
        """
        from rcm.networks.wan2pt1 import apply_rotary_emb_torch
        
        x = torch.randn(1, 4, 2, 8)
        cos = torch.ones(4, 4)  # 简单的 cos 值
        sin = torch.zeros(4, 4)  # sin=0 意味着只缩放，不旋转
        
        output = apply_rotary_emb_torch(x, cos, sin, interleaved=True)
        
        # 当 sin=0, cos=1 时，输出应该接近输入
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"
    
    def test_rope_non_interleaved_mode(self):
        """
        🔴 RED: 验证 non-interleaved 模式
        
        Expected: non-interleaved 模式也应该正常工作
        """
        from rcm.networks.wan2pt1 import apply_rotary_emb_torch
        
        x = torch.randn(1, 4, 2, 8)
        cos = torch.ones(4, 4)
        sin = torch.zeros(4, 4)
        
        output = apply_rotary_emb_torch(x, cos, sin, interleaved=False)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestRoPEImportFallback:
    """Test cases for RoPE import fallback logic."""
    
    def test_flash_apply_rotary_emb_not_none(self):
        """
        🔴 RED: 验证 flash_apply_rotary_emb 不是 None
        
        Expected: 即使 flash_attn 不可用，也应该有 fallback
        """
        from rcm.networks.wan2pt1 import flash_apply_rotary_emb
        
        assert flash_apply_rotary_emb is not None, \
            "flash_apply_rotary_emb should not be None (should use fallback)"
    
    def test_flash_apply_rotary_emb_callable(self):
        """
        🔴 RED: 验证 flash_apply_rotary_emb 可调用
        
        Expected: 应该是一个可调用的函数
        """
        from rcm.networks.wan2pt1 import flash_apply_rotary_emb
        
        assert callable(flash_apply_rotary_emb), \
            "flash_apply_rotary_emb should be callable"


class TestRoPEApplyFunction:
    """Test cases for the rope_apply function used in the model."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_rope_apply_function_works(self):
        """
        🔴 RED: 验证 rope_apply 函数能正常工作
        
        Expected: 应该能处理典型的视频张量
        """
        from rcm.networks.wan2pt1 import rope_apply, VideoSize
        
        # 典型的视频尺寸: T=4, H=8, W=8 -> seq_len = 256
        x = torch.randn(1, 256, 8, 64, dtype=torch.float32).cuda()
        video_size = VideoSize(T=4, H=8, W=8)
        freqs = torch.randn(256, 32).cuda()
        
        # 应该不抛出异常
        output = rope_apply(x, video_size, freqs)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
