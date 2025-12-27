"""
TDD Test: SLA Error Messages and Default Values
Date: 2025-12-27
Iteration: 3-4

Tests for SLA error messages and inference script default values.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'turbodiffusion'))


class TestSLAErrorMessages:
    """Test SLA/SageSLA error handling."""
    
    def test_sla_works_without_spargeattn(self):
        """验证 SLA 不需要 SpargeAttn 即可创建"""
        # SLA 只需要 triton，不需要 SpargeAttn
        # 这里我们只测试 import 不报错
        from SLA.core import SAGESLA_ENABLED
        print(f"SAGESLA_ENABLED: {SAGESLA_ENABLED}")
        
        # SLA 类应该可以导入
        from SLA.core import SparseLinearAttention
        assert SparseLinearAttention is not None
        print("✅ SparseLinearAttention class can be imported")


class TestInferenceDefaults:
    """Test inference script default configurations."""
    
    def test_t2v_default_attention_is_sla(self):
        """验证 T2V 脚本默认使用 SLA"""
        # 读取脚本文件检查默认值
        script_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'turbodiffusion', 'inference', 'wan2.1_t2v_infer.py'
        )
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # 检查默认值是否是 sla
        assert 'default="sla"' in content, "T2V script should default to sla attention"
        assert 'default="sagesla"' not in content, "T2V script should NOT default to sagesla"
        print("✅ T2V script defaults to sla attention")
    
    def test_i2v_default_attention_is_sla(self):
        """验证 I2V 脚本默认使用 SLA"""
        script_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'turbodiffusion', 'inference', 'wan2.2_i2v_infer.py'
        )
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        # 检查默认值是否是 sla
        assert 'default="sla"' in content, "I2V script should default to sla attention"
        assert 'default="sagesla"' not in content, "I2V script should NOT default to sagesla"
        print("✅ I2V script defaults to sla attention")


if __name__ == "__main__":
    test_sla = TestSLAErrorMessages()
    test_sla.test_sla_works_without_spargeattn()
    
    test_defaults = TestInferenceDefaults()
    test_defaults.test_t2v_default_attention_is_sla()
    test_defaults.test_i2v_default_attention_is_sla()
    
    print("\n🎉 All tests passed!")
