"""
TDD Test: Checkpoint Loading and Scripts
Date: 2025-12-27
Iteration: 5-6

Tests for checkpoint loading compatibility and quick start scripts.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'turbodiffusion'))


class TestCheckpointLoading:
    """Test checkpoint loading with mismatched keys."""
    
    def test_strict_false_in_create_model(self):
        """验证 create_model 函数使用 strict=False"""
        script_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'turbodiffusion', 'inference', 'modify_model.py'
        )
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        assert 'strict=False' in content, "modify_model.py should use strict=False"
        print("✅ modify_model.py uses strict=False")
    
    def test_warning_for_mismatched_keys(self):
        """验证有警告日志代码"""
        script_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'turbodiffusion', 'inference', 'modify_model.py'
        )
        
        with open(script_path, 'r') as f:
            content = f.read()
        
        assert 'unexpected_keys' in content, "Should handle unexpected_keys"
        assert 'missing_keys' in content, "Should handle missing_keys"
        print("✅ modify_model.py handles mismatched keys with warnings")


class TestQuickStartScripts:
    """Test quick start scripts exist and are executable."""
    
    def test_runpod_setup_exists(self):
        """验证 runpod_setup.sh 存在"""
        script_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'scripts', 'runpod_setup.sh'
        )
        assert os.path.exists(script_path), "runpod_setup.sh should exist"
        print("✅ runpod_setup.sh exists")
    
    def test_runpod_quickstart_exists(self):
        """验证 runpod_quickstart.sh 存在"""
        script_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'scripts', 'runpod_quickstart.sh'
        )
        assert os.path.exists(script_path), "runpod_quickstart.sh should exist"
        print("✅ runpod_quickstart.sh exists")
    
    def test_scripts_have_shebang(self):
        """验证脚本有正确的 shebang"""
        for script_name in ['runpod_setup.sh', 'runpod_quickstart.sh']:
            script_path = os.path.join(
                os.path.dirname(__file__), 
                '..', 'scripts', script_name
            )
            with open(script_path, 'r') as f:
                first_line = f.readline()
            assert first_line.startswith('#!/bin/bash'), f"{script_name} should have bash shebang"
        print("✅ Both scripts have correct shebang")
    
    def test_quickstart_uses_sla(self):
        """验证 quickstart 脚本使用 sla attention"""
        script_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'scripts', 'runpod_quickstart.sh'
        )
        with open(script_path, 'r') as f:
            content = f.read()
        
        assert '--attention_type sla' in content, "quickstart should use sla attention"
        print("✅ quickstart script uses sla attention")


if __name__ == "__main__":
    test_ckpt = TestCheckpointLoading()
    test_ckpt.test_strict_false_in_create_model()
    test_ckpt.test_warning_for_mismatched_keys()
    
    test_scripts = TestQuickStartScripts()
    test_scripts.test_runpod_setup_exists()
    test_scripts.test_runpod_quickstart_exists()
    test_scripts.test_scripts_have_shebang()
    test_scripts.test_quickstart_uses_sla()
    
    print("\n🎉 All tests passed!")
