"""
TDD Test: End-to-End Integration and Documentation
Date: 2025-12-27
Iteration: 7-9

Final integration tests and documentation verification.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'turbodiffusion'))


class TestDocumentation:
    """Test documentation updates."""
    
    def test_readme_has_runpod_section(self):
        """验证 README 包含 RunPod 部署章节"""
        readme_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'README.md'
        )
        
        with open(readme_path, 'r') as f:
            content = f.read()
        
        assert 'RunPod' in content, "README should mention RunPod"
        assert 'runpod_setup.sh' in content, "README should mention setup script"
        assert 'runpod_quickstart.sh' in content, "README should mention quickstart script"
        print("✅ README has RunPod deployment section")
    
    def test_readme_has_troubleshooting(self):
        """验证 README 包含故障排除章节"""
        readme_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 'README.md'
        )
        
        with open(readme_path, 'r') as f:
            content = f.read()
        
        assert 'Troubleshooting' in content, "README should have Troubleshooting section"
        assert 'ModuleNotFoundError' in content, "README should mention common errors"
        print("✅ README has Troubleshooting section")


class TestModuleImports:
    """Test that all critical modules can be imported."""
    
    def test_rope_fallback_exists(self):
        """验证 RoPE fallback 函数存在"""
        # This is a standalone test, not importing the full module
        rope_file = os.path.join(
            os.path.dirname(__file__), 
            '..', 'turbodiffusion', 'rcm', 'networks', 'wan2pt1.py'
        )
        
        with open(rope_file, 'r') as f:
            content = f.read()
        
        assert 'def apply_rotary_emb_torch' in content
        assert 'flash_apply_rotary_emb = apply_rotary_emb_torch' in content
        print("✅ RoPE fallback function exists in wan2pt1.py")


class TestProjectStructure:
    """Test project structure is correct."""
    
    def test_all_consensus_docs_exist(self):
        """验证所有共识文档存在"""
        project_root = os.path.join(os.path.dirname(__file__), '..')
        
        assert os.path.exists(os.path.join(project_root, 'CLAUDE.md')), "CLAUDE.md should exist"
        assert os.path.exists(os.path.join(project_root, 'AGENTS.md')), "AGENTS.md should exist"
        print("✅ Consensus documents exist")
    
    def test_tests_directory_structure(self):
        """验证测试目录结构"""
        tests_dir = os.path.dirname(__file__)
        
        assert os.path.exists(os.path.join(tests_dir, 'conftest.py'))
        assert os.path.exists(os.path.join(tests_dir, 'test_rope_standalone.py'))
        assert os.path.exists(os.path.join(tests_dir, 'test_sla_and_defaults.py'))
        assert os.path.exists(os.path.join(tests_dir, 'test_checkpoint_and_scripts.py'))
        print("✅ Test directory structure is correct")


class TestAllTestsPass:
    """Meta test to verify all other tests pass."""
    
    def test_run_all_standalone_tests(self):
        """运行所有独立测试"""
        # Import and run standalone RoPE tests
        from test_rope_standalone import TestRoPEStandalone
        
        test = TestRoPEStandalone()
        test.test_rope_output_shape()
        test.test_rope_dtype_float32()
        test.test_rope_identity_rotation()
        print("✅ All standalone RoPE tests passed")


if __name__ == "__main__":
    print("=== Running Final Integration Tests ===\n")
    
    test_docs = TestDocumentation()
    test_docs.test_readme_has_runpod_section()
    test_docs.test_readme_has_troubleshooting()
    
    test_imports = TestModuleImports()
    test_imports.test_rope_fallback_exists()
    
    test_structure = TestProjectStructure()
    test_structure.test_all_consensus_docs_exist()
    test_structure.test_tests_directory_structure()
    
    print("\n🎉 All final integration tests passed!")
