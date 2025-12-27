# TurboDiffusion - Claude AI 开发指南

**最后更新：** 2025-12-27 11:00 PST

---

## 🔒 核心共识 (Core Conventions)

### 1. 开发环境分离原则

| 环境 | 用途 | 限制 |
|------|------|------|
| **本地开发机** | 只写代码 | ❌ 不做测试（无 GPU） |
| **测试服务器 (RunPod)** | 部署 + 测试 | ✅ 有 GPU，必须在此测试 |

### 2. 测试服务器信息

```bash
# SSH 登录命令 (TCP 直连方式，支持 SCP & SFTP)
ssh root@157.157.221.29 -p 34430 -i ~/.ssh/id_ed25519

# 首次连接自动接受密钥
ssh -o StrictHostKeyChecking=accept-new root@157.157.221.29 -p 34430 -i ~/.ssh/id_ed25519

# 项目路径
/workspace/TurboDiffusion

# GPU 配置
2x NVIDIA RTX 4090 (24GB each)
```

### 3. 完整迭代定义

⚠️ **编完代码 ≠ 迭代完成**

一个完整的迭代包含以下步骤：

```
┌─────────────────────────────────────────────────────────────┐
│                    完整迭代流程                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 本地编写代码 ────────────────────────────── (约 1/3)     │
│     ↓                                                       │
│  2. 同步到测试服务器 ─────────────────────────              │
│     ↓                                                       │
│  3. 在测试服务器运行测试 ─────────────────────              │
│     ↓                                                       │
│  ┌─ 4. 测试通过？─────────────────────────────              │
│  │   ├─ ✅ 是 → 提交代码 (git commit) → 迭代完成           │
│  │   └─ ❌ 否 → 返回步骤 1 修复代码                         │
│  └────────────────────────────────────────────              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4. TDD 开发节奏

- 🔴 **红灯**：编写测试用例，确认失败
- 🟢 **绿灯**：编写代码使测试通过
- 🔵 **蓝灯**：提交代码到 GitHub

---

## 📁 项目结构

```
TurboDiffusion/
├── turbodiffusion/           # 主要代码
│   ├── inference/            # 推理脚本
│   ├── rcm/                  # 模型定义
│   ├── SLA/                  # 稀疏注意力实现
│   └── ops/                  # CUDA 算子
├── tests/                    # 测试用例
├── scripts/                  # 工具脚本
├── docs/                     # 文档
└── checkpoints/              # 模型权重（仅测试服务器）
```

---

## 🚀 常用命令

### 本地开发

```bash
# 无需运行任何测试命令
# 专注于代码编写
```

### 同步到测试服务器

```bash
# 从本地同步代码到 RunPod
rsync -avz --exclude '.git' --exclude '__pycache__' \
    -e "ssh -i ~/.ssh/id_ed25519" \
    /Users/changhong/Documents/WCH@2025/Project/TurboDiffusion/ \
    2ho75euf1tskl4-6441134f@ssh.runpod.io:/workspace/TurboDiffusion/
```

### 测试服务器操作

```bash
# SSH 登录
ssh 2ho75euf1tskl4-6441134f@ssh.runpod.io -i ~/.ssh/id_ed25519

# 设置环境
cd /workspace/TurboDiffusion
export PYTHONPATH=$PYTHONPATH:/workspace/TurboDiffusion/turbodiffusion

# 运行测试
pytest tests/ -v

# 运行推理
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth \
    --attention_type sla \
    --quant_linear \
    --prompt "Your prompt"
```

---

## ⚠️ 注意事项

1. **不要在本地运行 GPU 相关代码** - 会失败
2. **每次修改代码后必须同步到测试服务器**
3. **测试必须在 RunPod 上进行**
4. **遵循小步快跑原则** - 每个迭代 2-3 个功能点

---

## 📋 当前任务

参见 `docs/TDD_IMPLEMENTATION_PLAN_2025-12-27.md`
