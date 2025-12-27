# TurboDiffusion - AI Agent 协作规范

**最后更新：** 2025-12-27 11:00 PST

---

## 🎯 项目目标

使 TurboDiffusion 能够在 RunPod 等容器化环境中无需编译额外依赖即可运行。

---

## 🔒 核心共识

### 环境分离

| 环境 | 职责 | GPU |
|------|------|-----|
| 本地开发机 (Mac) | 编写代码 | ❌ 无 |
| 测试服务器 (RunPod) | 部署 + 测试 | ✅ 2x RTX 4090 |

### 测试服务器连接

```bash
# SSH over TCP (推荐，支持 SCP & SFTP)
ssh root@157.157.221.29 -p 34430 -i ~/.ssh/id_ed25519
```

### 完整迭代原则

**编完代码只是迭代的 1/3 或更少！**

完整迭代 = 编码 → 部署 → 测试 → (修复 → 部署 → 测试)* → 通过 → 提交

---

## 📐 开发规范

### TDD 开发流程

```
🔴 红灯：写测试，确认失败
    ↓
🟢 绿灯：写代码，测试通过
    ↓
🔵 蓝灯：git commit
```

### 代码同步

```bash
# 本地 → 测试服务器
rsync -avz --exclude '.git' --exclude '__pycache__' \
    -e "ssh -i ~/.ssh/id_ed25519" \
    ./ 2ho75euf1tskl4-6441134f@ssh.runpod.io:/workspace/TurboDiffusion/
```

### Git 提交规范

```
<type>(<scope>): <description> (Iteration N)

Types: feat, fix, docs, test, config, refactor
```

---

## 🚫 禁止事项

1. ❌ 在本地运行 GPU 测试
2. ❌ 不测试就声称迭代完成
3. ❌ 跳过测试服务器部署
4. ❌ 一次修改太多功能点

---

## ✅ 必须事项

1. ✅ 每个迭代在测试服务器验证
2. ✅ 测试通过后才 git commit
3. ✅ 遵循小步快跑原则
4. ✅ 记录每个迭代的状态

---

## 📊 当前迭代进度

| 迭代 | 名称 | 状态 |
|------|------|------|
| 1 | 测试基础设施搭建 | ⬜ 待开始 |
| 2 | RoPE Fallback 函数实现 | ⬜ 待开始 |
| 3 | RoPE Import 逻辑修复 | ⬜ 待开始 |
| 4 | SLA 错误信息优化 | ⬜ 待开始 |
| 5 | 推理脚本默认值调整 | ⬜ 待开始 |
| 6 | Checkpoint 加载兼容性 | ⬜ 待开始 |
| 7 | RunPod 快速启动脚本 | ⬜ 待开始 |
| 8 | 文档更新 | ⬜ 待开始 |
| 9 | 端到端集成测试 | ⬜ 待开始 |

状态：⬜待开始 🔴红灯 🟢绿灯 🔵蓝灯 ✅完成
