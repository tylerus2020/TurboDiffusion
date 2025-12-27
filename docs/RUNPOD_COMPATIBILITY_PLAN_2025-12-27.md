# TurboDiffusion RunPod 兼容性修改计划

**日期：** 2025-12-27 (美国西部时间)  
**作者：** TurboDiffusion 开发团队  
**状态：** 待执行

---

## 一、问题分析总结

根据昨天的调试日志和现有代码审查，问题的核心在于：

### 1. 核心依赖缺失

| 依赖 | 用途 | 问题 |
|------|------|------|
| `flash-attn` | 提供 `apply_rotary_emb` 函数实现 RoPE | 需要从源码编译，RunPod 磁盘空间不足 |
| `SpargeAttn` | 提供 `SageSLA` 注意力加速 | 需要从源码编译，RunPod 环境复杂 |

### 2. 代码缺陷

- **文件：** `turbodiffusion/rcm/networks/wan2pt1.py` (第26-30行)
- **问题：** 当 `flash_attn` 导入失败时，将 `flash_apply_rotary_emb` 设为 `None`，但后续代码仍直接调用，导致 `TypeError: 'NoneType' object is not callable`

```python
# 当前代码（第26-30行）
try:
    from flash_attn.layers.rotary import apply_rotary_emb as flash_apply_rotary_emb
except ImportError:
    flash_apply_rotary_emb = None  # ← 这里是None
    print("flash_attn is not installed.")
```

- **调用位置：** 第226行 `rope_apply` 函数中

```python
rotated = flash_apply_rotary_emb(x.to(torch.float32), cos, sin, interleaved=True, inplace=False)  # ← 这里调用None会崩溃
```

### 3. Checkpoint 兼容性

- 量化版 checkpoint (`TurboWan2.1-T2V-1.3B-480P-quant.pth`) 包含 SLA 特有的权重（`local_attn.proj_l.*`）
- 必须使用 `sla` 或 `sagesla` attention 类型才能正确加载
- 使用 `strict=False` 加载可以忽略不兼容的键

---

## 二、修改战略

采用 **"修改代码适应环境"** 的策略，而非 **"改变环境适应代码"**。

### 核心原则

1. **正确性 > 性能**：先跑通，再优化
2. **最小侵入性**：只修改必要的代码，不重构架构
3. **Fallback 机制**：为每个可选依赖提供纯 PyTorch 替代实现

---

## 三、分阶段实施计划

### 阶段 1：RoPE Fallback（核心修复）⭐ 最高优先级

**目标文件：** `turbodiffusion/rcm/networks/wan2pt1.py`

**修改内容：**

#### 1.1 在文件头部添加 PyTorch 原生 RoPE 实现（约第31行后）

```python
# ============================================
# Pure PyTorch RoPE fallback implementation
# ============================================
def apply_rotary_emb_torch(x, cos, sin, interleaved=True, inplace=False):
    """
    Pure PyTorch implementation of Rotary Position Embedding.
    Compatible with flash_attn's interface.
    
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
        # Interleaved format: [x0, x1, x2, x3, ...] -> pairs for rotation
        x_reshaped = x.reshape(batch, seq_len, n_heads, head_dim // 2, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]
        
        # Reshape cos/sin for broadcasting
        cos = cos.view(seq_len, 1, head_dim // 2)
        sin = sin.view(seq_len, 1, head_dim // 2)
        
        # Apply rotation
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        
        # Interleave back
        output = torch.stack([o1, o2], dim=-1).reshape(batch, seq_len, n_heads, head_dim)
    else:
        # Non-interleaved: first half and second half
        d = head_dim // 2
        x1 = x[..., :d]
        x2 = x[..., d:]
        
        cos = cos.view(seq_len, 1, d)
        sin = sin.view(seq_len, 1, d)
        
        o1 = x1 * cos - x2 * sin
        o2 = x1 * sin + x2 * cos
        output = torch.cat([o1, o2], dim=-1)
    
    return output
```

#### 1.2 修改 flash_attn 导入逻辑（第26-30行）

```python
try:
    from flash_attn.layers.rotary import apply_rotary_emb as flash_apply_rotary_emb
except ImportError:
    flash_apply_rotary_emb = apply_rotary_emb_torch
    print("Warning: flash_attn not found, using pure PyTorch RoPE fallback. This may be slower.")
```

**为什么这样改：**
- `apply_rotary_emb_torch` 是纯 PyTorch 实现，无需任何编译安装
- 保持与 `flash_attn` 相同的函数签名，无需修改调用代码
- 性能会稍慢，但对于推理 Demo 完全可接受

---

### 阶段 2：SLA Fallback（确保 SLA 无需 SpargeAttn）

**目标文件：** `turbodiffusion/SLA/core.py`

**分析：**
- `SparseLinearAttention`（SLA）：使用 `._attention`（triton kernel），不依赖 SpargeAttn ✅
- `SageSparseLinearAttention`（SageSLA）：依赖 SpargeAttn ❌

**修改内容：**

在 `SageSparseLinearAttention.__init__` 中添加更友好的错误信息（第135行）

```python
assert SAGESLA_ENABLED, (
    "SageSLA requires SpargeAttn library. Install it with:\n"
    "  pip install git+https://github.com/thu-ml/SpargeAttn.git --no-build-isolation\n"
    "Or use --attention_type sla instead (slower but no extra dependencies)."
)
```

**说明：** SLA 模式本身不需要 SpargeAttn，只要 triton 可用即可。我们确保用户使用 `--attention_type sla` 时能正常运行。

---

### 阶段 3：推理脚本默认值调整

**目标文件：** `turbodiffusion/inference/wan2.1_t2v_infer.py`

**修改内容：**（第50行）

```python
# 修改前
parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"], default="sagesla", ...)

# 修改后：将默认值改为 sla，避免 SpargeAttn 依赖
parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"], default="sla", ...)
```

**为什么：**
- `sagesla` 需要 SpargeAttn，不适合通用环境
- `sla` 使用 triton 实现，更通用
- 用户可以显式指定 `--attention_type sagesla` 如果环境支持

---

### 阶段 4：Checkpoint 加载兼容性

**目标文件：** `turbodiffusion/inference/modify_model.py`

**分析当前代码：**（第138行）

```python
net.load_state_dict(state_dict, assign=True)  # strict=True 是默认值
```

**修改内容：**

```python
# 修改为 strict=False 以允许加载包含 SLA 权重的 checkpoint
net.load_state_dict(state_dict, assign=True, strict=False)
```

**注意：** 根据昨天的日志，当使用 `original` attention 加载 SLA checkpoint 时会有不匹配的 keys。使用 `strict=False` 可以忽略这些，但最佳做法仍是使用正确的 attention 类型。

---

### 阶段 5：创建 RunPod 快速启动脚本

**新建文件：** `scripts/runpod_quickstart.sh`

```bash
#!/bin/bash
# TurboDiffusion RunPod Quick Start Script
# Date: 2025-12-27

set -e

echo "=== TurboDiffusion RunPod Quick Start ==="

# 1. Set Python Path
export PYTHONPATH=$PYTHONPATH:/workspace/TurboDiffusion/turbodiffusion
cd /workspace/TurboDiffusion

# 2. Check checkpoints
echo "Checking checkpoints..."
if [ ! -f "checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth" ]; then
    echo "Downloading model checkpoints..."
    mkdir -p checkpoints
    cd checkpoints
    wget https://huggingface.co/TurboDiffusion/TurboWan2.1-T2V-1.3B-480P/resolve/main/TurboWan2.1-T2V-1.3B-480P-quant.pth
    wget https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth
    wget https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
    cd ..
fi

# 3. Run inference
echo "Running TurboDiffusion inference..."
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth \
    --model Wan2.1-1.3B \
    --attention_type sla \
    --quant_linear \
    --resolution 480p \
    --num_frames 81 \
    --num_steps 4 \
    --prompt "${1:-A stylish woman walks down a Tokyo street}" \
    --save_path "${2:-output/generated_video.mp4}"

echo "=== Done! Video saved to ${2:-output/generated_video.mp4} ==="
```

---

### 阶段 6：文档更新

**目标文件：** `README.md`

添加 RunPod 专用章节：

```markdown
## RunPod Deployment

For deployment on RunPod or similar containerized environments with limited disk space:

### Quick Start (No Compilation Required)

\`\`\`bash
# Use SLA mode instead of SageSLA (no SpargeAttn dependency)
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth \
    --attention_type sla \
    --quant_linear \
    --prompt "Your prompt here"
\`\`\`

### Notes
- The code includes a pure PyTorch fallback for RoPE when `flash-attn` is not available
- Use `--attention_type sla` to avoid the SpargeAttn dependency
- SLA mode is slightly slower but produces identical results
```

---

## 四、修改优先级与依赖关系

```
┌─────────────────────────────────────────────────────────────┐
│                    修改执行顺序                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  阶段 1: RoPE Fallback (wan2pt1.py)                         │
│    ↓ [必须首先完成，否则代码无法运行]                          │
│                                                             │
│  阶段 2: SLA Fallback (core.py)                             │
│    ↓ [优化错误信息]                                          │
│                                                             │
│  阶段 3: 默认值调整 (wan2.1_t2v_infer.py)                    │
│    ↓ [改善开箱即用体验]                                       │
│                                                             │
│  阶段 4: Checkpoint 加载 (modify_model.py)                  │
│    ↓ [增强兼容性]                                            │
│                                                             │
│  阶段 5: QuickStart 脚本 (新文件)                            │
│    ↓ [用户体验]                                              │
│                                                             │
│  阶段 6: 文档更新 (README.md)                                │
│    [完成]                                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 五、风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| PyTorch RoPE 维度不匹配 | 中 | 高 | 添加详细的 shape 日志，便于调试 |
| Triton 版本不兼容 | 低 | 高 | SLA kernel 已较成熟，但需测试 |
| 性能下降明显 | 中 | 低 | 文档明确说明这是兼容性优先的方案 |

---

## 六、测试计划

完成修改后，按以下顺序测试：

### 1. 单元测试
验证 `apply_rotary_emb_torch` 输出与预期一致

### 2. 集成测试
使用以下命令生成视频：

```bash
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth \
    --model Wan2.1-1.3B \
    --attention_type sla \
    --quant_linear \
    --prompt "Batman and Spider-Man chase scene in New York City"
```

### 3. 输出验证
确认生成的视频文件可播放

---

## 七、预期成功命令

修改完成后，以下命令应该能在 RunPod 上直接运行：

```bash
# 设置环境
export PYTHONPATH=$PYTHONPATH:/workspace/TurboDiffusion/turbodiffusion
cd /workspace/TurboDiffusion

# 运行推理
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth \
    --model Wan2.1-1.3B \
    --attention_type sla \
    --quant_linear \
    --resolution 480p \
    --prompt "Batman and Spider-Man chase scene in New York City, cinematic lighting"
```

---

## 八、昨日成功经验总结

根据 2025-12-26 的调试日志，成功运行的关键要素：

1. ✅ 添加 `apply_rotary_emb_torch` 函数作为 PyTorch fallback
2. ✅ 将 fallback 函数定义放在 try-except 块之前
3. ✅ 使用 `--attention_type sla`（非 sagesla）
4. ✅ 使用 `--quant_linear` 配合量化 checkpoint
5. ✅ 使用 `strict=False` 加载 state_dict

---

**计划确认后，将依次执行阶段 1-6 的修改。**
