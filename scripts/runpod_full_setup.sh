#!/bin/bash
# ================================================================
# TurboDiffusion - RunPod 一键部署脚本
# ================================================================
# 日期: 2025-12-27
# 用途: 新开 Pod 后，一键安装所有依赖并准备好环境
# 
# 使用方法:
#   1. SSH 连接到 RunPod
#   2. cd /workspace/TurboDiffusion
#   3. chmod +x scripts/runpod_full_setup.sh
#   4. ./scripts/runpod_full_setup.sh
# ================================================================

set -e

echo "================================================================"
echo "  TurboDiffusion RunPod 一键部署"
echo "  日期: $(date)"
echo "================================================================"
echo ""

# 检查是否在 workspace 目录
if [ ! -d "/workspace" ]; then
    echo "错误: 不在 RunPod 环境中"
    exit 1
fi

cd /workspace

# ================================================================
# 步骤 1: 克隆或更新代码
# ================================================================
echo "[1/5] 检查代码..."
if [ -d "TurboDiffusion" ]; then
    echo "  → 代码已存在，拉取最新版本..."
    cd TurboDiffusion
    git pull origin main || echo "  → Git pull 失败，继续使用本地版本"
else
    echo "  → 克隆代码..."
    git clone https://github.com/tylerus2020/TurboDiffusion.git
    cd TurboDiffusion
fi

# ================================================================
# 步骤 2: 安装 Python 依赖
# ================================================================
echo ""
echo "[2/5] 安装 Python 依赖..."

pip install --upgrade pip -q

# 核心依赖
pip install -q \
    einops \
    loguru \
    tqdm \
    pillow \
    transformers \
    triton \
    imageio \
    imageio-ffmpeg \
    av \
    pandas \
    scipy \
    sentencepiece \
    protobuf \
    omegaconf \
    hydra-core \
    webdataset \
    ftfy \
    accelerate \
    safetensors \
    termcolor \
    pynvml \
    nvidia-ml-py

echo "  → 依赖安装完成"

# ================================================================
# 步骤 3: 下载模型 Checkpoints
# ================================================================
echo ""
echo "[3/5] 检查模型 Checkpoints..."

mkdir -p checkpoints

# T2V 模型 (非量化版，约 2.9GB)
if [ ! -f "checkpoints/TurboWan2.1-T2V-1.3B-480P.pth" ]; then
    echo "  → 下载 TurboWan2.1-T2V-1.3B-480P.pth (2.9GB)..."
    wget -q --show-progress -P checkpoints \
        https://huggingface.co/TurboDiffusion/TurboWan2.1-T2V-1.3B-480P/resolve/main/TurboWan2.1-T2V-1.3B-480P.pth
else
    echo "  → TurboWan2.1-T2V-1.3B-480P.pth 已存在"
fi

# VAE (约 485MB)
if [ ! -f "checkpoints/Wan2.1_VAE.pth" ]; then
    echo "  → 下载 Wan2.1_VAE.pth (485MB)..."
    wget -q --show-progress -P checkpoints \
        https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth
else
    echo "  → Wan2.1_VAE.pth 已存在"
fi

# 文本编码器 (约 11GB)
if [ ! -f "checkpoints/models_t5_umt5-xxl-enc-bf16.pth" ]; then
    echo "  → 下载 models_t5_umt5-xxl-enc-bf16.pth (11GB)..."
    wget -q --show-progress -P checkpoints \
        https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
else
    echo "  → models_t5_umt5-xxl-enc-bf16.pth 已存在"
fi

# ================================================================
# 步骤 4: 设置环境变量
# ================================================================
echo ""
echo "[4/5] 配置环境..."

# 添加到 bashrc
if ! grep -q "TurboDiffusion" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# TurboDiffusion 环境配置" >> ~/.bashrc
    echo "export PYTHONPATH=\$PYTHONPATH:/workspace/TurboDiffusion/turbodiffusion" >> ~/.bashrc
    echo "cd /workspace/TurboDiffusion" >> ~/.bashrc
fi

export PYTHONPATH=$PYTHONPATH:/workspace/TurboDiffusion/turbodiffusion

# ================================================================
# 步骤 5: 创建快捷脚本
# ================================================================
echo ""
echo "[5/5] 创建快捷命令..."

mkdir -p output

# 创建快捷生成脚本
cat > /usr/local/bin/turbogen << 'EOF'
#!/bin/bash
cd /workspace/TurboDiffusion
export PYTHONPATH=$PYTHONPATH:/workspace/TurboDiffusion/turbodiffusion
PROMPT="${1:-A stylish woman walks down a Tokyo street}"
OUTPUT="${2:-output/video_$(date +%Y%m%d_%H%M%S).mp4}"
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P.pth \
    --attention_type original \
    --num_frames 81 \
    --num_steps 4 \
    --prompt "$PROMPT" \
    --save_path "$OUTPUT"
echo "视频已保存到: $OUTPUT"
EOF
chmod +x /usr/local/bin/turbogen

echo ""
echo "================================================================"
echo "  ✅ 部署完成!"
echo "================================================================"
echo ""
echo "📝 使用方法:"
echo ""
echo "  方法1: 使用快捷命令"
echo "    turbogen \"你的提示词\" output/video.mp4"
echo ""
echo "  方法2: 完整命令"
echo "    python turbodiffusion/inference/wan2.1_t2v_infer.py \\"
echo "        --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P.pth \\"
echo "        --attention_type original \\"
echo "        --prompt \"你的提示词\""
echo ""
echo "📁 目录结构:"
echo "    /workspace/TurboDiffusion/"
echo "    ├── checkpoints/          <- 模型权重"
echo "    ├── output/               <- 生成的视频"
echo "    └── turbodiffusion/       <- 代码"
echo ""
echo "================================================================"
