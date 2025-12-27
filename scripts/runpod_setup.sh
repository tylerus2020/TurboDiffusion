#!/bin/bash
# TurboDiffusion RunPod Environment Setup
# Date: 2025-12-27
# 
# This script sets up the environment on a fresh RunPod instance.
# Run this once after cloning the repo.

set -e
echo "=== TurboDiffusion RunPod Setup ==="

# Set Python path
export PYTHONPATH=$PYTHONPATH:/workspace/TurboDiffusion/turbodiffusion
cd /workspace/TurboDiffusion

# Create directories
mkdir -p checkpoints output

# Download checkpoints if not exist
echo "Checking model checkpoints..."

if [ ! -f "checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth" ]; then
    echo "Downloading TurboWan2.1-T2V-1.3B-480P-quant.pth (1.4GB)..."
    wget -q --show-progress -P checkpoints https://huggingface.co/TurboDiffusion/TurboWan2.1-T2V-1.3B-480P/resolve/main/TurboWan2.1-T2V-1.3B-480P-quant.pth
fi

if [ ! -f "checkpoints/Wan2.1_VAE.pth" ]; then
    echo "Downloading Wan2.1_VAE.pth (485MB)..."
    wget -q --show-progress -P checkpoints https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/Wan2.1_VAE.pth
fi

if [ ! -f "checkpoints/models_t5_umt5-xxl-enc-bf16.pth" ]; then
    echo "Downloading models_t5_umt5-xxl-enc-bf16.pth (11GB)..."
    wget -q --show-progress -P checkpoints https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
fi

# Install dependencies
echo "Installing Python dependencies..."
pip install einops loguru tqdm pillow transformers triton -q

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To generate a video, run:"
echo "  ./scripts/runpod_quickstart.sh \"Your prompt here\" output/video.mp4"
echo ""
