#!/bin/bash
# TurboDiffusion RunPod Quick Start
# Date: 2025-12-27
#
# Usage: ./runpod_quickstart.sh "your prompt" [output.mp4]
#
# Example:
#   ./runpod_quickstart.sh "A stylish woman walks down a Tokyo street"
#   ./runpod_quickstart.sh "Batman and Spider-Man chase scene" output/batman.mp4

set -e

PROMPT="${1:-A stylish woman walks down a Tokyo street filled with neon lights}"
OUTPUT="${2:-output/generated_video.mp4}"

# Set Python path
export PYTHONPATH=$PYTHONPATH:/workspace/TurboDiffusion/turbodiffusion
cd /workspace/TurboDiffusion

echo "=== TurboDiffusion Quick Start ==="
echo "Prompt: $PROMPT"
echo "Output: $OUTPUT"
echo ""

# Check if checkpoints exist
if [ ! -f "checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth" ]; then
    echo "Error: Checkpoint not found. Run ./scripts/runpod_setup.sh first."
    exit 1
fi

# Create output directory
mkdir -p $(dirname "$OUTPUT")

# Run inference
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P-quant.pth \
    --model Wan2.1-1.3B \
    --attention_type sla \
    --quant_linear \
    --resolution 480p \
    --num_frames 81 \
    --num_steps 4 \
    --prompt "$PROMPT" \
    --save_path "$OUTPUT"

echo ""
echo "=== Done! ==="
echo "Video saved to: $OUTPUT"
