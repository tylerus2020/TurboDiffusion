#!/bin/bash
# ================================================================
# TurboDiffusion - 批量视频生成脚本
# 13 个 Batman vs Spider-Man 场景
# ================================================================

set -e

cd /workspace/TurboDiffusion
export PYTHONPATH=/workspace/TurboDiffusion/turbodiffusion:$PYTHONPATH

mkdir -p output/batch_$(date +%Y%m%d)
OUTPUT_DIR="output/batch_$(date +%Y%m%d)"

echo "================================================================"
echo "  批量生成 13 个视频"
echo "  开始时间: $(date)"
echo "================================================================"

# Prompt 1
echo ""
echo "[1/13] 生成中..."
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P.pth \
    --attention_type original \
    --num_frames 81 --num_steps 4 \
    --save_path "$OUTPUT_DIR/01_alleyway_chase.mp4" \
    --prompt "Continuous single take shot. The camera flies rapidly through a narrow New York alleyway. We see Spider-Man wall-running on the left wall, dodging a Batarang thrown by Batman who is grappling on the right wall. The camera zooms out as they both burst onto a busy street filled with yellow taxis. Reflections on the puddles, steam rising from sewer grates. Realistic lighting, 4k, fluid motion, no morphing, high fidelity, blockbuster movie trailer vibe."

# Prompt 2
echo ""
echo "[2/13] 生成中..."
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P.pth \
    --attention_type original \
    --num_frames 81 --num_steps 4 \
    --save_path "$OUTPUT_DIR/02_low_angle_skyscrapers.mp4" \
    --prompt "Low angle dynamic shot looking up from the street level. Towering skyscrapers surround the frame like a canyon. Spider-Man is mid-air upside down, shooting a web towards the camera. Batman is diving vertically from above, dark cape consuming the light. Stark contrast between the dark gritty Gotham-style alley and the bright vibrant Times Square lights in the distance. Ray tracing, HDR, color grading with teal and orange, dramatic shadows, masterpiece, visually stunning action composition."

# Prompt 3
echo ""
echo "[3/13] 生成中..."
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P.pth \
    --attention_type original \
    --num_frames 81 --num_steps 4 \
    --save_path "$OUTPUT_DIR/03_rooftop_parkour.mp4" \
    --prompt "Close-up tracking shot, intense action sequence. Spider-Man performing parkour on a rooftop, camera shaking slightly to mimic handheld realism. Batman lands heavily behind him, tactical armor wet from rain, eyes glowing white. Sparks flying from a nearby broken neon sign. Depth of field focuses on Spider-Man fabric texture while Batman is slightly blurred in the background. Slow motion transition to fast speed, rain droplets suspended in air, lens flare, IMAX cinematic quality, unreal engine 5 render style, dramatic lighting."

# Prompt 4 (numbered as 11)
echo ""
echo "[4/13] 生成中..."
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P.pth \
    --attention_type original \
    --num_frames 81 --num_steps 4 \
    --save_path "$OUTPUT_DIR/04_peter_parker_portrait.mp4" \
    --prompt "Close-up cinematic portrait of Peter Parker unmasked on a rainy New York rooftop at night. He is exhausted, leaning against a brick chimney. Individual strands of wet hair are gently blowing in the soft breeze, sticking to his forehead. Raindrops run down his face with realistic skin texture and pores visible. Shallow depth of field with blurred city bokeh in the background."

# Prompt 5 (numbered as 12)
echo ""
echo "[5/13] 生成中..."
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P.pth \
    --attention_type original \
    --num_frames 81 --num_steps 4 \
    --save_path "$OUTPUT_DIR/05_batman_armor_rain.mp4" \
    --prompt "Extreme macro shot of Batman tactical armor chest piece during a heavy downpour. Water droplets hit the metal surface and shatter into micro-mist. Rivulets of water stream down the carbon fiber texture realistically. The camera pans slowly across the wet surface, capturing the reflection of red and blue police lights flashing rhythmically on the wet armor."

# Prompt 6 (numbered as 13)
echo ""
echo "[6/13] 生成中..."
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P.pth \
    --attention_type original \
    --num_frames 81 --num_steps 4 \
    --save_path "$OUTPUT_DIR/06_batman_gargoyle.mp4" \
    --prompt "Wide shot, back view of Batman standing on the edge of a Chrysler Building gargoyle. His long, heavy cape is billowing violently in strong turbulent winds, creating complex, non-repetitive fabric folds. The physics of the heavy cloth struggling against the wind is palpable. Lightning flashes in the distance, illuminating the silhouette against a dark, stormy sky."

# Prompt 7 (numbered as 14)
echo ""
echo "[7/13] 生成中..."
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P.pth \
    --attention_type original \
    --num_frames 81 --num_steps 4 \
    --save_path "$OUTPUT_DIR/07_spiderman_pov_swing.mp4" \
    --prompt "First-person POV Point of View of Spider-Man swinging at high velocity through a narrow alleyway. The brick walls on the side rush past with directional motion blur. Wind effects are visible as trash and newspapers on the ground swirl up in his wake. The camera shakes slightly to simulate the G-force and physical impact of the webline snapping tight."

# Prompt 8 (numbered as 15)
echo ""
echo "[8/13] 生成中..."
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P.pth \
    --attention_type original \
    --num_frames 81 --num_steps 4 \
    --save_path "$OUTPUT_DIR/08_batarang_slowmo.mp4" \
    --prompt "Slow-motion sequence of Spider-Man dodging a Batarang. The camera focuses on the metal weapon spinning in the air, cutting through raindrops. As the Batarang passes, it creates a slipstream in the rain and mist. In the background, Spider-Man body contorts acrobatically, his spandex suit stretching and wrinkling realistically around his muscles."

# Prompt 9 (numbered as 16)
echo ""
echo "[9/13] 生成中..."
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P.pth \
    --attention_type original \
    --num_frames 81 --num_steps 4 \
    --save_path "$OUTPUT_DIR/09_gotham_alley_steam.mp4" \
    --prompt "Atmospheric scene in a grimy Gotham alley. Thick volumetric steam hisses out of a sewer grate, filling the frame. Batman walks slowly through the white steam, his cape swirling and parting the mist physically. God rays from a streetlamp above cut through the fog, creating dynamic light shafts that shift as he moves."

# Prompt 10 (numbered as 17)
echo ""
echo "[10/13] 生成中..."
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P.pth \
    --attention_type original \
    --num_frames 81 --num_steps 4 \
    --save_path "$OUTPUT_DIR/10_times_square_landing.mp4" \
    --prompt "Low-angle shot looking up from a wet asphalt street in Times Square. The ground is covered in puddles reflecting the vibrant, changing neon advertisements. Spider-Man lands heavily in a puddle, creating a complex water crown splash. The ripples distort the neon reflections in the water with high-fidelity fluid simulation."

# Prompt 11 (numbered as 18)
echo ""
echo "[11/13] 生成中..."
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P.pth \
    --attention_type original \
    --num_frames 81 --num_steps 4 \
    --save_path "$OUTPUT_DIR/11_glass_shatter.mp4" \
    --prompt "Dynamic destruction test. Batman crashes through a large plate-glass window in slow motion. Thousands of individual glass shards shatter, catching the light and rotating in the air. Dust particles and debris float in the shockwave. The interior lighting is warm, contrasting with the cool blue moonlight outside."

# Prompt 12 (numbered as 19)
echo ""
echo "[12/13] 生成中..."
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P.pth \
    --attention_type original \
    --num_frames 81 --num_steps 4 \
    --save_path "$OUTPUT_DIR/12_skyscraper_climb.mp4" \
    --prompt "A continuous tracking drone shot circling a skyscraper spire. Spider-Man is wall-crawling up the side while Batman grapples up behind him. The camera rotates 360 degrees around the building, maintaining consistent geometry of the architecture and the characters costumes without morphing or flickering, showcasing temporal consistency."

# Prompt 13 (numbered as 20)
echo ""
echo "[13/13] 生成中..."
python turbodiffusion/inference/wan2.1_t2v_infer.py \
    --dit_path checkpoints/TurboWan2.1-T2V-1.3B-480P.pth \
    --attention_type original \
    --num_frames 81 --num_steps 4 \
    --save_path "$OUTPUT_DIR/13_spiderman_eyes.mp4" \
    --prompt "Hyper-realistic close-up of Spider-Man mask eyes lenses adjusting. The mechanical shutter of the lens narrows and widens. The fabric texture of the mask shows microscopic threading. A reflection of Batman gliding towards him is visible in the glossy surface of the eye lenses. Soft ambient wind noise suggested by the subtle movement of the mask fabric."

echo ""
echo "================================================================"
echo "  ✅ 全部完成!"
echo "  结束时间: $(date)"
echo "  输出目录: $OUTPUT_DIR"
echo "================================================================"
ls -la "$OUTPUT_DIR"
