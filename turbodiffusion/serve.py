# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
TurboDiffusion TUI Server Mode

A persistent GPU server that loads models once and provides an interactive
text-based interface for video generation.

Usage:
    python -m turbodiffusion.serve --dit_path checkpoints/model.pth
    turbodiffusion-serve --dit_path checkpoints/model.pth
"""

import argparse
import math
import os
import sys

# Add inference directory to path for modify_model import
_inference_dir = os.path.join(os.path.dirname(__file__), "inference")
if _inference_dir not in sys.path:
    sys.path.insert(0, _inference_dir)

import torch
from einops import rearrange, repeat
from tqdm import tqdm

from imaginaire.utils.io import save_image_or_video
from imaginaire.utils import log

from rcm.datasets.utils import VIDEO_RES_SIZE_INFO
from rcm.utils.umt5 import get_umt5_embedding
from rcm.tokenizers.wan2pt1 import Wan2pt1VAEInterface

from modify_model import tensor_kwargs, create_model

torch._dynamo.config.suppress_errors = True

# Runtime-adjustable parameters and their types/validators
RUNTIME_PARAMS = {
    "num_steps": {"type": int, "choices": [1, 2, 3, 4]},
    "num_samples": {"type": int, "min": 1},
    "seed": {"type": int, "min": 0},
    "num_frames": {"type": int, "min": 1},
    "sigma_max": {"type": float, "min": 0.1},
}

# Immutable launch-only parameters
LAUNCH_ONLY_PARAMS = [
    "model", "dit_path", "resolution", "aspect_ratio",
    "attention_type", "sla_topk", "quant_linear", "default_norm",
    "vae_path", "text_encoder_path",
]


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for TUI server mode."""
    parser = argparse.ArgumentParser(
        description="TurboDiffusion TUI Server - Interactive video generation"
    )
    # Model configuration (launch-only)
    parser.add_argument("--dit_path", type=str, required=True,
                        help="Path to the DiT model checkpoint")
    parser.add_argument("--model", choices=["Wan2.1-1.3B", "Wan2.1-14B"],
                        default="Wan2.1-1.3B", help="Model to use")
    parser.add_argument("--vae_path", type=str, default="checkpoints/Wan2.1_VAE.pth",
                        help="Path to the Wan2.1 VAE")
    parser.add_argument("--text_encoder_path", type=str,
                        default="checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                        help="Path to the umT5 text encoder")

    # Resolution (launch-only)
    parser.add_argument("--resolution", default="480p", type=str,
                        help="Resolution of the generated output")
    parser.add_argument("--aspect_ratio", default="16:9", type=str,
                        help="Aspect ratio (width:height)")

    # Attention/quantization (launch-only)
    parser.add_argument("--attention_type", choices=["sla", "sagesla", "original"],
                        default="sagesla", help="Attention mechanism type")
    parser.add_argument("--sla_topk", type=float, default=0.1,
                        help="Top-k ratio for SLA/SageSLA attention")
    parser.add_argument("--quant_linear", action="store_true",
                        help="Use quantized linear layers")
    parser.add_argument("--default_norm", action="store_true",
                        help="Use default LayerNorm/RMSNorm (not optimized)")

    # Runtime-adjustable parameters (defaults)
    parser.add_argument("--num_steps", type=int, choices=[1, 2, 3, 4], default=4,
                        help="Number of inference steps (1-4)")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples to generate")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Number of frames to generate")
    parser.add_argument("--sigma_max", type=float, default=80,
                        help="Initial sigma for rCM")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")

    return parser.parse_args()


def load_models(args: argparse.Namespace):
    """Load models once at startup and keep on GPU."""
    log.info(f"Loading DiT model from {args.dit_path}")
    net = create_model(dit_path=args.dit_path, args=args)
    # Keep on GPU (don't move to CPU)
    net.cuda().eval()
    torch.cuda.empty_cache()
    log.success("Successfully loaded DiT model.")

    log.info(f"Loading VAE from {args.vae_path}")
    tokenizer = Wan2pt1VAEInterface(vae_pth=args.vae_path)
    log.success("Successfully loaded VAE.")

    return net, tokenizer


def generate(net, tokenizer, args: argparse.Namespace, prompt: str, output_path: str) -> str:
    """Generate video from prompt. Returns the output path."""
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]

    # Get text embedding
    log.info(f"Computing text embedding...")
    with torch.no_grad():
        text_emb = get_umt5_embedding(
            checkpoint_path=args.text_encoder_path,
            prompts=prompt
        ).to(**tensor_kwargs)

    condition = {
        "crossattn_emb": repeat(
            text_emb.to(**tensor_kwargs),
            "b l d -> (k b) l d",
            k=args.num_samples
        )
    }

    state_shape = [
        tokenizer.latent_ch,
        tokenizer.get_latent_num_frames(args.num_frames),
        h // tokenizer.spatial_compression_factor,
        w // tokenizer.spatial_compression_factor,
    ]

    generator = torch.Generator(device=tensor_kwargs["device"])
    generator.manual_seed(args.seed)

    init_noise = torch.randn(
        args.num_samples,
        *state_shape,
        dtype=torch.float32,
        device=tensor_kwargs["device"],
        generator=generator,
    )

    mid_t = [1.5, 1.4, 1.0][: args.num_steps - 1]

    t_steps = torch.tensor(
        [math.atan(args.sigma_max), *mid_t, 0],
        dtype=torch.float64,
        device=init_noise.device,
    )

    # Convert TrigFlow timesteps to RectifiedFlow
    t_steps = torch.sin(t_steps) / (torch.cos(t_steps) + torch.sin(t_steps))

    # Sampling loop
    x = init_noise.to(torch.float64) * t_steps[0]
    ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
    total_steps = t_steps.shape[0] - 1

    for i, (t_cur, t_next) in enumerate(tqdm(
        list(zip(t_steps[:-1], t_steps[1:])),
        desc="Sampling",
        total=total_steps
    )):
        with torch.no_grad():
            v_pred = net(
                x_B_C_T_H_W=x.to(**tensor_kwargs),
                timesteps_B_T=(t_cur.float() * ones * 1000).to(**tensor_kwargs),
                **condition
            ).to(torch.float64)

            x = (1 - t_next) * (x - t_cur * v_pred) + t_next * torch.randn(
                *x.shape,
                dtype=torch.float32,
                device=tensor_kwargs["device"],
                generator=generator,
            )

    samples = x.float()

    # Decode
    log.info("Decoding video...")
    with torch.no_grad():
        video = tokenizer.decode(samples)

    to_show = [video.float().cpu()]
    to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    save_image_or_video(
        rearrange(to_show, "n b c t h w -> c t (n h) (b w)"),
        output_path,
        fps=16
    )

    return output_path


def get_multiline_prompt() -> str:
    """Read multi-line prompt from user. Empty line finishes input."""
    lines = []
    while True:
        try:
            line = input("> " if lines else "")
            if line == "":
                break
            lines.append(line)
        except EOFError:
            if not lines:
                return None
            break
    return "\n".join(lines)


def print_help():
    """Print help for slash commands."""
    print("""
Commands:
  /help              Show this help message
  /show              Show current configuration
  /set <param> <val> Set a runtime parameter
  /reset             Reset runtime parameters to defaults
  /quit              Exit the server

Runtime parameters (adjustable with /set):
  num_steps   Number of inference steps (1-4)
  num_samples Number of samples per generation
  seed        Random seed
  num_frames  Number of video frames
  sigma_max   Initial sigma for rCM
""")


def print_config(args: argparse.Namespace, defaults: dict):
    """Print current configuration."""
    print("\n=== Launch Configuration (immutable) ===")
    print(f"  model:           {args.model}")
    print(f"  dit_path:        {args.dit_path}")
    print(f"  resolution:      {args.resolution}")
    print(f"  aspect_ratio:    {args.aspect_ratio}")
    print(f"  attention_type:  {args.attention_type}")
    print(f"  sla_topk:        {args.sla_topk}")
    print(f"  quant_linear:    {args.quant_linear}")
    print(f"  default_norm:    {args.default_norm}")

    print("\n=== Runtime Configuration (adjustable) ===")
    for param in RUNTIME_PARAMS:
        val = getattr(args, param)
        default = defaults[param]
        marker = "" if val == default else " *"
        print(f"  {param}: {val}{marker}")
    print()


def handle_set_command(args: argparse.Namespace, param: str, value: str) -> bool:
    """Handle /set command. Returns True if successful."""
    if param not in RUNTIME_PARAMS:
        print(f"Error: '{param}' is not a runtime parameter.")
        print(f"Adjustable parameters: {', '.join(RUNTIME_PARAMS.keys())}")
        return False

    spec = RUNTIME_PARAMS[param]
    try:
        typed_value = spec["type"](value)
    except ValueError:
        print(f"Error: Invalid value '{value}' for {param}")
        return False

    # Validate
    if "choices" in spec and typed_value not in spec["choices"]:
        print(f"Error: {param} must be one of {spec['choices']}")
        return False
    if "min" in spec and typed_value < spec["min"]:
        print(f"Error: {param} must be >= {spec['min']}")
        return False

    setattr(args, param, typed_value)
    print(f"{param} = {typed_value}")
    return True


def handle_command(cmd: str, args: argparse.Namespace, defaults: dict) -> bool:
    """Handle slash command. Returns False if should quit."""
    parts = cmd.strip().split()
    command = parts[0].lower()

    if command == "/quit":
        return False
    elif command == "/help":
        print_help()
    elif command == "/show":
        print_config(args, defaults)
    elif command == "/set":
        if len(parts) != 3:
            print("Usage: /set <param> <value>")
        else:
            handle_set_command(args, parts[1], parts[2])
    elif command == "/reset":
        for param, default in defaults.items():
            setattr(args, param, default)
        print("Runtime parameters reset to defaults.")
    else:
        print(f"Unknown command: {command}")
        print("Type /help for available commands.")

    return True


def print_header(args: argparse.Namespace):
    """Print server header with current config."""
    w, h = VIDEO_RES_SIZE_INFO[args.resolution][args.aspect_ratio]
    print(f"""
TurboDiffusion TUI Server
=========================
Model: {args.model} | Resolution: {args.resolution} ({w}x{h}) | Steps: {args.num_steps}
Type /help for commands, or enter a prompt to generate.
""")


def run_tui(net, tokenizer, args: argparse.Namespace):
    """Main TUI loop."""
    # Store defaults for /reset
    defaults = {param: getattr(args, param) for param in RUNTIME_PARAMS}
    last_output_path = "output/generated_video.mp4"

    print_header(args)

    while True:
        print("Prompt (empty line to generate):")
        prompt = get_multiline_prompt()

        if prompt is None:
            # EOF
            print("\nGoodbye!")
            break

        prompt = prompt.strip()

        if not prompt:
            continue

        if prompt.startswith("/"):
            if not handle_command(prompt, args, defaults):
                print("Goodbye!")
                break
            continue

        # Get output path
        try:
            user_path = input(f"Output path [{last_output_path}]: ").strip()
        except EOFError:
            print("\nGoodbye!")
            break

        output_path = user_path if user_path else last_output_path

        # Ensure .mp4 extension
        if not output_path.endswith(".mp4"):
            output_path += ".mp4"

        # Generate
        try:
            result_path = generate(net, tokenizer, args, prompt, output_path)
            log.success(f"Generated: {result_path}")
            last_output_path = result_path
        except Exception as e:
            log.error(f"Generation failed: {e}")

        print()


def main(passed_args: argparse.Namespace = None):
    """Main entry point for TUI server."""
    args = passed_args if passed_args is not None else parse_arguments()

    # Validate resolution
    if args.resolution not in VIDEO_RES_SIZE_INFO:
        log.error(f"Invalid resolution: {args.resolution}")
        log.info(f"Available: {list(VIDEO_RES_SIZE_INFO.keys())}")
        sys.exit(1)

    if args.aspect_ratio not in VIDEO_RES_SIZE_INFO[args.resolution]:
        log.error(f"Invalid aspect ratio: {args.aspect_ratio}")
        log.info(f"Available: {list(VIDEO_RES_SIZE_INFO[args.resolution].keys())}")
        sys.exit(1)

    # Load models
    net, tokenizer = load_models(args)

    # Run TUI
    try:
        run_tui(net, tokenizer, args)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")


if __name__ == "__main__":
    main()
