"""sprite-forge — Gradio Space for prompt-driven sprite sheets.

Runs on HuggingFace ZeroGPU (free tier). Uses Flux.1-schnell +
ControlNet Union for pose-conditioned generation.
"""

from __future__ import annotations

import json
import os
import tempfile
from io import BytesIO
from pathlib import Path

import gradio as gr
import torch
from diffusers import (
    FluxControlPipeline,
    FluxTransformer2DModel,
    ControlNetModel,
)
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from sprite_forge import build_positive_prompt, build_negative_prompt
from sprite_forge import slice_grid, pack_grid, build_atlas
from sprite_forge import template_path

# ── Constants ─────────────────────────────────────────────────────────

MODEL_ID = "black-forest-labs/FLUX.1-schnell"
CONTROLNET_ID = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro"
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EXAMPLES = [
    "armored knight with red sash",
    "wizard in purple robes with crystal staff",
    "cyberpunk ninja with neon visor",
    "cute fox girl with twin tails",
    "undead skeleton warrior with chipped sword",
    "slime monster with big eyes",
    "forest elf archer with longbow",
    "robot with antennas and glowing blue eyes",
    "pirate captain with eyepatch and parrot",
    "cat girl in maid outfit",
]

# ── Model loading (cached) ────────────────────────────────────────────

_PIPE: FluxControlPipeline | None = None


def load_pipeline() -> FluxControlPipeline:
    global _PIPE
    if _PIPE is not None:
        return _PIPE

    print("Loading Flux.1-schnell + ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        CONTROLNET_ID,
        torch_dtype=DTYPE,
    )

    _PIPE = FluxControlPipeline.from_pretrained(
        MODEL_ID,
        controlnet=controlnet,
        torch_dtype=DTYPE,
    ).to(DEVICE)

    # Enable memory optimizations for ZeroGPU / 16GB VRAM
    _PIPE.enable_model_cpu_offload()
    _PIPE.vae.enable_slicing()
    _PIPE.vae.enable_tiling()

    print("Model loaded.")
    return _PIPE


# ── Generation logic ──────────────────────────────────────────────────

DIRS_MAP = {"4 dirs": 4, "8 dirs": 8}
FRAMES_MAP = {"4 frames": 4, "6 frames": 6, "8 frames": 8}
STYLES = ["pixel-art", "painted", "chibi", "realistic", "cel-shaded"]
TEMPLATE_KEYS = ["4x4_walk", "4x8_walk", "8x4_walk", "4x4_idle"]


def generate(
    prompt: str,
    style: str,
    directions: str,
    frames: str,
    template: str,
    seed: int,
    progress=gr.Progress(),
) -> tuple[Image.Image, str, str]:
    """Run the full pipeline: prompt → generation → slice → pack → atlas."""
    rows = DIRS_MAP[directions]
    cols = FRAMES_MAP[frames]

    # 1. Build prompts
    progress(0.1, desc="Building prompt...")
    positive = build_positive_prompt(prompt, rows=rows, cols=cols, style=style)
    negative = build_negative_prompt()

    # 2. Load pose template
    progress(0.2, desc="Loading pose template...")
    pose_path = template_path(rows, cols)
    if not pose_path.exists():
        # Fall back to 4x4 if specific template missing
        pose_path = template_path(4, 4)
        rows, cols = 4, 4
    control_image = Image.open(pose_path).convert("RGB")

    # 3. Generate
    progress(0.3, desc="Generating sprite sheet with Flux...")
    pipe = load_pipeline()

    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    output = pipe(
        prompt=positive,
        negative_prompt=negative,
        control_image=control_image,
        height=control_image.height,
        width=control_image.width,
        num_inference_steps=4,  # Schnell is trained for 4 steps
        guidance_scale=0.0,     # Schnell uses guidance_scale=0
        generator=generator,
        controlnet_conditioning_scale=0.6,
    ).images[0]

    # 4. Slice
    progress(0.6, desc="Slicing frames...")
    tiles = slice_grid(output, rows=rows, cols=cols, trim_alpha=False)

    # 5. Pack into clean sheet
    progress(0.8, desc="Packing sprite sheet...")
    packed = pack_grid(tiles, rows=rows, cols=cols, padding=2)

    # 6. Build JSON atlas
    atlas = build_atlas(
        tiles, rows=rows, cols=cols,
        image_filename="sprite_sheet.png",
        cell_w=packed.width // cols,
        cell_h=packed.height // rows,
        padding=2,
    )

    # 7. Save outputs for download
    progress(0.9, desc="Preparing downloads...")
    sheet_path = tempfile.mktemp(suffix=".png")
    atlas_path = tempfile.mktemp(suffix=".json")
    packed.save(sheet_path)
    Path(atlas_path).write_text(json.dumps(atlas, indent=2))

    info = (
        f"**Prompt:** {prompt}\n"
        f"**Style:** {style} | **Grid:** {directions} × {frames}\n"
        f"**Seed:** {seed}\n"
        f"**Tile count:** {rows * cols} frames\n"
        f"**Templates:** {TEMPLATE_KEYS[0]} (default)"
    )

    return packed, sheet_path, atlas_path, info


# ── Gradio UI ─────────────────────────────────────────────────────────

CSS = """
#gallery img { image-rendering: pixelated; }
footer { display: none !important; }
h1 { text-align: center; }
"""


def build_ui():
    with gr.Blocks(
        css=CSS,
        title="sprite-forge",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            "# 🎮 sprite-forge\n"
            "Type a character description → get a 4-direction walk-cycle sprite sheet. "
            "Powered by Flux.1-schnell + ControlNet."
        )

        with gr.Row():
            with gr.Column(scale=1):
                prompt = gr.Textbox(
                    label="Character description",
                    placeholder="e.g. armored knight with red sash",
                    lines=2,
                )
                with gr.Row():
                    style = gr.Dropdown(
                        choices=STYLES,
                        value="pixel-art",
                        label="Style",
                    )
                    directions = gr.Dropdown(
                        choices=list(DIRS_MAP.keys()),
                        value="4 dirs",
                        label="Directions",
                    )
                    frames = gr.Dropdown(
                        choices=list(FRAMES_MAP.keys()),
                        value="4 frames",
                        label="Frames per row",
                    )
                seed = gr.Number(
                    value=42,
                    label="Seed",
                    precision=0,
                )
                btn = gr.Button("Generate", variant="primary", size="lg")

            with gr.Column(scale=2):
                output = gr.Image(
                    label="Sprite Sheet",
                    type="pil",
                    height=512,
                )
                info = gr.Markdown("")

        with gr.Row(visible=False) as download_row:
            sheet_dl = gr.File(label="Download sprite sheet", visible=True)
            atlas_dl = gr.File(label="Download JSON atlas", visible=True)

        gr.Examples(
            examples=[[e] for e in EXAMPLES],
            inputs=prompt,
            label="Try these",
        )

        btn.click(
            fn=generate,
            inputs=[prompt, style, directions, frames, seed],
            outputs=[output, sheet_dl, atlas_dl, info],
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=download_row,
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
