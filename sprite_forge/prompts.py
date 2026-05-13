"""Prompt assembly for sprite-sheet generation.

Pure Python, no ML deps. Separated from the ComfyUI node so it can be tested
and reused from the CLI.
"""

from __future__ import annotations

STYLE_TOKENS: dict[str, str] = {
    "pixel-art": "pixel art, 16-bit game sprite, clean pixel edges, limited palette",
    "painted": "hand-painted game art, soft shading, high detail",
    "chibi": "chibi style, big head, small body, cute proportions",
    "realistic": "realistic rendering, detailed textures, soft shadows",
    "cel-shaded": "cel-shaded, bold outlines, flat color regions",
}

DEFAULT_NEGATIVE = (
    "inconsistent character, different characters across frames, blurry gaps, "
    "extra limbs, merged figures, text, watermark, signature, photograph, "
    "3d render, distorted anatomy, cropped limbs, jpeg artifacts"
)

DIRECTION_LABELS_4 = ["front view (facing camera)", "right side view", "back view (facing away)", "left side view"]
DIRECTION_LABELS_8 = [
    "front view",
    "front-right 3/4 view",
    "right side view",
    "back-right 3/4 view",
    "back view",
    "back-left 3/4 view",
    "left side view",
    "front-left 3/4 view",
]


def _direction_labels(rows: int) -> list[str]:
    if rows == 4:
        return DIRECTION_LABELS_4
    if rows == 8:
        return DIRECTION_LABELS_8
    return [f"view {i + 1}" for i in range(rows)]


def build_positive_prompt(
    user_prompt: str,
    rows: int,
    cols: int,
    style: str = "pixel-art",
    background: str = "solid flat magenta background",
) -> str:
    """Construct a positive prompt that coaxes a coherent character sheet."""
    style_tokens = STYLE_TOKENS.get(style, style)
    directions = _direction_labels(rows)
    row_clauses = ", ".join(
        f"row {i + 1} {directions[i]}" for i in range(rows)
    )
    return (
        f"character sheet of {user_prompt.strip()}, "
        f"{rows}x{cols} grid, "
        f"walk cycle animation frames, "
        f"{row_clauses}, "
        f"each row shows {cols} consecutive walk poses, "
        f"consistent character design across all frames, "
        f"same outfit same proportions same color palette in every frame, "
        f"centered in each cell, even spacing between frames, "
        f"{background}, "
        f"{style_tokens}"
    )


def build_negative_prompt(extra: str = "") -> str:
    return f"{DEFAULT_NEGATIVE}, {extra}" if extra else DEFAULT_NEGATIVE
