"""JSON atlas metadata emission."""

from __future__ import annotations

import json
from pathlib import Path

from sprite_forge.slicing import Tile

DEFAULT_ROW_NAMES_4 = ["down", "right", "up", "left"]
DEFAULT_ROW_NAMES_8 = [
    "down", "down_right", "right", "up_right",
    "up", "up_left", "left", "down_left",
]


def _row_names(rows: int) -> list[str]:
    if rows == 4:
        return DEFAULT_ROW_NAMES_4
    if rows == 8:
        return DEFAULT_ROW_NAMES_8
    return [f"row_{i}" for i in range(rows)]


def build_atlas(
    tiles: list[Tile],
    rows: int,
    cols: int,
    image_filename: str,
    row_names: list[str] | None = None,
    animation_prefix: str = "walk",
    cell_w: int | None = None,
    cell_h: int | None = None,
    padding: int = 0,
) -> dict:
    """Build a JSON-serializable atlas dict describing the packed sheet."""
    if len(tiles) != rows * cols:
        raise ValueError(f"expected {rows * cols} tiles, got {len(tiles)}")

    names = row_names if row_names is not None else _row_names(rows)
    if len(names) != rows:
        raise ValueError(f"row_names length {len(names)} != rows {rows}")

    cw = cell_w if cell_w is not None else max(t.w + t.trim_offset[0] for t in tiles)
    ch = cell_h if cell_h is not None else max(t.h + t.trim_offset[1] for t in tiles)

    frames: dict[str, dict] = {}
    animations: dict[str, list[str]] = {}

    for t in tiles:
        frame_name = f"{names[t.row]}_{t.col}"
        ox, oy = t.trim_offset
        frames[frame_name] = {
            "x": padding + t.col * (cw + padding) + ox,
            "y": padding + t.row * (ch + padding) + oy,
            "w": t.w,
            "h": t.h,
        }
        animations.setdefault(f"{animation_prefix}_{names[t.row]}", []).append(frame_name)

    for anim in animations.values():
        anim.sort(key=lambda n: int(n.rsplit("_", 1)[1]))

    return {
        "image": image_filename,
        "cell_size": [cw, ch],
        "grid": {"rows": rows, "cols": cols, "padding": padding},
        "frames": frames,
        "animations": animations,
    }


def write_atlas(atlas: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(atlas, indent=2))
    return path
