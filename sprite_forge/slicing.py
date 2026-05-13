"""Deterministic grid slicing for sprite sheets."""

from __future__ import annotations

from dataclasses import dataclass
from PIL import Image


@dataclass(frozen=True)
class Tile:
    image: Image.Image
    row: int
    col: int
    x: int
    y: int
    w: int
    h: int
    trim_offset: tuple[int, int] = (0, 0)


def slice_grid(
    sheet: Image.Image,
    rows: int,
    cols: int,
    trim_alpha: bool = False,
) -> list[Tile]:
    """Slice a sheet into rows*cols tiles. Reading order: left-to-right, top-to-bottom."""
    if rows <= 0 or cols <= 0:
        raise ValueError(f"rows and cols must be positive, got rows={rows} cols={cols}")

    sheet_w, sheet_h = sheet.size
    if sheet_w % cols != 0 or sheet_h % rows != 0:
        raise ValueError(
            f"sheet size {sheet_w}x{sheet_h} not divisible by grid {cols}x{rows}"
        )

    tile_w = sheet_w // cols
    tile_h = sheet_h // rows

    tiles: list[Tile] = []
    for r in range(rows):
        for c in range(cols):
            x, y = c * tile_w, r * tile_h
            crop = sheet.crop((x, y, x + tile_w, y + tile_h))
            trim_offset = (0, 0)
            if trim_alpha and crop.mode in ("RGBA", "LA"):
                bbox = crop.getbbox()
                if bbox is not None:
                    trim_offset = (bbox[0], bbox[1])
                    crop = crop.crop(bbox)
            tiles.append(
                Tile(
                    image=crop,
                    row=r,
                    col=c,
                    x=x,
                    y=y,
                    w=crop.width,
                    h=crop.height,
                    trim_offset=trim_offset,
                )
            )
    return tiles
