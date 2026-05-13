"""Pack a list of tiles back into a uniform-grid sprite sheet."""

from __future__ import annotations

from PIL import Image

from sprite_forge.slicing import Tile


def pack_grid(
    tiles: list[Tile],
    rows: int,
    cols: int,
    tile_w: int | None = None,
    tile_h: int | None = None,
    padding: int = 0,
    background: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Image.Image:
    """Pack tiles into a rows*cols grid sheet.

    Tiles are placed by their row/col attributes. If tile_w/tile_h are omitted,
    the max tile dimensions are used. Trimmed tiles (see slice_grid) are re-pasted
    at their original trim_offset within the cell.
    """
    if len(tiles) != rows * cols:
        raise ValueError(f"expected {rows * cols} tiles, got {len(tiles)}")

    cell_w = tile_w if tile_w is not None else max(t.w + t.trim_offset[0] for t in tiles)
    cell_h = tile_h if tile_h is not None else max(t.h + t.trim_offset[1] for t in tiles)

    sheet_w = cols * cell_w + (cols + 1) * padding
    sheet_h = rows * cell_h + (rows + 1) * padding
    sheet = Image.new("RGBA", (sheet_w, sheet_h), background)

    for t in tiles:
        ox, oy = t.trim_offset
        x = padding + t.col * (cell_w + padding) + ox
        y = padding + t.row * (cell_h + padding) + oy
        sheet.paste(t.image, (x, y), t.image if t.image.mode == "RGBA" else None)

    return sheet
