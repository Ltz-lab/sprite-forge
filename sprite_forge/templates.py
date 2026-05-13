"""Registry of pre-baked OpenPose skeleton templates shipped with the package."""

from __future__ import annotations

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent
TEMPLATES_DIR = _PACKAGE_ROOT / "templates"


def template_path(rows: int, cols: int, motion: str = "walk") -> Path:
    """Resolve the on-disk path for a templated pose grid."""
    return TEMPLATES_DIR / f"{rows}x{cols}_{motion}.png"


def list_templates() -> list[Path]:
    if not TEMPLATES_DIR.exists():
        return []
    return sorted(p for p in TEMPLATES_DIR.iterdir() if p.suffix.lower() == ".png")


def list_template_keys() -> list[str]:
    """Return sortable stem keys like '4x4_walk' for UI dropdowns."""
    return [p.stem for p in list_templates()]
