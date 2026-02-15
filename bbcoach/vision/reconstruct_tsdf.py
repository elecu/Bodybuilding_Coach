from __future__ import annotations

from pathlib import Path
from typing import Optional


def is_available() -> bool:
    try:
        import open3d  # noqa: F401
        return True
    except Exception:
        return False


def reconstruct_tsdf(_frames: object, out_path: Path) -> Optional[Path]:
    if not is_available():
        raise RuntimeError("Open3D not available")
    # Placeholder: real TSDF integration will be added when depth source exists.
    raise RuntimeError("TSDF reconstruction not implemented")
