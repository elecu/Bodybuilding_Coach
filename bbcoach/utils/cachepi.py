from __future__ import annotations

from pathlib import Path
import shutil


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _cache_root() -> Path:
    root = _project_root() / ".cachepi"
    root.mkdir(parents=True, exist_ok=True)
    return root


def mirror_file(path: Path) -> None:
    try:
        src = Path(path)
        if not src.exists() or not src.is_file():
            return
        root = _project_root()
        try:
            rel = src.resolve().relative_to(root)
        except Exception:
            rel = Path("_external") / src.name
        dest = _cache_root() / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    except Exception:
        return
