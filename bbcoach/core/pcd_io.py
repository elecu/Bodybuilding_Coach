from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from ..vision.pointcloud import write_pcd, write_ply


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_pointcloud(dir_path: Path, name: str, points: np.ndarray, colors: Optional[np.ndarray] = None) -> Dict[str, str]:
    ensure_dir(dir_path)
    pcd_path = dir_path / f"{name}.pcd"
    write_pcd(pcd_path, points)
    out = {"pcd": str(pcd_path)}
    if colors is not None:
        ply_path = dir_path / f"{name}.ply"
        write_ply(ply_path, points, colors)
        out["ply"] = str(ply_path)
    return out


def save_debug_images(dir_path: Path, prefix: str, images: Dict[str, np.ndarray]) -> Dict[str, str]:
    ensure_dir(dir_path)
    saved = {}
    for key, img in images.items():
        if img is None:
            continue
        out_path = dir_path / f"{prefix}_{key}.png"
        cv2.imwrite(str(out_path), img)
        saved[key] = str(out_path)
    return saved


def save_meta(dir_path: Path, meta: Dict[str, Any]) -> Path:
    ensure_dir(dir_path)
    meta_path = dir_path / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta_path


def load_meta(dir_path: Path) -> Dict[str, Any]:
    meta_path = dir_path / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))
