from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


def _require_open3d():
    try:
        import open3d as o3d  # type: ignore
        return o3d
    except Exception as exc:
        raise RuntimeError("Open3D not available") from exc


def _canonical_view_paths(scan_dir: Path) -> List[Path]:
    out: List[Path] = []
    for angle in (0, 90, 180, 270):
        p = scan_dir / "raw" / f"view_{angle:03d}" / f"view_{angle:03d}.pcd"
        if p.exists():
            out.append(p)
    if out:
        return out
    return sorted(p for p in (scan_dir / "raw").glob("view_*/*.pcd") if not p.stem.endswith("_depth"))


@dataclass
class ManualAlignResult:
    transforms_path: Path
    message: str


def manual_align(scan_dir: Path) -> ManualAlignResult:
    o3d = _require_open3d()
    view_paths = _canonical_view_paths(scan_dir)
    if not view_paths:
        raise RuntimeError("No view PCDs found.")

    import numpy as np

    pcds = [o3d.io.read_point_cloud(str(p)) for p in view_paths]
    # Use numpy eye for compatibility across Open3D CPU/CUDA builds.
    transforms: List[List[float]] = [list(np.eye(4, dtype=np.float64).reshape(-1)) for _ in pcds]

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Manual Align (WASD=move, Z/X=Yaw, 1-4 select, S=save+exit)")
    idx = 0

    def _clone_pcd(pcd):
        try:
            return pcd.clone()
        except Exception:
            try:
                return pcd.copy()
            except Exception:
                pcd2 = o3d.geometry.PointCloud()
                pcd2.points = pcd.points
                pcd2.colors = pcd.colors
                pcd2.normals = pcd.normals
                return pcd2

    def _apply():
        vis.clear_geometries()
        for i, p in enumerate(pcds):
            mat = np.array(transforms[i], dtype=np.float64).reshape(4, 4)
            tmp = _clone_pcd(p)
            tmp.transform(mat.copy())
            vis.add_geometry(tmp, reset_bounding_box=(i == 0))
        vis.poll_events()
        vis.update_renderer()

    def _select(n: int):
        nonlocal idx
        idx = n
        return False

    def _translate(dx=0.0, dy=0.0, dz=0.0):
        mat = np.eye(4, dtype=np.float64)
        mat[0, 3] = dx
        mat[1, 3] = dy
        mat[2, 3] = dz
        cur = np.array(transforms[idx], dtype=np.float64).reshape(4, 4)
        cur = mat @ cur
        transforms[idx] = list(cur.reshape(-1))
        _apply()
        return False

    def _yaw(deg: float):
        import math
        ang = math.radians(deg)
        mat = np.eye(4, dtype=np.float64)
        mat[0, 0] = math.cos(ang)
        mat[0, 2] = math.sin(ang)
        mat[2, 0] = -math.sin(ang)
        mat[2, 2] = math.cos(ang)
        cur = np.array(transforms[idx], dtype=np.float64).reshape(4, 4)
        cur = mat @ cur
        transforms[idx] = list(cur.reshape(-1))
        _apply()
        return False

    def _save_exit(_vis):
        out_path = scan_dir / "derived" / "transforms.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"views": [str(p) for p in view_paths], "transforms": transforms}
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _vis.destroy_window()
        return True

    # Controls
    vis.register_key_callback(ord("1"), lambda _v: _select(0))
    vis.register_key_callback(ord("2"), lambda _v: _select(1))
    vis.register_key_callback(ord("3"), lambda _v: _select(2))
    vis.register_key_callback(ord("4"), lambda _v: _select(3))
    vis.register_key_callback(ord("W"), lambda _v: _translate(dy=0.01))
    vis.register_key_callback(ord("S"), _save_exit)
    vis.register_key_callback(ord("A"), lambda _v: _translate(dx=-0.01))
    vis.register_key_callback(ord("D"), lambda _v: _translate(dx=0.01))
    vis.register_key_callback(ord("Q"), lambda _v: _translate(dz=0.01))
    vis.register_key_callback(ord("E"), lambda _v: _translate(dz=-0.01))
    vis.register_key_callback(ord("Z"), lambda _v: _yaw(-2.0))
    vis.register_key_callback(ord("X"), lambda _v: _yaw(2.0))

    _apply()
    vis.run()
    out_path = scan_dir / "derived" / "transforms.json"
    if out_path.exists():
        return ManualAlignResult(out_path, "Saved transforms.")
    return ManualAlignResult(out_path, "No transforms saved.")
