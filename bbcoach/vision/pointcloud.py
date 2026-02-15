from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def _intrinsics_from_meta(intrinsics: Dict[str, float]) -> Tuple[float, float, float, float]:
    fx = float(intrinsics.get("fx", 0.0))
    fy = float(intrinsics.get("fy", 0.0))
    cx = float(intrinsics.get("cx", 0.0))
    cy = float(intrinsics.get("cy", 0.0))
    if fx <= 0 or fy <= 0:
        raise ValueError("Invalid intrinsics for point cloud.")
    return fx, fy, cx, cy


def depth_to_pointcloud(
    depth_m: np.ndarray,
    intrinsics: Dict[str, float],
    mask: Optional[np.ndarray] = None,
    rgb_bgr: Optional[np.ndarray] = None,
    stride: int = 2,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if depth_m is None or depth_m.size == 0:
        return np.zeros((0, 3), dtype=np.float32), None
    fx, fy, cx, cy = _intrinsics_from_meta(intrinsics)
    h, w = depth_m.shape[:2]
    ys = np.arange(0, h, stride, dtype=np.int32)
    xs = np.arange(0, w, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    z = depth_m[grid_y, grid_x]
    valid = np.isfinite(z) & (z > 0)
    if mask is not None and mask.shape[:2] == depth_m.shape[:2]:
        valid &= (mask[grid_y, grid_x] > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), None
    u = grid_x[valid].astype(np.float32)
    v = grid_y[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    colors = None
    if rgb_bgr is not None and rgb_bgr.shape[:2] == depth_m.shape[:2]:
        bgr = rgb_bgr[grid_y[valid], grid_x[valid]]
        if bgr.ndim == 2:
            bgr = np.repeat(bgr[:, None], 3, axis=1)
        if bgr.ndim == 3 and bgr.shape[1] == 1:
            bgr = np.repeat(bgr, 3, axis=1)
        if bgr.ndim == 3 and bgr.shape[1] >= 3:
            bgr = bgr[:, :3]
        colors = bgr[:, ::-1].astype(np.uint8, copy=False)
    return points, colors


def write_ply(path: str | Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    path = Path(path)
    n = int(points.shape[0]) if points is not None else 0
    if colors is not None:
        colors = np.asarray(colors)
        if colors.ndim > 2:
            colors = colors.reshape(colors.shape[0], -1)
        if colors.shape[0] != n or colors.shape[1] < 3:
            colors = None
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if n == 0:
            return
        if colors is None:
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            for p, c in zip(points, colors):
                c0, c1, c2 = int(c[0]), int(c[1]), int(c[2])
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c0} {c1} {c2}\n")


def write_pcd(path: str | Path, points: np.ndarray) -> None:
    path = Path(path)
    n = int(points.shape[0]) if points is not None else 0
    with path.open("w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {n}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\n")
        f.write("DATA ascii\n")
        if n == 0:
            return
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
