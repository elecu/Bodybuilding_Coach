from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .foreground_segmentation import clean_person_mask
from .quality_gates import compute_quality_gates
from ..metrics.harmony import (
    compute_flow_features,
    compute_landmarks_1d,
    compute_lr_symmetry_from_mask,
    compute_profile_widths,
)


@dataclass
class ViewMetrics:
    angle_deg: int
    silhouette_area_m2: float
    width_profile_m: List[float]
    height_bins: List[float]
    key_widths_m: Dict[str, float]
    slab_area_m2: List[float]
    slab_perimeter_m: List[float]


def _width_profile_from_mask(mask: np.ndarray, depth_m: np.ndarray, intrinsics: Dict[str, float]) -> Tuple[List[float], List[float]]:
    h, w = mask.shape[:2]
    fx = float(intrinsics.get("fx", 0.0))
    if fx <= 0:
        fx = 365.0
    rows = np.linspace(0, h - 1, num=60).astype(int)
    widths = []
    heights = []
    for y in rows:
        row = mask[y]
        xs = np.where(row > 0)[0]
        if xs.size < 2:
            widths.append(0.0)
            heights.append(float(y) / max(1, h - 1))
            continue
        x0, x1 = int(xs.min()), int(xs.max())
        z = depth_m[y, xs]
        z = z[np.isfinite(z) & (z > 0)]
        if z.size == 0:
            widths.append(0.0)
            heights.append(float(y) / max(1, h - 1))
            continue
        z_med = float(np.median(z))
        width_m = (x1 - x0) * z_med / fx
        widths.append(width_m)
        heights.append(float(y) / max(1, h - 1))
    return widths, heights


def _key_widths(widths: List[float], heights: List[float]) -> Dict[str, float]:
    def pick(h):
        if not heights:
            return 0.0
        idx = int(np.argmin(np.abs(np.array(heights) - h)))
        return float(widths[idx])

    return {
        "shoulders": pick(0.85),
        "chest": pick(0.78),
        "waist": pick(0.62),
        "hips": pick(0.55),
        "thigh": pick(0.40),
        "calf": pick(0.18),
    }


def _slab_metrics(points: np.ndarray) -> Tuple[List[float], List[float]]:
    if points.size == 0:
        return [], []
    y = points[:, 1]
    y_min, y_max = float(np.min(y)), float(np.max(y))
    bins = np.linspace(y_min, y_max, num=60)
    areas = []
    perims = []
    for i in range(len(bins) - 1):
        slab = points[(y >= bins[i]) & (y < bins[i + 1])]
        if slab.shape[0] < 30:
            areas.append(0.0)
            perims.append(0.0)
            continue
        xz = slab[:, [0, 2]]
        xz_mm = (xz * 1000.0).astype(np.int32)
        hull = cv2.convexHull(xz_mm)
        area = float(cv2.contourArea(hull)) / 1e6
        perim = float(cv2.arcLength(hull, True)) / 1000.0
        areas.append(area)
        perims.append(perim)
    return areas, perims


def compute_view_metrics(
    angle_deg: int,
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: Dict[str, float],
    points_xyz: Optional[np.ndarray] = None,
) -> ViewMetrics:
    silhouette_area = float(np.count_nonzero(mask > 0))
    # Approximate m^2 using median depth.
    z = depth_m[mask > 0]
    z = z[np.isfinite(z) & (z > 0)]
    if z.size:
        z_med = float(np.median(z))
    else:
        z_med = 2.0
    fx = float(intrinsics.get("fx", 365.0))
    fy = float(intrinsics.get("fy", 365.0))
    px_area_m2 = (z_med / fx) * (z_med / fy)
    silhouette_area_m2 = silhouette_area * px_area_m2

    widths, heights = _width_profile_from_mask(mask, depth_m, intrinsics)
    key_widths = _key_widths(widths, heights)

    slab_area, slab_perim = ([], [])
    if points_xyz is not None and points_xyz.size:
        slab_area, slab_perim = _slab_metrics(points_xyz)

    return ViewMetrics(
        angle_deg=angle_deg,
        silhouette_area_m2=silhouette_area_m2,
        width_profile_m=widths,
        height_bins=heights,
        key_widths_m=key_widths,
        slab_area_m2=slab_area,
        slab_perimeter_m=slab_perim,
    )


def fuse_circumference(front: List[float], side: List[float]) -> List[float]:
    out = []
    for wf, ws in zip(front, side):
        if wf <= 0 or ws <= 0:
            out.append(0.0)
            continue
        a = wf / 2.0
        b = ws / 2.0
        # Ramanujan approximation for ellipse circumference.
        c = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
        out.append(float(c))
    return out


def compute_metrics_for_scan(
    scan_dir: Path,
    view_masks: List[np.ndarray],
    view_depths: List[np.ndarray],
    view_points: List[Optional[np.ndarray]],
    intrinsics: Dict[str, float],
    meta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    angles = [0, 90, 180, 270]
    cleaned_masks: List[np.ndarray] = []
    for i in range(4):
        depth = view_depths[i]
        raw_mask = view_masks[i]
        cleaned = clean_person_mask(raw_mask)
        if cleaned is None or cleaned.shape[:2] != depth.shape[:2] or not np.any(cleaned):
            if raw_mask is not None and raw_mask.shape[:2] == depth.shape[:2]:
                cleaned = (raw_mask > 0).astype(np.uint8) * 255
            else:
                cleaned = np.zeros(depth.shape[:2], dtype=np.uint8)
        cleaned_masks.append(cleaned)

    views = []
    for i in range(4):
        vm = compute_view_metrics(
            angles[i],
            cleaned_masks[i],
            view_depths[i],
            intrinsics,
            view_points[i],
        )
        views.append(vm)

    # Combine front/back and left/right.
    front = views[0].width_profile_m
    back = views[2].width_profile_m
    side1 = views[1].width_profile_m
    side2 = views[3].width_profile_m

    front_back = [float(np.mean([f, b])) for f, b in zip(front, back)]
    side_avg = [float(np.mean([s1, s2])) for s1, s2 in zip(side1, side2)]
    circum = fuse_circumference(front_back, side_avg)

    out = {
        "views": [
            {
                "angle_deg": v.angle_deg,
                "silhouette_area_m2": v.silhouette_area_m2,
                "width_profile_m": v.width_profile_m,
                "height_bins": v.height_bins,
                "key_widths_m": v.key_widths_m,
                "slab_area_m2": v.slab_area_m2,
                "slab_perimeter_m": v.slab_perimeter_m,
            }
            for v in views
        ],
        "circumference_profile_m": circum,
        "height_bins": views[0].height_bins,
    }

    front_widths = views[0].width_profile_m if views else []
    front_heights = views[0].height_bins if views else []
    landmarks_1d = compute_landmarks_1d(front_widths, front_heights)
    flow = compute_flow_features(front_widths, front_heights)
    sym = compute_lr_symmetry_from_mask(cleaned_masks[0] if cleaned_masks else None)
    harmony_profile_widths = compute_profile_widths(front_widths, front_heights)
    out["landmarks_1d"] = landmarks_1d
    out["harmony_features"] = {
        "profile_widths_m": harmony_profile_widths,
        "flow_score": flow.get("flow_score"),
        "flow_stats": flow.get("stats") or {},
        "symmetry_proxy_score": sym.get("symmetry_score"),
        "symmetry_proxy_stats": sym.get("stats") or {},
    }

    derived_dir = scan_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)

    meta_obj: Dict[str, object] = {}
    if isinstance(meta, dict):
        meta_obj = meta
    else:
        try:
            from .pcd_io import load_meta

            loaded_meta = load_meta(derived_dir)
            if isinstance(loaded_meta, dict):
                meta_obj = loaded_meta
        except Exception:
            meta_obj = {}

    gates = compute_quality_gates(
        cleaned_masks,
        view_depths,
        view_points,
        intrinsics,
        out,
        meta=meta_obj or None,
    )
    gates_path = derived_dir / "quality_gates.json"
    gates_path.write_text(json.dumps(gates, indent=2), encoding="utf-8")

    metrics_path = derived_dir / "metrics.json"
    metrics_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    # Optional condition computation.
    try:
        from ..metrics.condition import compute_condition_for_session
        if meta_obj:
            condition = compute_condition_for_session(scan_dir, meta_obj, out)
            condition_path = derived_dir / "condition.json"
            condition_path.write_text(json.dumps(condition, indent=2), encoding="utf-8")
    except Exception:
        pass
    return out
