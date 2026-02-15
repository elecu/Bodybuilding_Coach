from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float32)))


def _safe_std(values: List[float]) -> Optional[float]:
    if len(values) < 2:
        return 0.0 if values else None
    return float(np.std(np.asarray(values, dtype=np.float32)))


def _depth_valid_ratio(mask: np.ndarray, depth_m: np.ndarray) -> Optional[float]:
    if mask is None or depth_m is None or mask.shape[:2] != depth_m.shape[:2]:
        return None
    inside = mask > 0
    total = int(np.count_nonzero(inside))
    if total <= 0:
        return None
    valid = inside & np.isfinite(depth_m) & (depth_m > 0.0)
    valid_count = int(np.count_nonzero(valid))
    return float(valid_count / float(total))


def _mask_band(mask: np.ndarray, low: float, high: float) -> np.ndarray:
    ys, _ = np.where(mask > 0)
    out = np.zeros_like(mask, dtype=bool)
    if ys.size == 0:
        return out
    y_min = int(np.min(ys))
    y_max = int(np.max(ys))
    span = max(1, y_max - y_min)
    y_lo = y_min + int(round(low * span))
    y_hi = y_min + int(round(high * span))
    y_lo = max(0, min(mask.shape[0] - 1, y_lo))
    y_hi = max(y_lo + 1, min(mask.shape[0], y_hi))
    out[y_lo:y_hi, :] = True
    return out


def _torso_depth(mask: np.ndarray, depth_m: np.ndarray, low: float = 0.55, high: float = 0.85) -> Optional[float]:
    if mask is None or depth_m is None or mask.shape[:2] != depth_m.shape[:2]:
        return None
    band = _mask_band(mask, low=low, high=high)
    sel = (mask > 0) & band & np.isfinite(depth_m) & (depth_m > 0.0)
    if not np.any(sel):
        return None
    return float(np.mean(depth_m[sel]))


def _estimate_holes_ratio(points_xyz: np.ndarray, voxel_xy_m: float = 0.02) -> Optional[float]:
    if points_xyz is None or points_xyz.ndim != 2 or points_xyz.shape[1] < 2 or points_xyz.shape[0] < 100:
        return None
    xy = points_xyz[:, :2]
    mins = np.min(xy, axis=0)
    maxs = np.max(xy, axis=0)
    spans = maxs - mins
    if np.any(spans <= 1e-6):
        return 1.0
    idx = np.floor((xy - mins) / max(voxel_xy_m, 1e-4)).astype(np.int32)
    uniq = np.unique(idx, axis=0).shape[0]
    gx = int(np.floor(spans[0] / max(voxel_xy_m, 1e-4))) + 1
    gy = int(np.floor(spans[1] / max(voxel_xy_m, 1e-4))) + 1
    grid = max(1, gx * gy)
    occ = float(uniq / float(grid))
    return float(np.clip(1.0 - occ, 0.0, 1.0))


def _to_float_list(values: List[Optional[float]]) -> List[float]:
    out: List[float] = []
    for v in values:
        if v is None:
            continue
        out.append(float(v))
    return out


def _pointcloud_view_stats(points_xyz: np.ndarray) -> Tuple[int, Optional[float], Optional[float]]:
    if points_xyz is None or points_xyz.ndim != 2 or points_xyz.shape[1] < 3:
        return 0, None, None
    n = int(points_xyz.shape[0])
    if n == 0:
        return 0, None, None
    y = points_xyz[:, 1]
    bbox_height_m = float(np.max(y) - np.min(y)) if y.size else None
    holes_ratio = _estimate_holes_ratio(points_xyz)
    return n, bbox_height_m, holes_ratio


def _flow_spikes(metrics: Dict[str, Any]) -> Tuple[Optional[int], Optional[int], Optional[float]]:
    harmony = metrics.get("harmony_features")
    if not isinstance(harmony, dict):
        return None, None, None
    flow_stats = harmony.get("flow_stats")
    flow_score = harmony.get("flow_score")
    if not isinstance(flow_stats, dict):
        return None, None, None
    spikes = flow_stats.get("spikes")
    curvature_outliers = flow_stats.get("curvature_outliers")
    try:
        spikes_v = int(spikes) if spikes is not None else None
    except Exception:
        spikes_v = None
    try:
        curvature_v = int(curvature_outliers) if curvature_outliers is not None else None
    except Exception:
        curvature_v = None
    try:
        flow_v = float(flow_score) if flow_score is not None else None
    except Exception:
        flow_v = None
    return spikes_v, curvature_v, flow_v


def compute_quality_gates(
    view_masks: List[np.ndarray],
    view_depths_m: List[np.ndarray],
    view_points: List[Optional[np.ndarray]],
    intrinsics,
    metrics: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    del intrinsics  # Reserved for future thresholds tied to camera setup.
    reasons: List[str] = []
    stats: Dict[str, Any] = {}
    meta = meta or {}

    depth_ratios_raw: List[Optional[float]] = []
    torso_depths_raw: List[Optional[float]] = []
    point_counts: List[int] = []
    bbox_heights: List[Optional[float]] = []
    holes_ratios: List[Optional[float]] = []

    n_views = min(len(view_masks), len(view_depths_m), 4)
    for idx in range(n_views):
        mask = view_masks[idx]
        depth_m = view_depths_m[idx]
        depth_ratios_raw.append(_depth_valid_ratio(mask, depth_m))
        torso_depths_raw.append(_torso_depth(mask, depth_m))

        pts = view_points[idx] if idx < len(view_points) else None
        n_pts, bbox_h, holes = _pointcloud_view_stats(pts) if pts is not None else (0, None, None)
        point_counts.append(int(n_pts))
        bbox_heights.append(bbox_h)
        holes_ratios.append(holes)

    depth_ratios = _to_float_list(depth_ratios_raw)
    torso_depths = _to_float_list(torso_depths_raw)

    # Gate 1: lighting_stable (depth completeness proxy).
    depth_ratio_mean = _safe_mean(depth_ratios)
    depth_ratio_std = _safe_std(depth_ratios)
    lighting_ratio_min = float(meta.get("quality_gate_lighting_depth_ratio_min", 0.65))
    lighting_ratio_std_max = float(meta.get("quality_gate_lighting_depth_ratio_std_max", 0.08))
    lighting_stable = bool(
        depth_ratio_mean is not None
        and depth_ratio_std is not None
        and depth_ratio_mean > lighting_ratio_min
        and depth_ratio_std < lighting_ratio_std_max
    )
    if depth_ratio_mean is None:
        reasons.append("lighting_depth_ratio_missing")
    elif depth_ratio_mean <= lighting_ratio_min:
        reasons.append("lighting_depth_ratio_low")
    if depth_ratio_std is None:
        reasons.append("lighting_depth_ratio_std_missing")
    elif depth_ratio_std >= lighting_ratio_std_max:
        reasons.append("lighting_depth_ratio_unstable")

    # Gate 2: distance_stable.
    torso_mean = _safe_mean(torso_depths)
    depth_var_ratio: Optional[float] = None
    if torso_depths and torso_mean and torso_mean > 0:
        depth_var_ratio = float((max(torso_depths) - min(torso_depths)) / torso_mean)
    distance_var_max = float(meta.get("quality_gate_distance_var_max", 0.06))
    torso_min_m = float(meta.get("quality_gate_torso_depth_min_m", 1.2))
    torso_max_m = float(meta.get("quality_gate_torso_depth_max_m", 3.5))
    distance_stable = bool(
        depth_var_ratio is not None
        and torso_mean is not None
        and depth_var_ratio < distance_var_max
        and torso_min_m <= torso_mean <= torso_max_m
    )
    if depth_var_ratio is None:
        reasons.append("distance_variation_unknown")
    elif depth_var_ratio >= distance_var_max:
        reasons.append("distance_variation_high")
    if torso_mean is None:
        reasons.append("distance_torso_depth_missing")
    elif not (torso_min_m <= torso_mean <= torso_max_m):
        reasons.append("distance_torso_depth_out_of_range")

    # Gate 3: pose_locked_ok.
    pose_mode = str(meta.get("pose_mode") or "").strip().lower()
    pose_locked_flag = meta.get("pose_locked")
    if isinstance(pose_locked_flag, bool):
        pose_locked_ok = bool(pose_locked_flag)
    elif pose_mode:
        pose_locked_ok = pose_mode == "locked"
    else:
        pose_locked_ok = True
        reasons.append("pose_lock_unknown_single_frame")
    if pose_locked_ok is False:
        reasons.append("pose_not_locked")

    # Gate 4: pointcloud_quality.
    has_points = any(int(c) > 0 for c in point_counts)
    min_points = int(meta.get("quality_gate_pointcloud_min_points", 80000))
    min_bbox_height = float(meta.get("quality_gate_pointcloud_bbox_height_min_m", 1.1))
    max_holes_ratio = float(meta.get("quality_gate_pointcloud_holes_ratio_max", 0.92))
    depth_proxy_min = float(meta.get("quality_gate_pointcloud_depth_proxy_min", 0.60))

    if has_points:
        per_view_good: List[bool] = []
        for count, bbox_h, holes in zip(point_counts, bbox_heights, holes_ratios):
            if count <= 0:
                continue
            good = bool(
                count >= min_points
                and (bbox_h is None or bbox_h >= min_bbox_height)
                and (holes is None or holes <= max_holes_ratio)
            )
            per_view_good.append(good)
        pointcloud_quality = bool(per_view_good and float(np.mean(per_view_good)) >= 0.75)
        if not pointcloud_quality:
            reasons.append("pointcloud_sparse_or_holes")
    else:
        pointcloud_quality = bool(depth_ratio_mean is not None and depth_ratio_mean >= depth_proxy_min)
        if not pointcloud_quality:
            reasons.append("pointcloud_proxy_depth_ratio_low")

    flow_spike_count, flow_curvature_outliers, flow_score = _flow_spikes(metrics)
    if (
        (flow_spike_count is not None and flow_spike_count >= 4)
        or (flow_curvature_outliers is not None and flow_curvature_outliers >= 3)
        or (flow_score is not None and flow_score < 45.0)
    ):
        pointcloud_quality = False
        reasons.append("flow_spikes_detected")

    # Additional stats from metrics payload.
    silhouette_areas = []
    for v in metrics.get("views") or []:
        try:
            silhouette_areas.append(float(v.get("silhouette_area_m2", 0.0)))
        except Exception:
            continue
    silhouette_mean = _safe_mean(silhouette_areas)
    silhouette_var = None
    if silhouette_areas and silhouette_mean and silhouette_mean > 0:
        silhouette_var = float((max(silhouette_areas) - min(silhouette_areas)) / silhouette_mean)

    stats.update(
        {
            "num_views": int(n_views),
            "depth_valid_ratio_per_view": [None if v is None else round(float(v), 5) for v in depth_ratios_raw],
            "depth_valid_ratio_mean": None if depth_ratio_mean is None else round(depth_ratio_mean, 5),
            "depth_valid_ratio_std": None if depth_ratio_std is None else round(depth_ratio_std, 5),
            "torso_depth_per_view_m": [None if v is None else round(float(v), 4) for v in torso_depths_raw],
            "torso_depth_mean_m": None if torso_mean is None else round(torso_mean, 4),
            "distance_variation_ratio": None if depth_var_ratio is None else round(depth_var_ratio, 5),
            "silhouette_area_mean_m2": None if silhouette_mean is None else round(silhouette_mean, 5),
            "silhouette_area_variation_ratio": None if silhouette_var is None else round(silhouette_var, 5),
            "point_count_per_view": point_counts,
            "point_bbox_height_per_view_m": [None if v is None else round(float(v), 4) for v in bbox_heights],
            "point_holes_ratio_per_view": [None if v is None else round(float(v), 5) for v in holes_ratios],
            "pose_mode": pose_mode or None,
            "flow_spikes": flow_spike_count,
            "flow_curvature_outliers": flow_curvature_outliers,
            "flow_score": None if flow_score is None else round(flow_score, 2),
        }
    )

    return {
        "lighting_stable": bool(lighting_stable),
        "distance_stable": bool(distance_stable),
        "pose_locked_ok": bool(pose_locked_ok),
        "pointcloud_quality": bool(pointcloud_quality),
        "reasons": sorted(set(reasons)),
        "stats": stats,
    }
