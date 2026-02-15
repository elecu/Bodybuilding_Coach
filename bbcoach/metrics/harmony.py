from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def _profile_arrays(width_profile_m: List[float], height_bins: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    widths = np.asarray(width_profile_m or [], dtype=np.float32)
    h_top = np.asarray(height_bins or [], dtype=np.float32)
    if widths.size == 0 or h_top.size == 0 or widths.size != h_top.size:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    valid = np.isfinite(widths) & np.isfinite(h_top) & (widths > 0.0)
    if not np.any(valid):
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    widths = widths[valid]
    h_bottom = 1.0 - h_top[valid]
    return widths, h_bottom


def _smooth_series(values: np.ndarray, window: int = 5) -> np.ndarray:
    if values.size == 0:
        return values
    k = int(max(3, window))
    if k % 2 == 0:
        k += 1
    if k > values.size:
        k = int(values.size if values.size % 2 == 1 else values.size - 1)
    if k < 3:
        return values.copy()
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(values, kernel, mode="same")


def _band_mask(h_bottom: np.ndarray, low: float, high: float) -> np.ndarray:
    if h_bottom.size == 0:
        return np.asarray([], dtype=bool)
    lo = min(low, high)
    hi = max(low, high)
    return (h_bottom >= lo) & (h_bottom <= hi)


def _mean_band(widths: np.ndarray, h_bottom: np.ndarray, low: float, high: float) -> Optional[float]:
    sel = _band_mask(h_bottom, low, high)
    if sel.size == 0 or not np.any(sel):
        return None
    return float(np.mean(widths[sel]))


def _argmin_band(widths: np.ndarray, h_bottom: np.ndarray, low: float, high: float) -> Tuple[Optional[float], Optional[float]]:
    sel = _band_mask(h_bottom, low, high)
    if sel.size == 0 or not np.any(sel):
        return None, None
    ys = h_bottom[sel]
    ws = widths[sel]
    order = np.argsort(ws)
    k = min(4, order.size)
    top = order[:k]
    return float(np.mean(ys[top])), float(np.mean(ws[top]))


def _argmax_band(widths: np.ndarray, h_bottom: np.ndarray, low: float, high: float) -> Tuple[Optional[float], Optional[float]]:
    sel = _band_mask(h_bottom, low, high)
    if sel.size == 0 or not np.any(sel):
        return None, None
    ys = h_bottom[sel]
    ws = widths[sel]
    order = np.argsort(-ws)
    k = min(4, order.size)
    top = order[:k]
    return float(np.mean(ys[top])), float(np.mean(ws[top]))


def compute_landmarks_1d(width_profile_m: List[float], height_bins: List[float]) -> Dict[str, Optional[float]]:
    widths, h_bottom = _profile_arrays(width_profile_m, height_bins)
    if widths.size == 0:
        return {
            "y_shoulder": None,
            "y_chest": None,
            "y_waist": None,
            "y_hip": None,
            "y_mid_thigh": None,
            "y_calf": None,
        }

    ws_s = _smooth_series(widths, window=5)

    y_waist, _ = _argmin_band(ws_s, h_bottom, 0.45, 0.70)
    if y_waist is None:
        y_waist, _ = _argmin_band(ws_s, h_bottom, 0.40, 0.72)
    y_hip, _ = _argmax_band(ws_s, h_bottom, 0.35, 0.55)
    y_shoulder, _ = _argmax_band(ws_s, h_bottom, 0.70, 0.92)
    if y_shoulder is None:
        y_shoulder, _ = _argmax_band(ws_s, h_bottom, 0.60, 0.92)
    y_chest, _ = _argmax_band(ws_s, h_bottom, 0.58, 0.80)
    if y_chest is None:
        y_chest, _ = _argmax_band(ws_s, h_bottom, 0.52, 0.82)

    y_mid_thigh = None
    thigh_sel = _band_mask(h_bottom, 0.25, 0.40)
    if thigh_sel.size and np.any(thigh_sel):
        y_mid_thigh = float(np.mean(h_bottom[thigh_sel]))

    y_calf = None
    calf_sel = _band_mask(h_bottom, 0.12, 0.25)
    if calf_sel.size and np.any(calf_sel):
        y_calf = float(np.mean(h_bottom[calf_sel]))

    return {
        "y_shoulder": y_shoulder,
        "y_chest": y_chest,
        "y_waist": y_waist,
        "y_hip": y_hip,
        "y_mid_thigh": y_mid_thigh,
        "y_calf": y_calf,
    }


def compute_profile_widths(width_profile_m: List[float], height_bins: List[float]) -> Dict[str, Optional[float]]:
    widths, h_bottom = _profile_arrays(width_profile_m, height_bins)
    if widths.size == 0:
        return {
            "shoulder_width": None,
            "chest_width": None,
            "waist_width": None,
            "hip_width": None,
            "thigh_width": None,
            "calf_width": None,
        }
    ws_s = _smooth_series(widths, window=5)
    _, waist_w = _argmin_band(ws_s, h_bottom, 0.45, 0.70)
    if waist_w is None:
        _, waist_w = _argmin_band(ws_s, h_bottom, 0.40, 0.72)
    _, hip_w = _argmax_band(ws_s, h_bottom, 0.35, 0.55)
    _, shoulder_w = _argmax_band(ws_s, h_bottom, 0.70, 0.92)
    if shoulder_w is None:
        _, shoulder_w = _argmax_band(ws_s, h_bottom, 0.60, 0.92)
    _, chest_w = _argmax_band(ws_s, h_bottom, 0.58, 0.80)
    if chest_w is None:
        _, chest_w = _argmax_band(ws_s, h_bottom, 0.52, 0.82)
    thigh_w = _mean_band(widths, h_bottom, 0.25, 0.40)
    calf_w = _mean_band(widths, h_bottom, 0.12, 0.25)
    return {
        "shoulder_width": shoulder_w,
        "chest_width": chest_w,
        "waist_width": waist_w,
        "hip_width": hip_w,
        "thigh_width": thigh_w,
        "calf_width": calf_w,
    }


def compute_flow_features(width_profile_m: List[float], height_bins: List[float], smooth_window: int = 7) -> Dict[str, Any]:
    widths, h_bottom = _profile_arrays(width_profile_m, height_bins)
    band = _band_mask(h_bottom, 0.12, 0.92)
    if band.size == 0 or not np.any(band):
        return {"flow_score": None, "stats": {}}

    ws = widths[band]
    hs = h_bottom[band]
    order = np.argsort(hs)
    hs = hs[order]
    ws = ws[order]
    if ws.size < 8:
        return {"flow_score": None, "stats": {"points_used": int(ws.size)}}

    k = int(max(3, smooth_window))
    if k % 2 == 0:
        k += 1
    if k > ws.size:
        k = int(ws.size if ws.size % 2 == 1 else ws.size - 1)
    if k < 3:
        k = 3
    kernel = np.ones(k, dtype=np.float32) / float(k)
    ws_smooth = np.convolve(ws, kernel, mode="same")

    d1 = np.gradient(ws_smooth)
    d2 = np.gradient(d1)

    eps = max(1e-5, float(np.median(np.abs(d1))) * 0.15)
    signs = np.sign(d1)
    signs[np.abs(d1) < eps] = 0
    nz = signs[signs != 0]
    sign_changes = 0
    if nz.size >= 2:
        sign_changes = int(np.sum(nz[1:] != nz[:-1]))

    curv_abs = np.abs(d2)
    median = float(np.median(curv_abs))
    mad = float(np.median(np.abs(curv_abs - median))) + 1e-6
    curv_thr = median + 3.5 * mad
    curvature_outliers = int(np.sum(curv_abs > curv_thr))

    p95 = float(np.percentile(curv_abs, 95))
    spike_thr = max(curv_thr, p95 * 1.20)
    spikes = int(np.sum(curv_abs > spike_thr))

    zigzag_excess = max(0, sign_changes - 3)
    penalty = float(zigzag_excess * 7 + curvature_outliers * 8 + spikes * 6)
    flow_score = _clamp(100.0 - penalty)

    return {
        "flow_score": round(float(flow_score), 2),
        "stats": {
            "points_used": int(ws.size),
            "smooth_window": int(k),
            "sign_changes": int(sign_changes),
            "zigzag_excess": int(zigzag_excess),
            "curvature_outliers": int(curvature_outliers),
            "spikes": int(spikes),
            "spike_threshold": round(float(spike_thr), 6),
            "curvature_threshold": round(float(curv_thr), 6),
        },
    }


def compute_lr_symmetry_from_mask(mask: Optional[np.ndarray]) -> Dict[str, Any]:
    if mask is None or mask.size == 0:
        return {"symmetry_score": None, "stats": {}}
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return {"symmetry_score": None, "stats": {}}

    x_min, x_max = int(np.min(xs)), int(np.max(xs))
    y_min, y_max = int(np.min(ys)), int(np.max(ys))
    span = max(1, y_max - y_min + 1)
    x_center = int(round((x_min + x_max) / 2.0))

    bands = {
        "upper": (0.62, 0.84),
        "thigh": (0.25, 0.40),
        "calf": (0.12, 0.25),
    }

    deviations: Dict[str, float] = {}
    similarities: List[float] = []

    for name, (low, high) in bands.items():
        y_hi = y_max - int(round(high * span))
        y_lo = y_max - int(round(low * span))
        y_hi = max(0, min(mask.shape[0] - 1, y_hi))
        y_lo = max(y_hi + 1, min(mask.shape[0], y_lo))
        region = mask[y_hi:y_lo, x_min : x_max + 1]
        if region.size == 0:
            continue
        rel_center = max(0, min(region.shape[1] - 1, x_center - x_min))
        left = region[:, : rel_center + 1]
        right = region[:, rel_center + 1 :]
        l_area = float(np.count_nonzero(left > 0))
        r_area = float(np.count_nonzero(right > 0))
        denom = l_area + r_area
        if denom <= 0:
            continue
        dev = abs(l_area - r_area) / denom
        sim = 1.0 - dev
        deviations[name] = float(dev)
        similarities.append(float(sim))

    if not similarities:
        return {"symmetry_score": None, "stats": {}}

    sym_score = _clamp(float(np.mean(similarities)) * 100.0)
    stats: Dict[str, Any] = {}
    for name, dev in deviations.items():
        stats[f"{name}_deviation"] = round(float(dev), 5)
        stats[f"{name}_deviation_pct"] = round(float(dev) * 100.0, 2)
    return {"symmetry_score": round(float(sym_score), 2), "stats": stats}
