from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


_SHADOW_BANDS = {
    "abs": (0.50, 0.65),
    "serratus": (0.60, 0.75),
    "quads": (0.20, 0.35),
    "glute_ham": (0.15, 0.28),
}

_EDGE_BANDS = {
    "waist": (0.55, 0.70),
    "delt_cap": (0.78, 0.92),
    "quad_sweep": (0.28, 0.45),
}

_EDGE_SCALE = 0.35


@dataclass(frozen=True)
class _ViewSignals:
    angle_deg: int
    shadow: Dict[str, float]
    edge: Dict[str, float]


def compute_condition_for_session(session_dir: Path, meta: dict, metrics: dict) -> dict:
    views_meta = meta.get("views") or meta.get("captures") or []
    notes: List[str] = []
    per_view: List[Dict[str, Any]] = []
    view_shadow_scores: List[Tuple[int, float]] = []
    view_edge_scores: List[Tuple[int, float]] = []
    rgb_missing = False
    mask_missing = False

    for v in views_meta:
        angle = int(v.get("angle_deg", 0))
        paths = v.get("paths") or {}
        mask_path = _resolve_path(session_dir, paths.get("mask"))
        rgb_path = _resolve_path(session_dir, paths.get("rgb"))
        if rgb_path is None:
            rgb_path = session_dir / "media" / f"rgb_{angle:03d}.jpg"
        if mask_path is None:
            mask_path = session_dir / "raw" / f"view_{angle:03d}" / f"view_{angle:03d}_mask.png"

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path and mask_path.exists() else None
        rgb = cv2.imread(str(rgb_path)) if rgb_path and rgb_path.exists() else None

        if mask is None:
            mask_missing = True
        if rgb is None:
            rgb_missing = True

        shadow = _compute_shadow_contrast(rgb, mask) if rgb is not None and mask is not None else _zeros(_SHADOW_BANDS)
        edge = _compute_edge_sharpness(mask) if mask is not None else _zeros(_EDGE_BANDS)

        per_view.append(
            {
                "angle_deg": angle,
                "shadow_contrast": shadow,
                "silhouette_edge_sharpness": edge,
            }
        )
        view_shadow_scores.append((angle, float(np.mean(list(shadow.values())))))
        view_edge_scores.append((angle, float(np.mean(list(edge.values())))))

    shadow_med = _weighted_median(view_shadow_scores)
    edge_med = _weighted_median(view_edge_scores)
    condition_score = float(np.clip((0.6 * shadow_med + 0.4 * edge_med) * 100.0, 0.0, 100.0))

    quality_ok = _quality_gates_ok(metrics, views_meta)
    lighting_ok, lighting_note = _lighting_calib_ok(session_dir, meta)
    if lighting_note:
        notes.append(lighting_note)

    if not quality_ok or rgb_missing or mask_missing or not lighting_ok:
        confidence = "low"
    else:
        confidence = "high" if _has_lighting_baseline(session_dir) else "med"

    evidence = _build_evidence(per_view)

    return {
        "version": "0.1",
        "condition_score": round(condition_score, 2),
        "confidence": confidence,
        "evidence": evidence,
        "per_view": per_view,
        "notes": notes,
    }


def _resolve_path(session_dir: Path, path_val: Optional[str]) -> Optional[Path]:
    if not path_val:
        return None
    p = Path(path_val)
    return p if p.is_absolute() else (session_dir / p)


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _band_mask(mask: np.ndarray, band: Tuple[float, float]) -> np.ndarray:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return np.zeros_like(mask, dtype=bool)
    _, y0, _, y1 = bbox
    h = max(1, y1 - y0)
    lo, hi = band
    y_hi = int(y1 - hi * h)
    y_lo = int(y1 - lo * h)
    y_hi = max(0, min(mask.shape[0] - 1, y_hi))
    y_lo = max(0, min(mask.shape[0], y_lo))
    if y_lo <= y_hi:
        y_lo = min(mask.shape[0], y_hi + 1)
    band_mask = np.zeros_like(mask, dtype=bool)
    band_mask[y_hi:y_lo, :] = True
    return band_mask


def _compute_shadow_contrast(rgb: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY).astype(np.float32)
    out: Dict[str, float] = {}
    for name, band in _SHADOW_BANDS.items():
        bm = _band_mask(mask, band)
        sel = gray[(mask > 0) & bm]
        if sel.size < 25:
            out[name] = 0.0
            continue
        p10 = float(np.percentile(sel, 10))
        p90 = float(np.percentile(sel, 90))
        denom = max(1.0, p90)
        val = (p90 - p10) / denom
        out[name] = float(np.clip(val, 0.0, 1.0))
    return out


def _compute_edge_sharpness(mask: np.ndarray) -> Dict[str, float]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return _zeros(_EDGE_BANDS)
    contour = max(contours, key=cv2.contourArea)
    pts = contour[:, 0, :].astype(np.float32)
    if pts.shape[0] < 10:
        return _zeros(_EDGE_BANDS)

    n = pts.shape[0]
    curvatures = np.zeros(n, dtype=np.float32)
    for i in range(n):
        p_prev = pts[(i - 1) % n]
        p_cur = pts[i]
        p_next = pts[(i + 1) % n]
        v1 = p_prev - p_cur
        v2 = p_next - p_cur
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cosang = float(np.dot(v1, v2) / (n1 * n2))
        cosang = max(-1.0, min(1.0, cosang))
        ang = float(np.arccos(cosang))
        curvatures[i] = abs(np.pi - ang)

    out: Dict[str, float] = {}
    for name, band in _EDGE_BANDS.items():
        bm = _band_mask(mask, band)
        xs = np.clip(pts[:, 0].astype(int), 0, mask.shape[1] - 1)
        ys = np.clip(pts[:, 1].astype(int), 0, mask.shape[0] - 1)
        mask_pts = bm[ys, xs]
        sel = curvatures[mask_pts]
        if sel.size < 10:
            out[name] = 0.0
            continue
        mean_curv = float(np.mean(sel))
        out[name] = float(np.clip(mean_curv / _EDGE_SCALE, 0.0, 1.0))
    return out


def _weighted_median(values: List[Tuple[int, float]]) -> float:
    if not values:
        return 0.0
    weights = []
    for angle, _ in values:
        if angle in (0, 180):
            weights.append(1.2)
        else:
            weights.append(1.0)
    pairs = sorted(zip(values, weights), key=lambda p: p[0][1])
    total = sum(w for _, w in pairs)
    acc = 0.0
    for (angle, val), w in pairs:
        acc += w
        if acc >= total / 2.0:
            return float(val)
    return float(pairs[-1][0][1])


def _quality_gates_ok(metrics: dict, views_meta: list) -> bool:
    if not views_meta or len(views_meta) < 4:
        return False
    views = metrics.get("views") or []
    if len(views) < 4:
        return False
    for v in views:
        if float(v.get("silhouette_area_m2", 0.0)) < 0.08:
            return False
    return True


def _gray_hist(path: Path, bins: int = 32) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    hist = cv2.calcHist([img], [0], None, [bins], [0, 256]).astype(np.float32)
    hist /= max(1.0, float(hist.sum()))
    return hist.flatten()


def _user_root(session_dir: Path) -> Optional[Path]:
    parts = session_dir.resolve().parts
    if "sessions" not in parts:
        return None
    idx = parts.index("sessions")
    if idx + 1 >= len(parts):
        return None
    return Path(*parts[: idx + 2])


def _has_lighting_baseline(session_dir: Path) -> bool:
    user_root = _user_root(session_dir)
    if user_root is None:
        return False
    return (user_root / "lighting_baseline.json").exists()


def _lighting_calib_ok(session_dir: Path, meta: dict) -> Tuple[bool, Optional[str]]:
    user_root = _user_root(session_dir)
    if user_root is None:
        return True, None
    baseline_path = user_root / "lighting_baseline.json"
    if not baseline_path.exists():
        return True, None
    try:
        import json

        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        ref = np.array(baseline.get("gray_hist") or [], dtype=np.float32)
    except Exception:
        return True, None
    if ref.size == 0:
        return True, None
    calib_rel = meta.get("rgb_calib_path")
    calib_path = _resolve_path(session_dir, calib_rel) if calib_rel else (session_dir / "media" / "rgb_calib.jpg")
    hist = _gray_hist(calib_path)
    if hist is None or hist.size == 0:
        return False, "Missing rgb_calib for lighting check."
    diff = float(np.sum(np.abs(hist - ref)))
    if diff > 0.25:
        return False, "Lighting calibration mismatch."
    return True, None


def _build_evidence(per_view: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not per_view:
        return []
    front = next((v for v in per_view if v.get("angle_deg") == 0), per_view[0])
    shadow_abs = float((front.get("shadow_contrast") or {}).get("abs", 0.0))
    edge_waist = float((front.get("silhouette_edge_sharpness") or {}).get("waist", 0.0))
    return [
        {"name": "shadow_contrast_abs", "value": round(shadow_abs, 3), "view": int(front.get("angle_deg", 0))},
        {"name": "edge_sharpness_waist", "value": round(edge_waist, 3), "view": int(front.get("angle_deg", 0))},
    ]


def _zeros(bands: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    return {k: 0.0 for k in bands.keys()}
