from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def _landmarks_bbox(landmarks: Dict[str, Tuple[float, float]], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    if not landmarks:
        return None
    xs = [int(pt[0] * w) for pt in landmarks.values() if pt is not None]
    ys = [int(pt[1] * h) for pt in landmarks.values() if pt is not None]
    if not xs or not ys:
        return None
    x0 = max(0, min(xs) - int(0.08 * w))
    y0 = max(0, min(ys) - int(0.08 * h))
    x1 = min(w - 1, max(xs) + int(0.08 * w))
    y1 = min(h - 1, max(ys) + int(0.08 * h))
    return (x0, y0, x1, y1)


def segment_person_depth(
    depth_m: np.ndarray,
    rgb_bgr: Optional[np.ndarray],
    landmarks: Optional[Dict[str, Tuple[float, float]]],
    depth_min: float = 0.4,
    depth_max: float = 4.0,
) -> Dict[str, object]:
    if depth_m is None or depth_m.size == 0:
        return {"mask": None, "bbox": None, "conf": 0.0, "depth_clean": None}
    depth = depth_m.astype(np.float32, copy=False)
    h, w = depth.shape[:2]

    valid = np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, 0.0)
    depth = np.where((depth >= depth_min) & (depth <= depth_max), depth, 0.0)

    # Smooth depth (median on a scaled copy).
    depth_mm = (depth * 1000.0).astype(np.uint16)
    depth_mm = cv2.medianBlur(depth_mm, 5)
    depth_sm = depth_mm.astype(np.float32) * 0.001
    depth_sm = np.where(depth_mm > 0, depth_sm, 0.0)

    if not np.any(depth_sm > 0):
        return {"mask": None, "bbox": None, "conf": 0.0, "depth_clean": depth_sm}

    min_depth = float(np.min(depth_sm[depth_sm > 0]))
    thresh = min_depth + 0.6
    cand = (depth_sm > 0) & (depth_sm <= thresh)
    cand_u8 = (cand.astype(np.uint8) * 255)

    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(cand_u8, connectivity=8)
    if num <= 1:
        return {"mask": cand_u8, "bbox": None, "conf": 0.3, "depth_clean": depth_sm}

    roi = _landmarks_bbox(landmarks or {}, w, h)
    best_idx = -1
    best_score = -1.0
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area < 300:
            continue
        score = float(area)
        if roi is not None:
            rx0, ry0, rx1, ry1 = roi
            ix0 = max(x, rx0)
            iy0 = max(y, ry0)
            ix1 = min(x + ww, rx1)
            iy1 = min(y + hh, ry1)
            inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            score += inter * 2.0
        else:
            # Prefer central components when no landmarks.
            cx = x + ww / 2.0
            cy = y + hh / 2.0
            score -= ((cx - w / 2) ** 2 + (cy - h / 2) ** 2) * 0.01
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx <= 0:
        return {"mask": cand_u8, "bbox": None, "conf": 0.3, "depth_clean": depth_sm}

    mask = (labels == best_idx).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    x, y, ww, hh, area = stats[best_idx]
    bbox = (int(x), int(y), int(x + ww), int(y + hh))
    area_ratio = float(area) / float(w * h)
    conf = min(1.0, max(0.1, area_ratio * 6.0))
    return {"mask": mask, "bbox": bbox, "conf": conf, "depth_clean": depth_sm}
