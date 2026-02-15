from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class ForegroundResult:
    mask: Optional[np.ndarray]
    depth_clean: Optional[np.ndarray]
    z_center: Optional[float]
    margin_px: int
    debug_mask: Optional[np.ndarray] = None
    debug_depth: Optional[np.ndarray] = None


def _largest_component(mask: np.ndarray) -> Optional[np.ndarray]:
    if mask is None or mask.size == 0:
        return None
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return mask
    best_idx = -1
    best_area = 0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_idx = i
    if best_idx <= 0:
        return mask
    return (labels == best_idx).astype(np.uint8) * 255


def _trim_low_foot_flare(mask: np.ndarray, flare_ratio: float = 1.45, trim_frac: float = 0.03) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    ys, _ = np.where(mask > 0)
    if ys.size == 0:
        return mask
    y_min = int(np.min(ys))
    y_max = int(np.max(ys))
    span = max(1, y_max - y_min + 1)

    widths = []
    rows = []
    for y in range(y_min, y_max + 1):
        xs = np.where(mask[y] > 0)[0]
        if xs.size < 2:
            continue
        widths.append(float(xs.max() - xs.min() + 1))
        rows.append(y)
    if not widths:
        return mask

    widths_arr = np.asarray(widths, dtype=np.float32)
    rows_arr = np.asarray(rows, dtype=np.int32)
    torso_sel = (rows_arr >= (y_min + int(0.35 * span))) & (rows_arr <= (y_min + int(0.75 * span)))
    if not np.any(torso_sel):
        return mask
    torso_med = float(np.median(widths_arr[torso_sel]))
    if torso_med <= 1.0:
        return mask

    foot_sel = rows_arr >= (y_min + int(0.88 * span))
    if not np.any(foot_sel):
        return mask
    foot_max = float(np.max(widths_arr[foot_sel]))
    if foot_max <= torso_med * flare_ratio:
        return mask

    cut_rows = max(2, int(round(span * trim_frac)))
    cut_y = max(y_min + 1, y_max - cut_rows + 1)
    out = mask.copy()
    out[cut_y : y_max + 1, :] = 0
    return out


def clean_person_mask(mask: Optional[np.ndarray], trim_foot_flare: bool = True) -> Optional[np.ndarray]:
    if mask is None or mask.size == 0:
        return None
    out = (mask > 0).astype(np.uint8) * 255
    out = _largest_component(out)
    if out is None or not np.any(out):
        return out
    if trim_foot_flare:
        out = _trim_low_foot_flare(out)
    out = _largest_component(out)
    return out


def _depth_to_vis(depth_m: np.ndarray) -> np.ndarray:
    depth = depth_m.copy()
    depth[~np.isfinite(depth)] = 0.0
    valid = depth > 0
    if not np.any(valid):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    vmin = float(np.percentile(depth[valid], 5))
    vmax = float(np.percentile(depth[valid], 95))
    if vmax <= vmin:
        vmax = float(depth[valid].max())
        vmin = float(depth[valid].min())
    scaled = np.clip((depth - vmin) / max(1e-6, (vmax - vmin)), 0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def extract_foreground(
    depth_m: np.ndarray,
    intrinsics: Optional[dict],
    mask: Optional[np.ndarray] = None,
    depth_min: float = 0.4,
    depth_max: float = 4.0,
    z_band: float = 0.45,
    margin_m: float = 0.10,
) -> ForegroundResult:
    if depth_m is None or depth_m.size == 0:
        return ForegroundResult(None, None, None, 0)

    depth = depth_m.astype(np.float32, copy=False)
    depth = np.where(np.isfinite(depth), depth, 0.0)
    depth = np.where((depth >= depth_min) & (depth <= depth_max), depth, 0.0)

    base_mask: Optional[np.ndarray] = None
    if mask is not None and mask.shape[:2] == depth.shape[:2]:
        base_mask = clean_person_mask(mask)
    if base_mask is None or not np.any(base_mask):
        # Fallback: depth-only segmentation, largest connected component.
        depth_bin = (depth > 0).astype(np.uint8) * 255
        base_mask = clean_person_mask(depth_bin)
    if base_mask is None or not np.any(base_mask):
        return ForegroundResult(None, depth, None, 0)

    # Median depth of masked region; if empty, use depth-only median.
    z_vals = depth[base_mask > 0]
    z_vals = z_vals[z_vals > 0]
    if z_vals.size == 0:
        z_vals = depth[depth > 0]
        if z_vals.size == 0:
            return ForegroundResult(None, depth, None, 0)
        base_mask = (depth > 0).astype(np.uint8) * 255
    z_center = float(np.median(z_vals))

    # Margin in pixels from approximate 10cm at Z.
    fx = float(intrinsics.get("fx", 0.0)) if intrinsics else 0.0
    if fx <= 0:
        fx = 365.0  # Kinect depth approx (fallback)
    margin_px = int(round(fx * margin_m / max(0.2, z_center)))
    margin_px = max(2, min(80, margin_px))

    kernel = np.ones((margin_px, margin_px), np.uint8)
    mask_dilated = cv2.dilate(base_mask, kernel, iterations=1)

    z_min = max(depth_min, z_center - z_band)
    z_max = min(depth_max, z_center + z_band)
    z_keep = (depth >= z_min) & (depth <= z_max)
    fg_mask = (mask_dilated > 0) & z_keep
    fg_mask_u8 = (fg_mask.astype(np.uint8) * 255)
    cleaned_mask = clean_person_mask(fg_mask_u8)
    if cleaned_mask is not None and np.any(cleaned_mask):
        fg_mask_u8 = cleaned_mask
        fg_mask = fg_mask_u8 > 0

    depth_clean = np.where(fg_mask, depth, 0.0).astype(np.float32)

    debug_mask = cv2.cvtColor(fg_mask_u8, cv2.COLOR_GRAY2BGR)
    debug_depth = _depth_to_vis(depth_clean)

    return ForegroundResult(
        mask=fg_mask_u8,
        depth_clean=depth_clean,
        z_center=z_center,
        margin_px=margin_px,
        debug_mask=debug_mask,
        debug_depth=debug_depth,
    )
