from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class ProxyBand:
    name: str
    low_frac: float  # fraction from bottom
    high_frac: float


_BANDS = [
    ProxyBand("upper_torso", 0.55, 0.70),
    ProxyBand("waist", 0.40, 0.55),
    ProxyBand("hips", 0.35, 0.45),
    ProxyBand("thighs", 0.20, 0.35),
    ProxyBand("calves", 0.10, 0.20),
]


def _band_width_px(mask: np.ndarray, low_frac: float, high_frac: float) -> Optional[float]:
    h = mask.shape[0]
    y0 = int(max(0, min(h, (1.0 - high_frac) * h)))
    y1 = int(max(0, min(h, (1.0 - low_frac) * h)))
    if y1 <= y0:
        return None
    widths = []
    for y in range(y0, y1):
        xs = np.where(mask[y] > 0)[0]
        if xs.size < 10:
            continue
        widths.append(float(xs.max() - xs.min()))
    if not widths:
        return None
    return float(np.mean(widths))


def compute_body_proxy(
    mask: Optional[np.ndarray],
    height_cm: float,
    weight_kg: Optional[float] = None,
    proportions: Optional[object] = None,
) -> Dict[str, object]:
    if mask is None or height_cm <= 0:
        return {}

    h = float(mask.shape[0])
    px_to_cm = height_cm / h

    regions: Dict[str, Dict[str, Optional[float]]] = {}
    for band in _BANDS:
        width_px = _band_width_px(mask, band.low_frac, band.high_frac)
        if width_px is None:
            regions[band.name] = {
                "width_px": None,
                "width_cm": None,
                "ratio": None,
            }
            continue
        width_cm = width_px * px_to_cm
        regions[band.name] = {
            "width_px": float(width_px),
            "width_cm": float(width_cm),
            "ratio": float(width_cm / height_cm),
        }

    props_out: Dict[str, Optional[float]] = {}
    if proportions is not None:
        for key in ("shoulder_to_waist", "chest_to_waist", "hip_to_waist", "upper_to_lower_area"):
            try:
                props_out[key] = float(getattr(proportions, key)) if getattr(proportions, key) is not None else None
            except Exception:
                props_out[key] = None

    return {
        "type": "proxy",
        "height_cm": float(height_cm),
        "weight_kg": float(weight_kg) if weight_kg is not None else None,
        "proxy_regions": regions,
        "proportions": props_out or None,
    }
