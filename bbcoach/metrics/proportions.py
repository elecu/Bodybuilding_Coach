from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ProportionMetrics:
    shoulder_to_waist: Optional[float]
    chest_to_waist: Optional[float]
    hip_to_waist: Optional[float]
    upper_to_lower_area: Optional[float]


def _width_at(mask: np.ndarray, y: int) -> Optional[int]:
    if y < 0 or y >= mask.shape[0]:
        return None
    row = mask[y, :]
    xs = np.where(row > 0)[0]
    if xs.size < 10:
        return None
    return int(xs.max() - xs.min())


def compute_from_mask(mask: np.ndarray) -> ProportionMetrics:
    """Compute rough ratios from a binary person mask.

    Works best with consistent camera distance, lighting, and tight clothing.
    The output is mainly for *trend tracking*, not absolute measurements.
    """
    h, w = mask.shape[:2]

    # sample widths at relative heights
    y_sh = int(h * 0.25)
    y_ch = int(h * 0.33)
    y_wa = int(h * 0.45)
    y_hp = int(h * 0.55)

    ws = _width_at(mask, y_sh)
    wc = _width_at(mask, y_ch)
    ww = _width_at(mask, y_wa)
    wh = _width_at(mask, y_hp)

    def ratio(a, b):
        if a is None or b is None or b == 0:
            return None
        return float(a) / float(b)

    shoulder_to_waist = ratio(ws, ww)
    chest_to_waist = ratio(wc, ww)
    hip_to_waist = ratio(wh, ww)

    # upper vs lower silhouette area split at waist
    if ww is None:
        split = int(h * 0.5)
    else:
        split = y_wa

    upper = float(mask[:split, :].sum())
    lower = float(mask[split:, :].sum())
    upper_to_lower_area = (upper / lower) if lower > 1 else None

    return ProportionMetrics(
        shoulder_to_waist=shoulder_to_waist,
        chest_to_waist=chest_to_waist,
        hip_to_waist=hip_to_waist,
        upper_to_lower_area=upper_to_lower_area,
    )
