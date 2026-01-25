from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PoseLandmarks:
    # Normalised coords in [0,1] (x,y)
    pts: Dict[str, Tuple[float, float]]


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    # angle ABC in degrees
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-8:
        return 0.0
    cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosang))


def compute_features(lm: PoseLandmarks) -> Dict[str, float]:
    p = lm.pts

    # Required landmarks
    needed = [
        "left_shoulder", "right_shoulder",
        "left_hip", "right_hip",
        "left_elbow", "right_elbow",
        "nose",
    ]
    for k in needed:
        if k not in p:
            return {}

    ls = np.array(p["left_shoulder"], dtype=float)
    rs = np.array(p["right_shoulder"], dtype=float)
    lh = np.array(p["left_hip"], dtype=float)
    rh = np.array(p["right_hip"], dtype=float)
    le = np.array(p["left_elbow"], dtype=float)
    re = np.array(p["right_elbow"], dtype=float)
    nose = np.array(p["nose"], dtype=float)

    # Shoulder level: y-diff (degrees proxy). Convert to degrees-ish by scaling.
    shoulder_level = (ls[1] - rs[1]) * 180.0

    # Torso upright: angle between mid-hip->mid-shoulder and vertical
    mid_sh = 0.5 * (ls + rs)
    mid_hip = 0.5 * (lh + rh)
    v = mid_sh - mid_hip
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-8:
        torso_upright = 0.0
    else:
        v = v / v_norm
        # vertical axis is (0,-1) in image coords
        vert = np.array([0.0, -1.0])
        torso_upright = math.degrees(math.acos(float(np.clip(np.dot(v, vert), -1.0, 1.0))))

    # Elbow symmetry: distance from shoulders
    elbow_sym = (np.linalg.norm(le - ls) - np.linalg.norm(re - rs)) * 100.0

    # Elbow height symmetry (relative to shoulders)
    elbow_height = ((le[1] - ls[1]) - (re[1] - rs[1])) * 180.0

    return {
        "shoulder_level": shoulder_level,
        "torso_upright": torso_upright,
        "elbow_sym": elbow_sym,
        "elbow_height": elbow_height,
    }
