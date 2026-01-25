from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


LINKS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]


def _px(pt: Tuple[float, float], w: int, h: int) -> Tuple[int, int]:
    return (int(pt[0] * w), int(pt[1] * h))


def draw_pose_overlay(
    frame: np.ndarray,
    landmarks: Dict[str, Tuple[float, float]],
    joint_ok: Dict[str, bool],
    line_ok: Dict[Tuple[str, str], bool],
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # lines
    for a, b in LINKS:
        if a in landmarks and b in landmarks:
            ok = line_ok.get((a, b), True)
            colour = (0, 200, 0) if ok else (0, 0, 255)
            cv2.line(out, _px(landmarks[a], w, h), _px(landmarks[b], w, h), colour, 3)

    # joints
    for k, pt in landmarks.items():
        ok = joint_ok.get(k, True)
        colour = (0, 200, 0) if ok else (0, 0, 255)
        cv2.circle(out, _px(pt, w, h), 6, colour, -1)

    return out


def draw_mask_outline(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = frame.copy()
    if mask is None:
        return out
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 0), 2)
    return out
