from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class PoseResult:
    landmarks: Dict[str, Tuple[float, float]]
    mask: Optional[np.ndarray]


class PoseBackend:
    """Mediapipe-based pose + (optional) segmentation mask."""

    def __init__(self) -> None:
        # Lazy import so the module can be imported even if mediapipe isn't installed yet
        import mediapipe as mp

        # MediaPipe removed the legacy "Solutions" API from mediapipe>=0.10.30.
        # This project currently depends on mp.solutions.pose.* (pose + segmentation).
        # If you see this error, pin mediapipe to <0.10.30.
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Your mediapipe package does not include the Solutions API (mp.solutions.*). "
                "Install a compatible version, e.g.:\n\n"
                "  pip uninstall -y mediapipe\n"
                "  pip install 'mediapipe<0.10.30' 'numpy<2'\n"
            )

        self.mp = mp
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_bgr(self, frame_bgr: np.ndarray) -> PoseResult:
        mp = self.mp
        frame_rgb = frame_bgr[:, :, ::-1]
        res = self.pose.process(frame_rgb)

        lm: Dict[str, Tuple[float, float]] = {}
        if res.pose_landmarks is not None:
            # Map subset of keypoints
            idx = mp.solutions.pose.PoseLandmark
            def get(i):
                p = res.pose_landmarks.landmark[i]
                return (float(p.x), float(p.y))

            lm = {
                "nose": get(idx.NOSE),
                "left_shoulder": get(idx.LEFT_SHOULDER),
                "right_shoulder": get(idx.RIGHT_SHOULDER),
                "left_elbow": get(idx.LEFT_ELBOW),
                "right_elbow": get(idx.RIGHT_ELBOW),
                "left_wrist": get(idx.LEFT_WRIST),
                "right_wrist": get(idx.RIGHT_WRIST),
                "left_hip": get(idx.LEFT_HIP),
                "right_hip": get(idx.RIGHT_HIP),
                "left_knee": get(idx.LEFT_KNEE),
                "right_knee": get(idx.RIGHT_KNEE),
                "left_ankle": get(idx.LEFT_ANKLE),
                "right_ankle": get(idx.RIGHT_ANKLE),
            }

        mask = None
        if getattr(res, "segmentation_mask", None) is not None:
            m = res.segmentation_mask
            # binarise
            mask = (m > 0.5).astype(np.uint8) * 255

        return PoseResult(landmarks=lm, mask=mask)
