"""Video sources.

Right now we support a standard webcam via OpenCV.

Design note:
- This module exists so we can later add Kinect v2 (libfreenect2) without
  touching the pose/metrics/coach logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union
import time

import cv2
import numpy as np


Frame = np.ndarray  # BGR uint8


class FrameSource:
    def open(self) -> None:
        raise NotImplementedError

    def read(self) -> Tuple[bool, Optional[Frame], float]:
        """Return (ok, frame, timestamp_seconds)."""
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


@dataclass
class OpenCVCameraSource(FrameSource):
    # Accept either an integer index (0, 1, 2, ...) or a Linux device path
    # such as /dev/video2.
    device: Union[int, str] = 0
    width: int = 1280
    height: int = 720
    fps: int = 30

    # On Linux, forcing CAP_V4L2 often avoids backend auto-selection issues.
    backend: int = cv2.CAP_V4L2

    _cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        cap = cv2.VideoCapture(self.device, self.backend)
        if not cap.isOpened():
            # Fallback to OpenCV auto backend.
            cap.release()
            cap = cv2.VideoCapture(self.device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        cap.set(cv2.CAP_PROP_FPS, float(self.fps))
        self._cap = cap

        if not cap.isOpened():
            raise RuntimeError(
                "OpenCV could not open the camera. Try a different --camera value (e.g. 1, 2, 3) "
                "or a device path like /dev/video2. Also check that no other app is using the camera."
            )

    def read(self) -> Tuple[bool, Optional[Frame], float]:
        if self._cap is None:
            return False, None, time.time()
        ok, frame = self._cap.read()
        return ok, frame if ok else None, time.time()

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class KinectV2Source(FrameSource):
    """Placeholder for a future Kinect v2 source.

    Implementation idea:
    - Use libfreenect2 / pylibfreenect2 to fetch colour + depth.
    - Return a colour frame for pose tracking, and optionally a depth map
      for volumetric/proportion metrics.

    We keep this stub so the rest of the program can depend on the interface.
    """

    def open(self) -> None:
        raise RuntimeError("KinectV2Source is not implemented in this webcam-first build.")

    def read(self) -> Tuple[bool, Optional[Frame], float]:
        return False, None, time.time()

    def close(self) -> None:
        return
