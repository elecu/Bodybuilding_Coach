from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from ..core.scan_capture import FourViewScanCapture, ScanStatus
from ..storage.session_paths import SessionPaths


@dataclass
class ScanWizard:
    user: str
    out_root: Path
    pose_name: Optional[str] = None
    capture: Optional[FourViewScanCapture] = None
    status: Optional[ScanStatus] = None
    scan_dir: Optional[Path] = None
    active: bool = False

    def start(self) -> None:
        paths = SessionPaths.default()
        if self.out_root.name == "sessions":
            paths = SessionPaths(root=self.out_root)
        scan_dir = paths.new_scan3d_session_dir(user=self.user, mode="locked")
        self.capture = FourViewScanCapture(
            self.user,
            scan_dir,
            pose_name=self.pose_name,
            pose_mode="locked",
        )
        self.active = True
        self.status = None
        self.scan_dir = None

    def update(
        self,
        depth: Optional[np.ndarray],
        intrinsics: Optional[Dict[str, Any]],
        mask: Optional[np.ndarray],
        rgb: Optional[np.ndarray],
    ) -> ScanStatus:
        if not self.capture:
            self.start()
        assert self.capture is not None
        if depth is None or intrinsics is None:
            self.status = ScanStatus("Depth not available.", 0, 4, ready=False, done=False)
            return self.status
        self.status = self.capture.update(depth, intrinsics, mask, rgb)
        if self.capture.done:
            self.scan_dir = self.capture.scan_dir
        return self.status

    def draw_overlay(self, frame: np.ndarray) -> None:
        if not self.status:
            return
        msg = self.status.message
        h, w = frame.shape[:2]
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_TRIPLEX, 0.9, 2)
        x = max(12, (w - tw) // 2)
        y = max(40, int(h * 0.12))
        cv2.rectangle(frame, (x - 12, y - th - 12), (x + tw + 12, y + 12), (0, 0, 0), -1)
        cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (255, 230, 160), 2, cv2.LINE_AA)
