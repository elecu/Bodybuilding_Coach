from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..vision.source import open_source
from ..core.scan_capture import FourViewScanCapture
from ..core.metrics import compute_metrics_for_scan
from ..core.registration import auto_merge
from ..storage.session_paths import SessionPaths


def run_scan_capture(user: str, out_dir: Path, width: int = 1280, height: int = 720) -> int:
    source = open_source("kinect2", cam_index=None, width=width, height=height)
    source.start()
    paths = SessionPaths(root=out_dir)
    scan_dir = paths.new_scan3d_session_dir(user=user, mode="locked")
    cap = FourViewScanCapture(user, scan_dir, pose_mode="locked")
    window = "BB Coach Scan 4-View"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    try:
        while True:
            packet = source.read()
            rgb = packet.get("rgb_aligned") or packet.get("rgb")
            depth = packet.get("depth")
            intr = (packet.get("meta") or {}).get("intrinsics")
            if rgb is None:
                continue
            if depth is None or intr is None:
                frame = rgb.copy()
                cv2.putText(frame, "Depth/intrinsics missing", (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            status = cap.update(depth, intr, mask=None, rgb_bgr=rgb, timestamp=time.time())
            frame = rgb.copy()
            cv2.putText(frame, status.message, (20, 40), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (255, 230, 160), 2)
            cv2.imshow(window, frame)
            if status.done:
                break
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.finalize()
        print(f"[bbcoach] Scan saved to: {cap.scan_dir}")
        return 0
    finally:
        source.stop()
        cv2.destroyWindow(window)


def run_compute_metrics(scan_dir: Path) -> int:
    meta = scan_dir / "derived" / "meta.json"
    if not meta.exists():
        print("[bbcoach] Missing meta.json")
        return 2
    import json
    meta_obj = json.loads(meta.read_text(encoding="utf-8"))
    views_meta = meta_obj.get("views") or meta_obj.get("captures") or []
    intr = views_meta[0].get("intrinsics", {}) if views_meta else {}
    views = sorted((scan_dir / "raw").glob("view_*"))
    if len(views) < 4:
        print("[bbcoach] Missing view folders")
        return 2
    masks = []
    depths = []
    points = []
    for v in views[:4]:
        mask_path = next(v.glob("*_mask.png"), None)
        depth_path = next(v.glob("*_depth.npy"), None)
        if not mask_path or not depth_path:
            print(f"[bbcoach] Missing artifacts in {v}")
            return 2
        masks.append(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE))
        depths.append(np.load(str(depth_path)))
        points.append(None)
    compute_metrics_for_scan(scan_dir, masks, depths, points, intr)
    print("[bbcoach] Metrics computed.")
    return 0


def run_auto_merge(scan_dir: Path) -> int:
    res = auto_merge(scan_dir)
    print(f"[bbcoach] {res.message} (fitness={res.fitness:.3f}, rmse={res.inlier_rmse:.4f})")
    if not res.ok:
        return 2
    return 0
