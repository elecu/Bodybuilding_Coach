from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np

from .foreground_segmentation import extract_foreground
from .pcd_io import ensure_dir, save_debug_images, save_meta, save_pointcloud
from ..vision.pointcloud import depth_to_pointcloud


@dataclass
class ViewCapture:
    angle_deg: int
    captured: bool = False
    timestamp: Optional[float] = None
    view_dir: Optional[Path] = None
    z_center: Optional[float] = None
    margin_px: int = 0
    paths: Dict[str, str] = field(default_factory=dict)


@dataclass
class ScanStatus:
    message: str
    step_idx: int
    total_steps: int
    ready: bool = False
    done: bool = False
    captured_now: bool = False


class FourViewScanCapture:
    def __init__(
        self,
        user: str,
        scan_dir: Path,
        pose_name: Optional[str] = None,
        pose_mode: Literal["locked", "free"] = "locked",
        voxel_size_m: float = 0.008,
        stability_seconds: float = 1.0,
        stability_rms_m: float = 0.01,
        rgb_burst_frames: int = 12,
        exposure_locked: bool = False,
        lighting_profile_id: str = "default",
        pump_state: str = "none",
    ) -> None:
        self.user = user
        self.pose_name = pose_name or "pose"
        self.pose_mode = pose_mode
        self.scan_dir = scan_dir
        self.scan_dir.mkdir(parents=True, exist_ok=True)
        self.voxel_size_m = voxel_size_m
        self.stability_seconds = stability_seconds
        self.stability_rms_m = stability_rms_m

        self._angles = [0, 90, 180, 270]
        self._views = [ViewCapture(angle_deg=a) for a in self._angles]
        self._step = 0
        self._last_depth: Optional[np.ndarray] = None
        self._stable_since: Optional[float] = None
        self._rgb_buffer: Deque[np.ndarray] = deque(maxlen=max(1, int(rgb_burst_frames)))
        self._rgb_calib_saved = False
        self._rgb_calib_path: Optional[Path] = None

        self.raw_dir = ensure_dir(self.scan_dir / "raw")
        self.derived_dir = ensure_dir(self.scan_dir / "derived")
        self.media_dir = ensure_dir(self.scan_dir / "media")
        self.reports_dir = ensure_dir(self.scan_dir / "reports")
        self.exports_dir = ensure_dir(self.scan_dir / "exports")
        for angle in self._angles:
            ensure_dir(self.raw_dir / f"view_{angle:03d}")

        self._meta: Dict[str, Any] = {
            "user": user,
            "pose": self.pose_name,
            "pose_id": self.pose_name,
            "pose_mode": self.pose_mode,
            "exposure_locked": exposure_locked,
            "lighting_profile_id": lighting_profile_id,
            "pump_state": pump_state,
            "rgb_saved": False,
            "rgb_burst_frames": int(rgb_burst_frames),
            "voxel_size_m": voxel_size_m,
            "angles": self._angles,
            "views": [],
            "captures": [],
        }

    @property
    def done(self) -> bool:
        return self._step >= len(self._views)

    def status(self) -> ScanStatus:
        if self.done:
            return ScanStatus("Captured ✅ (4/4)", self._step, len(self._views), ready=True, done=True)
        view = self._views[self._step]
        if view.captured:
            msg = f"Captured ✅ ({self._step + 1}/4)"
        elif self._step == 0:
            msg = "Freeze… Capturing (1/4)"
        else:
            msg = f"Turn 90°… Freeze… Capturing ({self._step + 1}/4)"
        return ScanStatus(msg, self._step, len(self._views), ready=False, done=False)

    def update(
        self,
        depth_m: np.ndarray,
        intrinsics: Optional[Dict[str, Any]],
        mask: Optional[np.ndarray],
        rgb_bgr: Optional[np.ndarray],
        timestamp: Optional[float] = None,
    ) -> ScanStatus:
        if self.done:
            return self.status()
        ts = timestamp or time.time()

        if depth_m is None or depth_m.size == 0 or intrinsics is None:
            return ScanStatus("Depth/intrinsics not available.", self._step, len(self._views), ready=False, done=False)

        if rgb_bgr is not None:
            self._push_rgb(rgb_bgr)

        # Stability check on masked depth.
        stable = self._is_stable(depth_m, mask, ts)
        if not stable:
            return self.status()

        view = self._views[self._step]
        if not view.captured:
            self._capture_view(view, depth_m, intrinsics, mask, rgb_bgr, ts)
            self._step += 1
            st = self.status()
            st.captured_now = True
            return st
        return self.status()

    def capture_now(
        self,
        depth_m: np.ndarray,
        intrinsics: Optional[Dict[str, Any]],
        mask: Optional[np.ndarray],
        rgb_bgr: Optional[np.ndarray],
        timestamp: Optional[float] = None,
    ) -> bool:
        if self.done:
            return False
        if depth_m is None or depth_m.size == 0 or intrinsics is None:
            return False
        if rgb_bgr is not None:
            self._push_rgb(rgb_bgr)
        view = self._views[self._step]
        if view.captured:
            return False
        ts = timestamp or time.time()
        self._capture_view(view, depth_m, intrinsics, mask, rgb_bgr, ts)
        if view.captured:
            self._step += 1
            return True
        return False

    def _is_stable(self, depth_m: np.ndarray, mask: Optional[np.ndarray], ts: float) -> bool:
        depth = depth_m.astype(np.float32, copy=False)
        if mask is not None and mask.shape[:2] == depth.shape[:2]:
            valid = (mask > 0) & np.isfinite(depth) & (depth > 0)
        else:
            valid = np.isfinite(depth) & (depth > 0)

        if self._last_depth is None:
            self._last_depth = depth.copy()
            self._stable_since = None
            return False

        if not np.any(valid):
            self._stable_since = None
            self._last_depth = depth.copy()
            return False

        diff = depth - self._last_depth
        rms = float(np.sqrt(np.mean((diff[valid]) ** 2)))
        self._last_depth = depth.copy()
        if rms < self.stability_rms_m:
            if self._stable_since is None:
                self._stable_since = ts
            if (ts - self._stable_since) >= self.stability_seconds:
                return True
        else:
            self._stable_since = None
        return False

    def _capture_view(
        self,
        view: ViewCapture,
        depth_m: np.ndarray,
        intrinsics: Dict[str, Any],
        mask: Optional[np.ndarray],
        rgb_bgr: Optional[np.ndarray],
        ts: float,
    ) -> None:
        if view.angle_deg == self._angles[0]:
            self._ensure_rgb_calib(rgb_bgr)
        fg = extract_foreground(depth_m, intrinsics, mask=mask)
        if fg.mask is None or fg.depth_clean is None:
            # Hard fallback: use depth-only mask.
            depth_only = (depth_m > 0).astype(np.uint8) * 255
            fg_mask = depth_only
            fg_depth = np.where(depth_m > 0, depth_m, 0.0).astype(np.float32)
            fg = type(fg)(
                mask=fg_mask,
                depth_clean=fg_depth,
                z_center=float(np.median(depth_m[depth_m > 0])) if np.any(depth_m > 0) else None,
                margin_px=4,
                debug_mask=None,
                debug_depth=None,
            )

        pts, cols = depth_to_pointcloud(
            fg.depth_clean,
            intrinsics=intrinsics,
            mask=fg.mask,
            rgb_bgr=rgb_bgr,
            stride=2,
        )
        if pts is None or pts.size == 0:
            # Last resort: use raw depth without mask.
            pts, cols = depth_to_pointcloud(
                fg.depth_clean,
                intrinsics=intrinsics,
                mask=None,
                rgb_bgr=rgb_bgr,
                stride=2,
            )
        view_dir = ensure_dir(self.raw_dir / f"view_{view.angle_deg:03d}")
        name = f"view_{view.angle_deg:03d}"
        if pts is None or pts.size == 0:
            return
        paths = save_pointcloud(view_dir, name, pts, cols)
        # Convenience alias expected by some tools/users.
        save_pointcloud(view_dir, f"{name}_depth", pts, None)
        depth_path = view_dir / f"{name}_depth.npy"
        np.save(str(depth_path), fg.depth_clean)
        # Ensure write happened.
        if not (view_dir / f"{name}.pcd").exists():
            return
        debug = save_debug_images(
            view_dir,
            name,
            {"mask": fg.debug_mask, "depth": fg.debug_depth},
        )
        rgb_path, used_frames = self._save_rgb_burst(view.angle_deg, rgb_bgr, mask=fg.mask)
        view.captured = True
        view.timestamp = ts
        view.view_dir = view_dir
        view.z_center = fg.z_center
        view.margin_px = fg.margin_px
        raw_paths = {**paths, **debug, "depth_npy": str(depth_path)}
        if rgb_path is not None:
            raw_paths["rgb"] = str(rgb_path)
            self._meta["rgb_saved"] = True
        view.paths = self._rel_paths(raw_paths)
        view_payload = {
            "angle_deg": view.angle_deg,
            "timestamp": ts,
            "dir": self._rel_path(view_dir),
            "paths": view.paths,
            "z_center_m": fg.z_center,
            "margin_px": fg.margin_px,
            "frame_shape": list(depth_m.shape[:2]),
            "intrinsics": intrinsics,
        }
        if used_frames:
            view_payload["rgb_burst_used"] = int(used_frames)
        self._meta["views"].append(view_payload)
        self._meta["captures"].append(view_payload)
        save_meta(self.derived_dir, self._meta)
        # Prevent cross-view ghosting in saved RGB media by resetting burst state per capture.
        self._rgb_buffer.clear()

    def finalize(self) -> Path:
        return save_meta(self.derived_dir, self._meta)

    def _push_rgb(self, rgb_bgr: np.ndarray) -> None:
        if rgb_bgr is None or rgb_bgr.size == 0:
            return
        self._rgb_buffer.append(rgb_bgr.copy())

    def _burst_average(self) -> tuple[Optional[np.ndarray], int]:
        if not self._rgb_buffer:
            return None, 0
        stack = np.stack([f.astype(np.float32) for f in self._rgb_buffer], axis=0)
        avg = np.mean(stack, axis=0)
        avg = np.clip(avg, 0, 255).astype(np.uint8)
        return avg, len(self._rgb_buffer)

    def _ensure_rgb_calib(self, rgb_bgr: Optional[np.ndarray]) -> None:
        if self._rgb_calib_saved:
            return
        img, used = self._burst_average()
        if img is None and rgb_bgr is not None:
            img = rgb_bgr
            used = 1
        if img is None:
            return
        calib_path = self.media_dir / "rgb_calib.jpg"
        cv2.imwrite(str(calib_path), img)
        self._rgb_calib_saved = True
        self._rgb_calib_path = calib_path
        self._meta["rgb_calib_path"] = self._rel_path(calib_path)

    def _save_rgb_burst(
        self,
        angle_deg: int,
        rgb_bgr: Optional[np.ndarray],
        mask: Optional[np.ndarray] = None,
    ) -> tuple[Optional[Path], int]:
        img, used = self._burst_average()
        if img is None and rgb_bgr is not None:
            img = rgb_bgr
            used = 1
        if img is None:
            return None, 0
        # Optional white background cutout for cleaner per-view report images.
        if mask is not None and mask.shape[:2] == img.shape[:2]:
            m = (mask > 0).astype(np.uint8)
            if np.any(m):
                out = np.full_like(img, 255)
                out[m > 0] = img[m > 0]
                img = out
        out_path = self.media_dir / f"rgb_{angle_deg:03d}.jpg"
        cv2.imwrite(str(out_path), img)
        return out_path, used

    def _rel_path(self, path: Path) -> str:
        try:
            return path.relative_to(self.scan_dir).as_posix()
        except Exception:
            return str(path)

    def _rel_paths(self, paths: Dict[str, str]) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for key, val in paths.items():
            if not val:
                continue
            p = Path(val)
            out[key] = self._rel_path(p)
        return out
