"""Video sources.

Supported backends:
- V4L2 (OpenCV) for standard webcams.
- Kinect v2 (libfreenect2) for RGB + depth + optional IR.
"""

from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os
import time
import traceback
from typing import Any, Dict, Optional, Protocol, Union

import cv2
import numpy as np

from . import capture_kinect


SourceFrame = Dict[str, Any]


class SourceObject(Protocol):
    def start(self) -> None: ...

    def read(self) -> SourceFrame: ...

    def stop(self) -> None: ...


_TRUTHY = {"1", "true", "yes", "on"}


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in _TRUTHY


def _kinect_runtime_config(depth_align: bool) -> dict[str, bool]:
    align = bool(depth_align)
    align_env = os.environ.get("BBCOACH_KINECT_ALIGN")
    if align_env is not None and str(align_env).strip():
        align = str(align_env).strip().lower() in _TRUTHY
    return {
        "align": align,
        "enable_depth": _env_flag("BBCOACH_KINECT_ENABLE_DEPTH", default=True),
        "enable_ir": _env_flag("BBCOACH_KINECT_ENABLE_IR", default=True),
    }


def open_source(
    kind: str,
    cam_index: int | str | None,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    depth_align: bool = True,
    depth_scale: str = "meters",
) -> SourceObject:
    kind_norm = (kind or "v4l2").strip().lower()
    if kind_norm == "v4l2":
        if cam_index is None:
            raise ValueError("cam_index is required for v4l2 sources.")
        if isinstance(cam_index, str) and cam_index.isdigit():
            cam_index = int(cam_index)
        return V4L2Source(device=cam_index, width=width, height=height, fps=fps)
    if kind_norm in ("kinect2", "kinect"):
        if depth_scale != "meters":
            raise ValueError("Kinect2Source only supports depth_scale='meters'.")
        if _env_flag("BBCOACH_KINECT_ISOLATE", default=False):
            return Kinect2IsolatedSource(
                width=width,
                height=height,
                fps=fps,
                depth_align=depth_align,
                depth_scale=depth_scale,
            )
        return Kinect2Source(
            width=width,
            height=height,
            fps=fps,
            depth_align=depth_align,
            depth_scale=depth_scale,
        )
    raise ValueError(f"Unknown source kind: {kind}")


@dataclass
class V4L2Source:
    # Accept either an integer index (0, 1, 2, ...) or a Linux device path
    # such as /dev/video2.
    device: Union[int, str] = 0
    width: int = 1280
    height: int = 720
    fps: int = 30

    # On Linux, forcing CAP_V4L2 often avoids backend auto-selection issues.
    backend: int = cv2.CAP_V4L2

    _cap: Optional[cv2.VideoCapture] = None
    _frame_id: int = 0

    def start(self) -> None:
        cap: Optional[cv2.VideoCapture] = None
        tried: list[tuple[Union[int, str], Optional[int], bool]] = []
        allow_cap_any = os.environ.get("BBCOACH_V4L2_CAP_ANY_FALLBACK", "").strip() == "1"

        def _try(dev: Union[int, str], backend: Optional[int] = None) -> bool:
            nonlocal cap
            if backend is None:
                cap = cv2.VideoCapture(dev)
            else:
                cap = cv2.VideoCapture(dev, backend)
            ok = cap.isOpened()
            tried.append((dev, backend, ok))
            if not ok:
                cap.release()
                cap = None
            return ok

        # Prefer explicit V4L2 backend only (stable on Linux). CAP_ANY fallback
        # is opt-in via BBCOACH_V4L2_CAP_ANY_FALLBACK=1.
        if isinstance(self.device, str) and self.device.startswith("/dev/video"):
            suffix = self.device.replace("/dev/video", "")
            if suffix.isdigit():
                idx = int(suffix)
                if not _try(idx, self.backend):
                    _try(self.device, self.backend)
                    if cap is None and allow_cap_any:
                        if not _try(idx):
                            _try(self.device)
            else:
                if not _try(self.device, self.backend) and allow_cap_any:
                    _try(self.device)
        else:
            if not _try(self.device, self.backend) and allow_cap_any:
                _try(self.device)

            if cap is None and isinstance(self.device, int):
                dev_path = f"/dev/video{self.device}"
                if not _try(dev_path, self.backend) and allow_cap_any:
                    _try(dev_path)

        if cap is None:
            raise RuntimeError(
                "OpenCV could not open the camera. Try a different --cam-index value (e.g. 1, 2, 3) "
                "or a device path like /dev/video2. Also check that no other app is using the camera."
            )

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.height))
        cap.set(cv2.CAP_PROP_FPS, float(self.fps))
        self._cap = cap

    def read(self) -> SourceFrame:
        ts = time.time()
        if self._cap is None:
            return self._empty_frame(ts, width=self.width, height=self.height)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return self._empty_frame(ts, width=self.width, height=self.height)
        self._frame_id += 1
        h, w = frame.shape[:2]
        return {
            "rgb": frame,
            "depth": None,
            "ir": None,
            "timestamp": ts,
            "meta": {
                "kind": "v4l2",
                "frame_id": self._frame_id,
                "intrinsics": None,
                "width": int(w),
                "height": int(h),
            },
        }

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _empty_frame(self, ts: float, width: int, height: int) -> SourceFrame:
        return {
            "rgb": None,
            "depth": None,
            "ir": None,
            "timestamp": ts,
            "meta": {
                "kind": "v4l2",
                "frame_id": self._frame_id,
                "intrinsics": None,
                "width": int(width),
                "height": int(height),
            },
        }

    # Backwards-compat alias
    def open(self) -> None:  # pragma: no cover - legacy alias
        self.start()

    def close(self) -> None:  # pragma: no cover - legacy alias
        self.stop()


@dataclass
class Kinect2Source:
    width: int = 1280
    height: int = 720
    fps: int = 30
    depth_align: bool = True
    depth_scale: str = "meters"

    _capture: Optional[capture_kinect.Kinect2Capture] = None
    _frame_id: int = 0

    def start(self) -> None:
        if self._capture is not None:
            return
        cfg = _kinect_runtime_config(self.depth_align)
        self._capture = capture_kinect.Kinect2Capture(
            align=cfg["align"],
            enable_ir=cfg["enable_ir"],
            enable_depth=cfg["enable_depth"],
        )
        self._capture.start()

    def read(self) -> SourceFrame:
        ts = time.time()
        if self._capture is None:
            return self._empty_frame(ts)
        rgb_aligned, depth, ir, aligned = self._capture.read()
        rgb_raw = self._capture.last_color_raw
        rgb = rgb_raw if rgb_raw is not None else rgb_aligned
        if rgb is None and depth is None and ir is None:
            return self._empty_frame(ts)

        if depth is not None:
            if depth.dtype != np.float32:
                depth = depth.astype(np.float32)
        if rgb is not None and rgb.dtype != np.uint8:
            rgb = rgb.astype(np.uint8)
        if rgb_aligned is not None and rgb_aligned.dtype != np.uint8:
            rgb_aligned = rgb_aligned.astype(np.uint8)
        if rgb_raw is not None and rgb_raw.dtype != np.uint8:
            rgb_raw = rgb_raw.astype(np.uint8)
        if ir is not None and ir.dtype != np.uint16:
            ir = ir.astype(np.uint16)

        self._frame_id += 1
        rgb_h, rgb_w = rgb.shape[:2] if rgb is not None else (None, None)
        depth_h, depth_w = depth.shape[:2] if depth is not None else (None, None)
        width = rgb_w or depth_w or self.width
        height = rgb_h or depth_h or self.height

        intrinsics = self._capture.intrinsics
        if not aligned:
            if rgb is not None and self._capture.color_intrinsics:
                intrinsics = self._capture.color_intrinsics
            elif depth is not None and self._capture.depth_intrinsics:
                intrinsics = self._capture.depth_intrinsics

        depth_aligned_for_rgb = bool(aligned and rgb is not None and depth is not None and rgb.shape[:2] == depth.shape[:2])
        meta: Dict[str, Any] = {
            "kind": "kinect2",
            "frame_id": self._frame_id,
            "intrinsics": intrinsics,
            "width": int(width),
            "height": int(height),
            "depth_units": "meters",
            "depth_aligned": depth_aligned_for_rgb,
        }
        if aligned and not depth_aligned_for_rgb:
            meta["depth_aligned_to_rgb_aligned"] = True
        if self._capture.last_error:
            meta["kinect_error"] = self._capture.last_error
        if self._capture.pipeline_name:
            meta["kinect_pipeline"] = self._capture.pipeline_name

        return {
            "rgb": rgb,
            "rgb_aligned": rgb_aligned,
            "rgb_raw": rgb_raw,
            "depth": depth,
            "ir": ir,
            "timestamp": ts,
            "meta": meta,
        }

    def stop(self) -> None:
        if self._capture is not None:
            self._capture.stop()
            self._capture = None

    def _empty_frame(self, ts: float) -> SourceFrame:
        return {
            "rgb": None,
            "depth": None,
            "ir": None,
            "timestamp": ts,
            "meta": {
                "kind": "kinect2",
                "frame_id": self._frame_id,
                "intrinsics": self._capture.intrinsics if self._capture else None,
                "width": int(self.width),
                "height": int(self.height),
            },
        }


def _kinect_worker_main(conn: Any, cfg: dict[str, bool]) -> None:
    cap: Optional[capture_kinect.Kinect2Capture] = None
    try:
        cap = capture_kinect.Kinect2Capture(
            align=bool(cfg.get("align", True)),
            enable_ir=bool(cfg.get("enable_ir", False)),
            enable_depth=bool(cfg.get("enable_depth", False)),
        )
        cap.start()
        conn.send(
            {
                "type": "started",
                "pipeline": cap.pipeline_name,
                "enable_depth": bool(cfg.get("enable_depth", False)),
                "enable_ir": bool(cfg.get("enable_ir", False)),
            }
        )

        while True:
            try:
                cmd = conn.recv()
            except EOFError:
                break
            if not isinstance(cmd, dict):
                continue
            action = str(cmd.get("cmd", "")).strip().lower()
            if action == "stop":
                break
            if action != "read":
                conn.send({"type": "error", "error": f"Unknown worker command: {action}"})
                continue

            ts = float(time.time())
            rgb_aligned, depth, ir, aligned = cap.read()
            rgb_raw = cap.last_color_raw
            rgb = rgb_raw if rgb_raw is not None else rgb_aligned

            payload: dict[str, Any] = {
                "type": "frame",
                "timestamp": ts,
                "aligned": bool(aligned),
                "kinect_error": cap.last_error,
                "pipeline": cap.pipeline_name,
                "rgb_jpg": None,
                "depth_png": None,
                "ir_png": None,
            }

            if rgb is not None:
                if rgb.dtype != np.uint8:
                    rgb = rgb.astype(np.uint8)
                ok, enc = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if ok:
                    payload["rgb_jpg"] = enc.tobytes()

            if depth is not None:
                if depth.dtype != np.float32:
                    depth = depth.astype(np.float32)
                depth_mm = np.clip(depth * 1000.0, 0.0, 65535.0).astype(np.uint16)
                ok, enc = cv2.imencode(".png", depth_mm)
                if ok:
                    payload["depth_png"] = enc.tobytes()

            if ir is not None:
                if ir.dtype != np.uint16:
                    ir = ir.astype(np.uint16)
                ok, enc = cv2.imencode(".png", ir)
                if ok:
                    payload["ir_png"] = enc.tobytes()

            conn.send(payload)
    except Exception as exc:
        err = f"{exc.__class__.__name__}: {exc}"
        try:
            conn.send(
                {
                    "type": "error",
                    "error": err,
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
    finally:
        if cap is not None:
            try:
                cap.stop()
            except Exception:
                pass
        try:
            conn.close()
        except Exception:
            pass


@dataclass
class Kinect2IsolatedSource:
    width: int = 1280
    height: int = 720
    fps: int = 30
    depth_align: bool = True
    depth_scale: str = "meters"

    _process: Optional[Any] = None
    _parent_conn: Optional[Any] = None
    _frame_id: int = 0
    _pipeline_name: Optional[str] = None
    _startup_timeout_s: float = 3.0
    _read_timeout_s: float = 0.20
    _recovery_cooldown_s: float = 1.0
    _request_in_flight: bool = False
    _last_request_ts: float = 0.0
    _last_recovery_attempt_ts: float = 0.0
    _consecutive_failures: int = 0
    _last_rgb: Optional[np.ndarray] = None
    _last_depth: Optional[np.ndarray] = None
    _last_ir: Optional[np.ndarray] = None

    def start(self) -> None:
        if self._process is not None and bool(self._process.is_alive()):
            return

        cfg = _kinect_runtime_config(self.depth_align)
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        proc = ctx.Process(
            target=_kinect_worker_main,
            args=(child_conn, cfg),
            daemon=True,
            name="bbcoach_kinect_worker",
        )
        proc.start()
        child_conn.close()

        self._process = proc
        self._parent_conn = parent_conn
        self._pipeline_name = None
        self._request_in_flight = False
        self._last_request_ts = 0.0
        self._consecutive_failures = 0

        deadline = time.time() + float(self._startup_timeout_s)
        while time.time() < deadline:
            if parent_conn.poll(0.1):
                msg = parent_conn.recv()
                if isinstance(msg, dict) and msg.get("type") == "started":
                    self._pipeline_name = str(msg.get("pipeline") or "") or None
                    return
                self.stop()
                err = str(getattr(msg, "get", lambda _k, _d=None: None)("error", "Kinect worker failed to start."))
                tb = str(getattr(msg, "get", lambda _k, _d=None: None)("traceback", "") or "").strip()
                if tb:
                    raise RuntimeError(f"{err}\n{tb}")
                raise RuntimeError(err)
            if not proc.is_alive():
                self.stop()
                raise RuntimeError("Kinect worker process exited during startup.")

        self.stop()
        raise RuntimeError("Timed out waiting for Kinect worker startup.")

    def read(self) -> SourceFrame:
        ts = float(time.time())
        proc = self._process
        conn = self._parent_conn
        if proc is None or conn is None:
            if self._try_recover(ts, "Kinect worker not started."):
                return self._empty_frame(ts, note="Recovering Kinect worker...")
            return self._fallback_or_empty(ts, note="Kinect worker not started.")
        if not proc.is_alive():
            if self._try_recover(ts, "Kinect worker process exited."):
                return self._empty_frame(ts, note="Recovering Kinect worker...")
            raise RuntimeError("Kinect worker process exited.")

        if not self._request_in_flight:
            try:
                conn.send({"cmd": "read"})
                self._request_in_flight = True
                self._last_request_ts = ts
            except (BrokenPipeError, EOFError, OSError) as exc:
                self._request_in_flight = False
                if self._try_recover(ts, "Kinect worker channel closed."):
                    return self._empty_frame(ts, note="Recovering Kinect worker...")
                raise RuntimeError("Kinect worker channel closed.") from exc

        if not conn.poll(0.0):
            if (ts - self._last_request_ts) > self._read_timeout_s:
                self._request_in_flight = False
                self._consecutive_failures += 1
                if self._consecutive_failures >= 3 and self._try_recover(ts, "Kinect worker read timeout."):
                    return self._empty_frame(ts, note="Recovering Kinect worker...")
                return self._fallback_or_empty(ts, note="Kinect worker read timeout.")
            return self._fallback_or_empty(ts, note="Waiting for Kinect frame...")

        try:
            msg = conn.recv()
        except (EOFError, ConnectionResetError, BrokenPipeError, OSError) as exc:
            self._request_in_flight = False
            self._consecutive_failures += 1
            if self._try_recover(ts, "Kinect worker connection reset while reading frame."):
                return self._empty_frame(ts, note="Recovering Kinect worker...")
            raise RuntimeError("Kinect worker connection reset while reading frame.") from exc
        self._request_in_flight = False

        if not isinstance(msg, dict):
            self._consecutive_failures += 1
            raise RuntimeError("Kinect worker sent invalid frame payload.")

        msg_type = str(msg.get("type", "")).strip().lower()
        if msg_type == "error":
            err = str(msg.get("error") or "Kinect worker error.")
            tb = str(msg.get("traceback") or "").strip()
            self._consecutive_failures += 1
            if self._try_recover(ts, err):
                return self._empty_frame(ts, note="Recovering Kinect worker...")
            if tb:
                raise RuntimeError(f"{err}\n{tb}")
            raise RuntimeError(err)
        if msg_type != "frame":
            self._consecutive_failures += 1
            raise RuntimeError(f"Unexpected Kinect worker payload type: {msg_type}")

        rgb = self._decode_color(msg.get("rgb_jpg"))
        depth = self._decode_depth(msg.get("depth_png"))
        ir = self._decode_ir(msg.get("ir_png"))
        aligned = bool(msg.get("aligned", False))
        worker_ts = float(msg.get("timestamp") or ts)
        kinect_error = str(msg.get("kinect_error") or "").strip()
        pipeline = str(msg.get("pipeline") or "").strip()
        if pipeline:
            self._pipeline_name = pipeline

        if rgb is None and depth is None and ir is None:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 6 and self._try_recover(ts, kinect_error or "Empty frame burst."):
                return self._empty_frame(worker_ts, note="Recovering Kinect worker...")
            return self._fallback_or_empty(worker_ts, note=(kinect_error or "Empty Kinect frame from worker."))

        self._consecutive_failures = 0
        if rgb is not None:
            self._last_rgb = rgb.copy()
        if depth is not None:
            self._last_depth = depth.copy()
        if ir is not None:
            self._last_ir = ir.copy()

        self._frame_id += 1
        rgb_h, rgb_w = rgb.shape[:2] if rgb is not None else (None, None)
        depth_h, depth_w = depth.shape[:2] if depth is not None else (None, None)
        width = rgb_w or depth_w or self.width
        height = rgb_h or depth_h or self.height

        depth_aligned_for_rgb = bool(aligned and rgb is not None and depth is not None and rgb.shape[:2] == depth.shape[:2])
        meta: Dict[str, Any] = {
            "kind": "kinect2",
            "frame_id": self._frame_id,
            "intrinsics": None,
            "width": int(width),
            "height": int(height),
            "depth_units": "meters",
            "depth_aligned": depth_aligned_for_rgb,
        }
        if aligned and not depth_aligned_for_rgb:
            meta["depth_aligned_to_rgb_aligned"] = True
        if kinect_error:
            meta["kinect_error"] = kinect_error
        if self._pipeline_name:
            meta["kinect_pipeline"] = self._pipeline_name
        meta["kinect_isolated"] = True

        return {
            "rgb": rgb,
            "rgb_aligned": None,
            "rgb_raw": rgb,
            "depth": depth,
            "ir": ir,
            "timestamp": worker_ts,
            "meta": meta,
        }

    def stop(self) -> None:
        conn = self._parent_conn
        proc = self._process
        if conn is not None:
            try:
                conn.send({"cmd": "stop"})
            except Exception:
                pass
        if proc is not None:
            try:
                proc.join(timeout=1.0)
            except Exception:
                pass
            if proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.join(timeout=0.5)
                except Exception:
                    pass
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        self._process = None
        self._parent_conn = None
        self._request_in_flight = False

    def _empty_frame(self, ts: float, note: Optional[str] = None) -> SourceFrame:
        meta: Dict[str, Any] = {
            "kind": "kinect2",
            "frame_id": int(self._frame_id),
            "intrinsics": None,
            "width": int(self.width),
            "height": int(self.height),
            "kinect_isolated": True,
        }
        if self._pipeline_name:
            meta["kinect_pipeline"] = self._pipeline_name
        if note:
            meta["kinect_error"] = note
        return {
            "rgb": None,
            "depth": None,
            "ir": None,
            "timestamp": float(ts),
            "meta": meta,
        }

    def _fallback_or_empty(self, ts: float, note: Optional[str] = None) -> SourceFrame:
        if self._last_rgb is None and self._last_depth is None and self._last_ir is None:
            return self._empty_frame(ts, note=note)
        meta: Dict[str, Any] = {
            "kind": "kinect2",
            "frame_id": int(self._frame_id),
            "intrinsics": None,
            "width": int(self.width),
            "height": int(self.height),
            "kinect_isolated": True,
            "kinect_stale_frame": True,
        }
        if self._pipeline_name:
            meta["kinect_pipeline"] = self._pipeline_name
        if note:
            meta["kinect_error"] = note
        return {
            "rgb": self._last_rgb,
            "rgb_aligned": None,
            "rgb_raw": self._last_rgb,
            "depth": self._last_depth,
            "ir": self._last_ir,
            "timestamp": float(ts),
            "meta": meta,
        }

    def _try_recover(self, ts: float, reason: str) -> bool:
        if (ts - self._last_recovery_attempt_ts) < self._recovery_cooldown_s:
            return False
        self._last_recovery_attempt_ts = ts
        try:
            self.stop()
        except Exception:
            pass
        try:
            self.start()
            return True
        except Exception:
            return False

    @staticmethod
    def _decode_color(payload: Any) -> Optional[np.ndarray]:
        if payload in (None, b""):
            return None
        try:
            arr = np.frombuffer(payload, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                return None
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            return img
        except Exception:
            return None

    @staticmethod
    def _decode_depth(payload: Any) -> Optional[np.ndarray]:
        if payload in (None, b""):
            return None
        try:
            arr = np.frombuffer(payload, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if img.dtype == np.uint16:
                return img.astype(np.float32) * 0.001
            if img.dtype != np.float32:
                return img.astype(np.float32)
            return img
        except Exception:
            return None

    @staticmethod
    def _decode_ir(payload: Any) -> Optional[np.ndarray]:
        if payload in (None, b""):
            return None
        try:
            arr = np.frombuffer(payload, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if img.dtype != np.uint16:
                img = img.astype(np.uint16)
            return img
        except Exception:
            return None

    # Backwards-compat aliases
    def open(self) -> None:  # pragma: no cover - legacy alias
        self.start()

    def close(self) -> None:  # pragma: no cover - legacy alias
        self.stop()


# Backwards-compat alias for older code paths.
OpenCVCameraSource = V4L2Source
