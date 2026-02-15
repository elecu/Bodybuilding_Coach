from __future__ import annotations

from dataclasses import dataclass
import os
import time
from typing import Optional, Tuple

import numpy as np


KINECT_BACKEND_ERROR = (
    "Kinect v2 backend not available. Install/build libfreenect2 + python bindings. See docs/SOURCES.md"
)


@dataclass
class DepthSource:
    available: bool = False
    error: Optional[str] = "Depth source unavailable"

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        return False, None


def get_depth_source() -> DepthSource:
    return DepthSource()


def _load_backend():
    try:
        import pylibfreenect2 as pfn  # type: ignore
        return pfn
    except Exception:
        try:
            import freenect2 as pfn  # type: ignore
            return pfn
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(KINECT_BACKEND_ERROR) from exc


def _select_pipeline(pfn):
    # Stability-first default:
    # OpenGL/OpenCL pipelines can segfault on some driver stacks. Prefer CPU
    # unless explicitly overridden.
    preferred = str(os.environ.get("BBCOACH_KINECT_PIPELINE", "cpu")).strip().lower()
    orders = {
        "cpu": ("CpuPacketPipeline", "OpenCLPacketPipeline", "OpenGLPacketPipeline"),
        "opencl": ("OpenCLPacketPipeline", "CpuPacketPipeline", "OpenGLPacketPipeline"),
        "opengl": ("OpenGLPacketPipeline", "OpenCLPacketPipeline", "CpuPacketPipeline"),
        # Legacy behavior used OpenGL first and tends to be the most compatible
        # on systems where Kinect v2 was previously working in this project.
        "auto": ("OpenGLPacketPipeline", "OpenCLPacketPipeline", "CpuPacketPipeline"),
    }
    names = orders.get(preferred, orders["cpu"])
    for name in names:
        if hasattr(pfn, name):
            try:
                return getattr(pfn, name)()
            except Exception:
                continue
    return None


def _camera_params_to_intrinsics(params) -> Optional[dict]:
    if params is None:
        return None
    data = {
        "fx": float(getattr(params, "fx", 0.0)),
        "fy": float(getattr(params, "fy", 0.0)),
        "cx": float(getattr(params, "cx", 0.0)),
        "cy": float(getattr(params, "cy", 0.0)),
    }
    for key in ("k1", "k2", "k3", "p1", "p2", "k4", "k5", "k6"):
        if hasattr(params, key):
            data[key] = float(getattr(params, key))
    return data


def _to_bgr(color: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if color is None:
        return None
    if color.ndim == 3 and color.shape[2] >= 3:
        if color.shape[2] == 4:
            color = color[:, :, :3]
    elif color.ndim == 2:
        color = np.repeat(color[:, :, None], 3, axis=2)
    if color.dtype != np.uint8:
        color = color.astype(np.uint8)
    return color


def _to_depth(depth: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if depth is None:
        return None
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)
    return depth


def _to_ir(ir: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if ir is None:
        return None
    if ir.dtype != np.uint16:
        ir = ir.astype(np.uint16)
    return ir


def _frame_asarray(frame, dtype: np.dtype) -> Optional[np.ndarray]:
    if frame is None:
        return None
    try:
        return frame.asarray(dtype)
    except Exception:
        try:
            arr = frame.asarray()
            return arr.astype(dtype)
        except Exception:
            return None


def _depth_to_m(depth: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if depth is None:
        return None
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)
    if not np.isfinite(depth).all():
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = float(np.max(depth)) if depth.size else 0.0
    # Heuristic: freenect2 depth is typically in mm (1000-4000).
    if max_val > 20.0:
        depth = depth * 0.001
    return depth


def _normalize_uint8_bgr(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    if arr.ndim == 2:
        return np.repeat(arr[:, :, None], 3, axis=2)
    if arr.ndim == 3:
        if arr.shape[2] >= 3:
            return arr[:, :, :3]
        if arr.shape[2] == 1:
            return np.repeat(arr, 3, axis=2)
    return None


def _color_from_frame(frame) -> Optional[np.ndarray]:
    arr_u8 = _frame_asarray(frame, np.uint8)
    if arr_u8 is not None:
        arr_u8 = _normalize_uint8_bgr(arr_u8)
        if arr_u8 is not None:
            if float(np.max(arr_u8)) > 8.0:
                return arr_u8
    arr_f = _frame_asarray(frame, np.float32)
    if arr_f is None:
        return arr_u8
    if not np.isfinite(arr_f).all():
        arr_f = np.nan_to_num(arr_f, nan=0.0, posinf=0.0, neginf=0.0)
    max_val = float(np.max(arr_f)) if arr_f.size else 0.0
    if max_val <= 1.5:
        arr_f = arr_f * 255.0
    arr_f = np.clip(arr_f, 0.0, 255.0).astype(np.uint8)
    arr_f = _normalize_uint8_bgr(arr_f)
    if arr_f is not None:
        return arr_f
    return arr_u8


class Kinect2Capture:
    def __init__(self, align: bool = True, enable_ir: bool = True, enable_depth: bool = True) -> None:
        self.align = align
        self.enable_ir = enable_ir
        self.enable_depth = bool(enable_depth)
        self._pfn = None
        self._fn = None
        self._device = None
        self._listener = None
        self._registration = None
        self._undistorted = None
        self._registered = None
        self._depth_intrinsics: Optional[dict] = None
        self._color_intrinsics: Optional[dict] = None
        self._last_rgb: Optional[np.ndarray] = None
        self._last_color_raw: Optional[np.ndarray] = None
        self._last_error: Optional[str] = None
        self._error_times: list[float] = []
        self._pipeline_name: Optional[str] = None

    def start(self) -> None:
        if self._device is not None:
            return
        pfn = _load_backend()
        self._pfn = pfn

        fn = pfn.Freenect2()
        if fn.enumerateDevices() <= 0:
            raise RuntimeError("No Kinect v2 device detected.")
        serial = fn.getDeviceSerialNumber(0)
        pipeline = _select_pipeline(pfn)
        self._pipeline_name = type(pipeline).__name__ if pipeline is not None else None
        device = None
        listener = None
        try:
            if pipeline is None:
                device = fn.openDevice(serial)
            else:
                device = fn.openDevice(serial, pipeline=pipeline)
            if device is None:
                raise RuntimeError("Failed to open Kinect v2 device (device is busy).")

            frame_types = pfn.FrameType.Color
            if self.enable_depth and hasattr(pfn.FrameType, "Depth"):
                frame_types |= pfn.FrameType.Depth
            if self.enable_depth and self.enable_ir and hasattr(pfn.FrameType, "Ir"):
                frame_types |= pfn.FrameType.Ir

            listener = pfn.SyncMultiFrameListener(frame_types)
            device.setColorFrameListener(listener)
            if self.enable_depth:
                device.setIrAndDepthFrameListener(listener)
            device.start()
        except Exception as exc:
            try:
                if device is not None:
                    device.stop()
            except Exception:
                pass
            try:
                if device is not None:
                    device.close()
            except Exception:
                pass
            raise RuntimeError(
                "Failed to open Kinect v2 device. It may be busy (LIBUSB_ERROR_BUSY). "
                "Close other Kinect apps, replug the sensor, and try again."
            ) from exc

        self._fn = fn
        self._device = device
        self._listener = listener
        self._depth_intrinsics = _camera_params_to_intrinsics(device.getIrCameraParams())
        self._color_intrinsics = _camera_params_to_intrinsics(device.getColorCameraParams())

        if self.align:
            try:
                self._registration = pfn.Registration(device.getIrCameraParams(), device.getColorCameraParams())
                # Kinect v2 depth resolution is 512x424.
                self._undistorted = pfn.Frame(512, 424, 4)
                self._registered = pfn.Frame(512, 424, 4)
            except Exception:
                self._registration = None
                self._undistorted = None
                self._registered = None

    def stop(self) -> None:
        if self._device is not None:
            try:
                self._device.stop()
            except Exception:
                pass
            try:
                self._device.close()
            except Exception:
                pass
        self._device = None
        self._listener = None
        self._fn = None
        self._registration = None
        self._undistorted = None
        self._registered = None

    def _record_error(self, msg: str) -> None:
        self._last_error = msg
        now = time.time()
        self._error_times = [t for t in self._error_times if now - t < 5.0]
        self._error_times.append(now)
        if len(self._error_times) >= 5:
            try:
                self.stop()
                self.start()
                self._error_times = []
            except Exception as exc:
                self._last_error = str(exc)

    @property
    def intrinsics(self) -> Optional[dict]:
        if self.align:
            return self._depth_intrinsics
        return self._color_intrinsics

    @property
    def depth_intrinsics(self) -> Optional[dict]:
        return self._depth_intrinsics

    @property
    def color_intrinsics(self) -> Optional[dict]:
        return self._color_intrinsics

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], bool]:
        if self._listener is None:
            raise RuntimeError("Kinect v2 capture not started.")
        frames = None
        try:
            frames = self._listener.waitForNewFrame()
        except Exception as exc:
            self._record_error(str(exc))
            return self._last_rgb, None, None, False
        if frames is None:
            self._record_error("Empty Kinect frame")
            return self._last_rgb, None, None, False
        aligned = False
        color_arr: Optional[np.ndarray] = None
        depth_arr: Optional[np.ndarray] = None
        ir_arr: Optional[np.ndarray] = None
        try:
            color = None
            depth = None
            ir = None
            if frames is not None:
                try:
                    color = frames["color"]
                except Exception:
                    color = None
                try:
                    depth = frames["depth"]
                except Exception:
                    depth = None
                try:
                    ir = frames["ir"]
                except Exception:
                    ir = None

            can_align = (
                self.align
                and self._registration is not None
                and self._undistorted is not None
                and self._registered is not None
                and color is not None
                and depth is not None
            )
            if can_align:
                self._registration.apply(color, depth, self._undistorted, self._registered)
                reg = _frame_asarray(self._registered, np.uint8)
                if reg is not None:
                    reg = np.ascontiguousarray(reg)
                reg_bgr = _normalize_uint8_bgr(reg) if reg is not None else None
                undist = _frame_asarray(self._undistorted, np.float32)
                if undist is not None:
                    depth_arr = _depth_to_m(undist)
                    if depth_arr is not None:
                        depth_arr = np.ascontiguousarray(depth_arr, dtype=np.float32)
                if reg_bgr is not None and depth_arr is not None:
                    # If the registered color is basically empty, fall back to raw color.
                    if float(np.max(reg_bgr)) > 8.0:
                        color_arr = np.copy(reg_bgr)
                        aligned = True

            # Always try to keep a raw color frame for display.
            if color is not None:
                raw_col = _color_from_frame(color)
                if raw_col is not None:
                    self._last_color_raw = raw_col

            if not aligned:
                if self._last_color_raw is not None:
                    color_arr = np.copy(self._last_color_raw)
                if depth is not None:
                    dep = _frame_asarray(depth, np.float32)
                    if dep is not None:
                        depth_arr = _depth_to_m(dep)
                        if depth_arr is not None:
                            depth_arr = np.ascontiguousarray(depth_arr, dtype=np.float32)
            if self.enable_ir and ir is not None:
                irr = _frame_asarray(ir, np.uint16)
                if irr is not None:
                    ir_arr = np.ascontiguousarray(irr, dtype=np.uint16)
        finally:
            if frames is not None:
                self._listener.release(frames)

        rgb = _to_bgr(color_arr)
        if rgb is not None:
            self._last_rgb = rgb
            self._last_error = None
        elif self._last_rgb is not None:
            rgb = self._last_rgb
        if rgb is None and depth_arr is None and ir_arr is None and self._last_error is None:
            self._record_error("Empty Kinect frame")
        return rgb, _to_depth(depth_arr), _to_ir(ir_arr), aligned

    @property
    def last_color_raw(self) -> Optional[np.ndarray]:
        return self._last_color_raw

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    @property
    def pipeline_name(self) -> Optional[str]:
        return self._pipeline_name
