from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import traceback
from typing import Any, Callable, Optional

import numpy as np

from .storage.session_paths import SessionPaths
from .vision.source import open_source


class DeviceState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    STREAMING = "STREAMING"
    ERROR = "ERROR"


class InputSource(str, Enum):
    NONE = "none"
    WEBCAM = "webcam"
    KINECT = "kinect"


@dataclass(frozen=True)
class DeviceSnapshot:
    state: DeviceState
    selected_source: InputSource
    status_message: str
    technical_details: str


SourceFactory = Callable[..., Any]
WebcamNodesProvider = Callable[[], list[Path]]


def _default_webcam_nodes_provider() -> list[Path]:
    return sorted(Path("/dev").glob("video*"))


class DeviceManager:
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        camera: int | str | None = 0,
        initial_source: str = InputSource.NONE.value,
        source_factory: SourceFactory = open_source,
        webcam_nodes_provider: WebcamNodesProvider = _default_webcam_nodes_provider,
        config_path: Optional[Path] = None,
        log_path: Optional[Path] = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        self.width = int(width)
        self.height = int(height)
        self.camera = camera
        self.source_factory = source_factory
        self.webcam_nodes_provider = webcam_nodes_provider
        self.config_path = config_path or (repo_root / "config" / "device_connection.json")
        sessions_root = SessionPaths.default().root
        self.log_path = log_path or (sessions_root / "logs" / "device.log")

        self._source = None
        self._lost_frame_limit = 20
        self._empty_frame_count = 0
        self._frame_count = 0

        self.state = DeviceState.DISCONNECTED
        self.selected_source = self._normalize_source(initial_source)
        self.status_message = "Offline: no live input selected."
        self.technical_details = ""

        self._load_selected_source()
        self._set_disconnected_message_for_selection()
        self._log("INFO", f"Device manager initialized with selected_source={self.selected_source.value}")

    def snapshot(self) -> DeviceSnapshot:
        return DeviceSnapshot(
            state=self.state,
            selected_source=self.selected_source,
            status_message=self.status_message,
            technical_details=self.technical_details,
        )

    def select_source(self, source: str | InputSource) -> DeviceSnapshot:
        new_source = self._normalize_source(source)
        self._log("INFO", f"event=select_source value={new_source.value}")
        if new_source != self.selected_source and self.state in (DeviceState.CONNECTING, DeviceState.STREAMING):
            self._disconnect_source()
            self.state = DeviceState.DISCONNECTED
        self.selected_source = new_source
        self._persist_selected_source()
        self.technical_details = ""
        self._set_disconnected_message_for_selection()
        return self.snapshot()

    def connect(self) -> DeviceSnapshot:
        self._log("INFO", f"event=connect source={self.selected_source.value}")
        if self.selected_source == InputSource.NONE:
            self.state = DeviceState.DISCONNECTED
            self.status_message = "Offline: no live input selected."
            self.technical_details = ""
            return self.snapshot()
        if self.state == DeviceState.STREAMING and self._source is not None:
            return self.snapshot()

        self.state = DeviceState.CONNECTING
        self.status_message = f"Connecting to {self._source_label(self.selected_source)}..."
        self.technical_details = ""

        try:
            if self.selected_source == InputSource.WEBCAM:
                source = self._open_webcam_source_with_fallback()
            else:
                source = self._open_selected_source()
                source.start()
            self._source = source
            self._empty_frame_count = 0
            self._frame_count = 0
            self.state = DeviceState.STREAMING
            self.status_message = f"Streaming from {self._source_label(self.selected_source)}."
            self.technical_details = ""
            self._log("INFO", f"state={self.state.value} source={self.selected_source.value}")
        except Exception as exc:
            details = traceback.format_exc()
            self._source = None
            self._set_connect_error(exc, details)
        return self.snapshot()

    def disconnect(self) -> DeviceSnapshot:
        self._log("INFO", f"event=disconnect source={self.selected_source.value}")
        self._disconnect_source()
        self.state = DeviceState.DISCONNECTED
        self.technical_details = ""
        self._set_disconnected_message_for_selection(disconnected=True)
        return self.snapshot()

    def retry(self) -> DeviceSnapshot:
        self._log("INFO", f"event=retry source={self.selected_source.value}")
        if self.state == DeviceState.STREAMING:
            return self.snapshot()
        return self.connect()

    def device_lost(self, reason: str, details: str = "") -> DeviceSnapshot:
        self._log("ERROR", f"event=device_lost source={self.selected_source.value} reason={reason}")
        self._disconnect_source()
        self.state = DeviceState.ERROR
        self.status_message = "Device disconnected. Reconnect and press Retry."
        merged_details = details.strip()
        if reason and reason not in merged_details:
            merged_details = f"{reason}\n{merged_details}".strip()
        self.technical_details = merged_details
        if merged_details:
            self._log("ERROR", merged_details)
        return self.snapshot()

    def read_stream_frame(self) -> tuple[Optional[np.ndarray], Optional[dict]]:
        if self.state != DeviceState.STREAMING or self._source is None:
            return None, None
        try:
            packet = self._source.read()
        except Exception:
            details = traceback.format_exc()
            self.device_lost("Read failed while streaming.", details=details)
            return None, None

        frame = packet.get("rgb")
        if self.selected_source == InputSource.KINECT:
            aligned = packet.get("rgb_aligned")
            if aligned is not None:
                frame = aligned

        if frame is None:
            self._empty_frame_count += 1
            lost_limit = self._lost_frame_limit
            if self.selected_source == InputSource.KINECT:
                # Kinect streams can have short frame gaps; avoid premature disconnect.
                lost_limit = max(lost_limit, 120)
            if self._empty_frame_count >= lost_limit:
                self.device_lost("No frames received from the active device.")
            return None, packet

        self._empty_frame_count = 0
        self._frame_count += 1
        return frame, packet

    def read_recent_logs(self, max_lines: int = 120) -> str:
        if not self.log_path.exists():
            return "No logs available."
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            return f"Unable to read logs: {exc}"
        tail = lines[-max_lines:] if max_lines > 0 else lines
        return "\n".join(tail)

    def shutdown(self) -> None:
        self._log("INFO", "event=shutdown")
        self._disconnect_source()

    def _camera_index(self) -> int | str | None:
        if self.selected_source == InputSource.KINECT:
            return None
        if isinstance(self.camera, str) and self.camera.isdigit():
            return int(self.camera)
        return self.camera if self.camera is not None else 0

    def _webcam_candidates(self) -> list[int | str]:
        candidates: list[int | str] = []
        seen: set[str] = set()

        def _add(value: int | str | None) -> None:
            if value in (None, ""):
                return
            candidate: int | str
            if isinstance(value, str) and value.isdigit():
                candidate = int(value)
            else:
                candidate = value
            key = f"{type(candidate).__name__}:{candidate}"
            if key in seen:
                return
            seen.add(key)
            candidates.append(candidate)

        _add(self._camera_index())
        if isinstance(self.camera, str) and self.camera.startswith("/dev/video"):
            suffix = self.camera.replace("/dev/video", "")
            if suffix.isdigit():
                _add(int(suffix))
        nodes = self.webcam_nodes_provider()
        for node in nodes:
            node_txt = str(node)
            _add(node_txt)
            if node_txt.startswith("/dev/video"):
                suffix = node_txt.replace("/dev/video", "")
                if suffix.isdigit():
                    _add(int(suffix))
        return candidates

    def _open_webcam_source_with_fallback(self):
        candidates = self._webcam_candidates()
        if not candidates:
            self._log("WARNING", "No webcam candidates available before connect attempt.")
            candidates = [0]
        errors: list[str] = []
        for cam_candidate in candidates:
            source = self.source_factory(
                kind="v4l2",
                cam_index=cam_candidate,
                width=self.width,
                height=self.height,
                fps=30,
                depth_align=True,
                depth_scale="meters",
            )
            try:
                source.start()
                self.camera = cam_candidate
                self._log("INFO", f"Webcam stream opened with camera={cam_candidate!r}")
                return source
            except Exception as exc:
                errors.append(f"{cam_candidate!r}: {exc}")
                self._log("WARNING", f"Webcam candidate failed camera={cam_candidate!r}: {exc}")
                try:
                    source.stop()
                except Exception:
                    pass
        tried_text = ", ".join(str(c) for c in candidates)
        raise RuntimeError(
            "OpenCV could not open the webcam from the available candidates. "
            f"Tried: {tried_text}. Select another device and retry."
        )

    def _open_selected_source(self):
        if self.selected_source == InputSource.WEBCAM:
            nodes = self.webcam_nodes_provider()
            if not nodes:
                self._log("WARNING", "No /dev/video* devices detected before webcam connect attempt.")
            return self.source_factory(
                kind="v4l2",
                cam_index=self._camera_index(),
                width=self.width,
                height=self.height,
                fps=30,
                depth_align=True,
                depth_scale="meters",
            )
        if self.selected_source == InputSource.KINECT:
            return self.source_factory(
                kind="kinect2",
                cam_index=None,
                width=self.width,
                height=self.height,
                fps=30,
                depth_align=True,
                depth_scale="meters",
            )
        raise RuntimeError("No live source selected.")

    def _disconnect_source(self) -> None:
        if self._source is not None:
            try:
                self._source.stop()
            except Exception as exc:
                self._log("WARNING", f"Source stop failed: {exc}")
            finally:
                self._source = None
        self._empty_frame_count = 0

    def _set_connect_error(self, exc: Exception, details: str) -> None:
        self.state = DeviceState.ERROR
        lower = str(exc).lower()
        if self.selected_source == InputSource.WEBCAM:
            if "permission" in lower:
                msg = "Webcam access denied. Grant camera access and press Retry."
            elif "opencv could not open" in lower or "could not open the webcam" in lower:
                msg = "Could not open the selected webcam. Use Device > Select Webcam Device and press Retry."
            else:
                msg = "No Webcam detected or webcam connection failed. Check cable/power and press Retry."
        elif self.selected_source == InputSource.KINECT:
            if "backend not available" in lower or "libfreenect2" in lower:
                msg = "Kinect backend unavailable. Install Kinect dependencies and press Retry."
            elif "worker process exited" in lower or "worker channel closed" in lower:
                msg = "Kinect driver crashed while starting stream. Try reconnecting USB/power and press Retry."
            elif "timed out waiting for kinect worker startup" in lower or "read timeout" in lower:
                msg = "Kinect did not return frames in time. Check cable/power and press Retry."
            else:
                msg = "No Kinect detected. Please connect it and press Retry."
        else:
            msg = "Connection failed. Please press Retry."

        self.status_message = msg
        self.technical_details = details.strip() or str(exc)
        self._log("ERROR", f"Connect failed for source={self.selected_source.value}: {exc}")
        if self.technical_details:
            self._log("ERROR", self.technical_details)

    def _set_disconnected_message_for_selection(self, disconnected: bool = False) -> None:
        if self.selected_source == InputSource.NONE:
            self.status_message = "Offline: no live input selected."
            return
        source_name = self._source_label(self.selected_source)
        if disconnected:
            self.status_message = f"{source_name} disconnected."
        else:
            self.status_message = f"{source_name} selected. Use Device > Power On."

    def _persist_selected_source(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"selected_source": self.selected_source.value}
        self.config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_selected_source(self) -> None:
        if not self.config_path.exists():
            self._persist_selected_source()
            return
        try:
            payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._log("WARNING", f"Invalid device config, using default: {exc}")
            self._persist_selected_source()
            return
        loaded = payload.get("selected_source")
        if loaded is None:
            self._persist_selected_source()
            return
        try:
            self.selected_source = self._normalize_source(loaded)
        except ValueError:
            self._log("WARNING", f"Unknown selected_source in config: {loaded}")
            self._persist_selected_source()

    def _normalize_source(self, source: str | InputSource) -> InputSource:
        if isinstance(source, InputSource):
            return source
        value = str(source or "").strip().lower()
        if value in ("none", "none (offline)", "offline", "off"):
            return InputSource.NONE
        if value in ("webcam", "v4l2", "camera"):
            return InputSource.WEBCAM
        if value in ("kinect", "kinect2"):
            return InputSource.KINECT
        raise ValueError(f"Unsupported source: {source}")

    def _source_label(self, source: InputSource) -> str:
        if source == InputSource.WEBCAM:
            return "Webcam"
        if source == InputSource.KINECT:
            return "Kinect"
        return "Offline"

    def _log(self, level: str, message: str) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"{timestamp} [{level}] {message}\n"
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            return
