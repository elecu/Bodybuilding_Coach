from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from bbcoach.device_manager import DeviceManager, DeviceState, InputSource


class _DummySource:
    def __init__(self, frames: list[dict] | None = None) -> None:
        self.frames = list(frames or [])
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def read(self) -> dict:
        if self.frames:
            return self.frames.pop(0)
        return {"rgb": None, "timestamp": None}


class DeviceManagerTests(unittest.TestCase):
    def test_no_auto_connect_on_startup(self) -> None:
        calls = {"open": 0}

        def source_factory(**_kwargs):
            calls["open"] += 1
            return _DummySource()

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeviceManager(
                source_factory=source_factory,
                config_path=Path(tmpdir) / "config.json",
                log_path=Path(tmpdir) / "device.log",
            )
            self.assertEqual(calls["open"], 0)
            self.assertEqual(manager.snapshot().state, DeviceState.DISCONNECTED)

    def test_connect_failure_moves_to_error_with_actionable_message(self) -> None:
        def source_factory(**_kwargs):
            raise RuntimeError("camera not available")

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeviceManager(
                source_factory=source_factory,
                webcam_nodes_provider=lambda: [],
                config_path=Path(tmpdir) / "config.json",
                log_path=Path(tmpdir) / "device.log",
            )
            manager.select_source(InputSource.WEBCAM)
            snap = manager.connect()
            self.assertEqual(snap.state, DeviceState.ERROR)
            self.assertIn("No Webcam detected", snap.status_message)

    def test_switching_source_disconnects_previous_stream(self) -> None:
        webcam_source = _DummySource(frames=[{"rgb": None}])
        kinect_source = _DummySource()

        def source_factory(**kwargs):
            if kwargs.get("kind") == "v4l2":
                return webcam_source
            return kinect_source

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeviceManager(
                source_factory=source_factory,
                webcam_nodes_provider=lambda: [Path("/dev/video0")],
                config_path=Path(tmpdir) / "config.json",
                log_path=Path(tmpdir) / "device.log",
            )
            manager.select_source(InputSource.WEBCAM)
            self.assertEqual(manager.connect().state, DeviceState.STREAMING)
            manager.select_source(InputSource.KINECT)
            self.assertTrue(webcam_source.stopped)
            self.assertEqual(manager.snapshot().state, DeviceState.DISCONNECTED)

    def test_retry_recovers_after_initial_connect_failure(self) -> None:
        calls = {"n": 0}

        def source_factory(**_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("first attempt fails")
            return _DummySource()

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeviceManager(
                source_factory=source_factory,
                webcam_nodes_provider=lambda: [Path("/dev/video0")],
                config_path=Path(tmpdir) / "config.json",
                log_path=Path(tmpdir) / "device.log",
            )
            manager.select_source(InputSource.WEBCAM)
            self.assertEqual(manager.connect().state, DeviceState.ERROR)
            self.assertEqual(manager.retry().state, DeviceState.STREAMING)

    def test_device_lost_transition_on_repeated_empty_frames(self) -> None:
        source = _DummySource(frames=[{"rgb": None}, {"rgb": None}, {"rgb": None}])

        def source_factory(**_kwargs):
            return source

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = DeviceManager(
                source_factory=source_factory,
                webcam_nodes_provider=lambda: [Path("/dev/video0")],
                config_path=Path(tmpdir) / "config.json",
                log_path=Path(tmpdir) / "device.log",
            )
            manager.select_source(InputSource.WEBCAM)
            manager.connect()
            manager._lost_frame_limit = 2
            manager.read_stream_frame()
            manager.read_stream_frame()
            self.assertEqual(manager.snapshot().state, DeviceState.ERROR)
            self.assertIn("Device disconnected", manager.snapshot().status_message)

    def test_selected_source_persists_to_local_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            log_path = Path(tmpdir) / "device.log"
            manager = DeviceManager(config_path=config_path, log_path=log_path)
            manager.select_source(InputSource.KINECT)

            manager2 = DeviceManager(config_path=config_path, log_path=log_path)
            self.assertEqual(manager2.snapshot().selected_source, InputSource.KINECT)


if __name__ == "__main__":
    unittest.main()
