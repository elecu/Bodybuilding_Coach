from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, Optional, List


_DEFAULT_WAV_PATH = "/tmp/bbcoach_tts.wav"


class TTSSpeaker:
    def __init__(self, backend: str = "piper_bin", debounce_seconds: float = 6.0) -> None:
        self.backend = backend
        self.debounce_seconds = debounce_seconds
        self._backend: Optional[str] = None
        self._piper_bin: Optional[Path] = None
        self._model_onnx: Optional[Path] = None
        self._player_cmd: Optional[List[str]] = None
        self._warning: Optional[str] = None
        self._backend_label: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._q: "queue.Queue[Optional[str]]" = queue.Queue()
        self._recent: Dict[str, float] = {}
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._backend = self._resolve_backend(self.backend)
        if self._backend is None:
            raise RuntimeError("No TTS backend available. Ensure vendored Alan assets are present.")
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            try:
                self._q.put_nowait(None)
            except queue.Full:
                pass
            self._thread.join(timeout=1.0)
            self._thread = None

    def say(self, text: str) -> None:
        line = self._normalize_text(text)
        if not line:
            return
        now = time.time()
        with self._lock:
            last = self._recent.get(line, 0.0)
            if now - last < self.debounce_seconds:
                return
            self._recent[line] = now
            if len(self._recent) > 64:
                cutoff = now - (self.debounce_seconds * 2.0)
                self._recent = {k: v for k, v in self._recent.items() if v >= cutoff}
        try:
            self._q.put_nowait(line)
        except queue.Full:
            return

    def clear_pending(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    def say_priority(self, text: str, clear_pending: bool = False) -> None:
        line = self._normalize_text(text)
        if not line:
            return
        if clear_pending:
            self.clear_pending()
        try:
            self._q.put_nowait(line)
        except queue.Full:
            return

    def say_blocking(self, text: str) -> None:
        """Speak immediately on the current thread (used for tight countdown sync)."""
        line = self._normalize_text(text)
        if not line:
            return
        # Bypass debounce for blocking calls.
        self._speak(line)

    def pop_warning(self) -> Optional[str]:
        with self._lock:
            msg = self._warning
            self._warning = None
        return msg

    def backend_label(self) -> str:
        return self._backend_label or "unknown"

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text:
            return ""
        return " ".join(text.strip().split())

    def _run(self) -> None:
        while self._running.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if not item:
                continue
            self._speak(item)

    def _speak(self, text: str) -> None:
        if self._backend == "piper_bin":
            self._speak_piper(text)
            return

    def _speak_piper(self, text: str) -> None:
        if not self._piper_bin or not self._model_onnx:
            return
        try:
            subprocess.run(
                [str(self._piper_bin), "-m", str(self._model_onnx), "-f", _DEFAULT_WAV_PATH],
                input=text,
                text=True,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(self._piper_bin.parent),
            )
        except Exception:
            return
        if not self._player_cmd:
            return
        try:
            subprocess.run(
                self._player_cmd + [_DEFAULT_WAV_PATH],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return

    def _resolve_backend(self, backend: str) -> Optional[str]:
        pref = (backend or "piper_bin").lower()
        if pref == "auto":
            pref = "piper_bin"
        if pref not in ("piper_bin",):
            raise RuntimeError(f"Unknown TTS backend '{backend}'. Use piper_bin.")

        if pref == "piper_bin":
            ok, warn = self._setup_piper()
            if ok:
                return "piper_bin"
            if warn:
                with self._lock:
                    self._warning = warn
            return None

        return None

    def _setup_piper(self) -> tuple[bool, Optional[str]]:
        root = Path(__file__).resolve().parents[2]
        piper_bin = root / "vendor" / "piper" / "linux_x86_64" / "piper"
        model_dir = root / "data" / "tts" / "en_GB-alan-medium"
        model_onnx = model_dir / "voice.onnx"
        model_json = model_dir / "voice.onnx.json"

        missing: list[str] = []
        if not piper_bin.exists():
            missing.append(str(piper_bin))
        if not model_onnx.exists():
            missing.append(str(model_onnx))
        if not model_json.exists():
            missing.append(str(model_json))
        if missing:
            return False, "Missing vendored Alan assets: " + ", ".join(missing)
        if not os.access(piper_bin, os.X_OK):
            return False, f"Piper not executable: {piper_bin}"

        player = shutil.which("paplay")
        if player:
            player_cmd = [player]
        else:
            aplay = shutil.which("aplay")
            player_cmd = [aplay] if aplay else None

        if not player_cmd:
            return False, "Audio player missing (paplay/aplay)."

        self._piper_bin = piper_bin
        self._model_onnx = model_onnx
        self._player_cmd = player_cmd
        self._backend_label = "piper(alan vendored)"
        return True, None
