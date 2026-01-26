from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import queue
import threading
import time
from typing import Deque, Dict, Iterable, Optional, Tuple


DEFAULT_COMMANDS: Dict[str, Tuple[str, ...]] = {
    "next_pose": (
        "next pose",
        "next",
        "siguiente pose",
        "pose siguiente",
        "siguiente",
    ),
    "prev_pose": (
        "prev pose",
        "previous pose",
        "previous",
        "anterior pose",
        "pose anterior",
        "anterior",
    ),
}


@dataclass(frozen=True)
class VoiceCommandConfig:
    model_path: str
    sample_rate: int = 16000
    debounce_seconds: float = 1.2
    commands: Dict[str, Tuple[str, ...]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.commands is None:
            object.__setattr__(self, "commands", DEFAULT_COMMANDS)


class VoiceCommandListener:
    def __init__(self, config: VoiceCommandConfig) -> None:
        self.config = config
        self._pending: Deque[str] = deque()
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._q: "queue.Queue[bytes]" = queue.Queue()
        self._last_text = ""
        self._last_cmd = ""
        self._last_cmd_ts = 0.0
        self._error: Optional[str] = None
        self._sd = None
        self._vosk = None
        self._rec = None

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            import sounddevice as sd
            import vosk
        except Exception as exc:  # pragma: no cover - optional deps
            raise RuntimeError(
                "Missing optional dependencies for voice commands. "
                "Install: pip install vosk sounddevice"
            ) from exc

        self._sd = sd
        self._vosk = vosk
        model = vosk.Model(self.config.model_path)
        self._rec = vosk.KaldiRecognizer(model, self.config.sample_rate)
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def pop_command(self) -> Optional[str]:
        with self._lock:
            if self._pending:
                return self._pending.popleft()
        return None

    def last_text(self) -> str:
        return self._last_text

    def error(self) -> Optional[str]:
        return self._error

    def _run(self) -> None:
        assert self._sd is not None
        assert self._rec is not None

        def _callback(indata, _frames, _time_info, status) -> None:
            if status:
                return
            self._q.put(bytes(indata))

        try:
            with self._sd.RawInputStream(
                samplerate=self.config.sample_rate,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=_callback,
            ):
                while self._running.is_set():
                    try:
                        data = self._q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if self._rec.AcceptWaveform(data):
                        res = json.loads(self._rec.Result())
                        text = str(res.get("text", "")).strip().lower()
                        if text:
                            self._handle_text(text)
        except Exception as exc:
            self._error = str(exc)
        finally:
            self._running.clear()

    def _handle_text(self, text: str) -> None:
        self._last_text = text
        cmd = self._match_command(text)
        if not cmd:
            return
        now = time.time()
        if cmd == self._last_cmd and (now - self._last_cmd_ts) < self.config.debounce_seconds:
            return
        self._last_cmd = cmd
        self._last_cmd_ts = now
        with self._lock:
            self._pending.append(cmd)

    def _match_command(self, text: str) -> Optional[str]:
        for cmd, phrases in self.config.commands.items():
            for phrase in phrases:
                if phrase in text:
                    return cmd
        return None
