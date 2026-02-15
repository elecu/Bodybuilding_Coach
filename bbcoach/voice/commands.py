from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import os
import queue
import subprocess
import threading
import time
import unicodedata
from typing import Deque, Dict, Optional, Tuple


DEFAULT_COMMANDS: Dict[str, Tuple[str, ...]] = {
    "scan_3d": (
        "scan 3d",
        "scan three d",
        "scan three dee",
    ),
    "next_pose": (
        "next pose",
        "pose next",
        "go next",
        "next next",
        "next one",
        "next",
        "siguiente pose",
        "pose siguiente",
        "sigue",
        "siguiente",
    ),
    "prev_pose": (
        "prev pose",
        "previous pose",
        "pose previous",
        "go back",
        "back",
        "previous previous",
        "previous one",
        "previous",
        "anterior pose",
        "pose anterior",
        "atras",
        "anterior",
    ),
    "start_scan": (
        "start scan",
        "start session",
        "start metrics",
        "start capture",
    ),
    "stop_scan": (
        "stop scan",
        "stop session",
        "stop metrics",
        "stop capture",
    ),
    "save_model": (
        "save model",
        "save 3d",
        "save mesh",
    ),
    "open_metrics": (
        "metrics",
        "open metrics",
        "show metrics",
        "tab metrics",
        "go metrics",
    ),
    "open_posing": (
        "posing",
        "open posing",
        "show posing",
        "tab posing",
        "go posing",
    ),
}


@dataclass(frozen=True)
class VoiceCommandConfig:
    model_path: str
    sample_rate: int = 16000
    debounce_seconds: float = 1.2
    mic_prefer: Optional[str] = None
    mic_fallback: Optional[str] = None
    source_kind: str = "v4l2"
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
        self._device: Optional[int] = None
        self._norm_commands: Dict[str, Tuple[str, ...]] = {
            cmd: tuple(self._normalize_text(p) for p in phrases)
            for cmd, phrases in self.config.commands.items()
        }

    def _build_vosk_grammar(self) -> str:
        items: list[str] = []
        seen: set[str] = set()
        for phrases in self._norm_commands.values():
            for phrase in phrases:
                p = phrase.strip()
                if p and p not in seen:
                    seen.add(p)
                    items.append(p)
        return json.dumps(items, ensure_ascii=False)

    def _list_pactl_sources(self) -> list[str]:
        try:
            output = subprocess.check_output(
                ["pactl", "list", "sources", "short"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return []
        sources: list[str] = []
        for line in output.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                sources.append(parts[1])
        return sources

    def _select_pulse_source(self, prefer: Optional[str], fallback: Optional[str]) -> Optional[str]:
        sources = self._list_pactl_sources()
        if not sources:
            return None

        def _is_kinect(name: str) -> bool:
            n = name.lower()
            return ("xbox" in n) or ("nui" in n) or ("kinect" in n)

        def _is_monitor(name: str) -> bool:
            return ".monitor" in name.lower()

        def _match(needle: str) -> Optional[str]:
            needle_l = needle.lower()
            for name in sources:
                if needle_l in name.lower():
                    return name
            return None

        if prefer:
            hit = _match(prefer)
            if hit:
                return hit
        if fallback:
            hit = _match(fallback)
            if hit:
                return hit

        source_kind = (self.config.source_kind or "").strip().lower()
        live_inputs = [s for s in sources if not _is_monitor(s)]

        if source_kind in ("kinect2", "kinect"):
            for name in live_inputs:
                if _is_kinect(name):
                    return name
            return None

        # Webcam mode: explicitly avoid Kinect when possible.
        non_kinect = [s for s in live_inputs if not _is_kinect(s)]
        if non_kinect:
            # Prefer obvious mic sources first.
            preferred_tokens = (
                "webcam",
                "camera",
                "internal",
                "built-in",
                "builtin",
                "analog-stereo",
                "usb",
                "mic",
            )
            for token in preferred_tokens:
                for name in non_kinect:
                    if token in name.lower():
                        return name
            return non_kinect[0]

        if live_inputs:
            return live_inputs[0]
        return None

    def _resolve_sounddevice_input(self, source_name: str) -> Optional[int]:
        if self._sd is None:
            return None
        try:
            devices = self._sd.query_devices()
        except Exception:
            return None
        needle = source_name.lower()
        for idx, dev in enumerate(devices):
            try:
                if dev.get("max_input_channels", 0) > 0 and needle in str(dev.get("name", "")).lower():
                    return idx
            except Exception:
                continue
        return None

    def start(self) -> None:
        if self._thread is not None:
            return
        try:
            import vosk
        except Exception as exc:  # pragma: no cover - optional deps
            raise RuntimeError(
                "Missing optional dependencies for voice commands. "
                "Install: pip install vosk sounddevice"
            ) from exc

        prefer = self.config.mic_prefer
        fallback = self.config.mic_fallback
        selected_source = self._select_pulse_source(prefer, fallback)
        if selected_source:
            os.environ["PULSE_SOURCE"] = selected_source
            print(f"[bbcoach] Using audio input: {selected_source}")
        elif prefer or fallback:
            print("[bbcoach] Requested mic not found; using default input.")

        try:
            import sounddevice as sd
        except Exception as exc:  # pragma: no cover - optional deps
            raise RuntimeError(
                "Missing optional dependencies for voice commands. "
                "Install: pip install vosk sounddevice"
            ) from exc

        self._sd = sd
        self._vosk = vosk
        if selected_source:
            self._device = self._resolve_sounddevice_input(selected_source)
        model = vosk.Model(self.config.model_path)
        grammar = self._build_vosk_grammar()
        try:
            self._rec = vosk.KaldiRecognizer(model, self.config.sample_rate, grammar)
            print(f"[bbcoach] Voice grammar enabled ({len(json.loads(grammar))} phrases).")
        except TypeError:
            # Fallback for older vosk builds without grammar constructor.
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
            kwargs = dict(
                samplerate=self.config.sample_rate,
                blocksize=8000,
                dtype="int16",
                channels=1,
                callback=_callback,
            )
            if self._device is not None:
                kwargs["device"] = self._device
            with self._sd.RawInputStream(**kwargs):
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
                    else:
                        pres = json.loads(self._rec.PartialResult())
                        ptext = str(pres.get("partial", "")).strip().lower()
                        if ptext:
                            self._handle_text(ptext)
        except Exception as exc:
            self._error = str(exc)
        finally:
            self._running.clear()

    def _normalize_text(self, text: str) -> str:
        norm = unicodedata.normalize("NFKD", text).lower()
        norm = "".join(ch for ch in norm if not unicodedata.combining(ch))
        norm = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in norm)
        return " ".join(norm.split())

    def _handle_text(self, text: str) -> None:
        norm_text = self._normalize_text(text)
        self._last_text = norm_text
        cmd = self._match_command(norm_text)
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
        for cmd, phrases in self._norm_commands.items():
            for phrase in phrases:
                if phrase in text:
                    return cmd
        return None
