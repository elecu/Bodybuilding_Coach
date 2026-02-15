from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


def _alpha_suffix(idx: int) -> str:
    # 0 -> a, 25 -> z, 26 -> aa, ...
    n = idx + 1
    out = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out.append(chr(ord("a") + rem))
    return "".join(reversed(out))


@dataclass(frozen=True)
class SessionPaths:
    root: Path  # sessions root

    @staticmethod
    def default() -> "SessionPaths":
        root = Path(__file__).resolve().parent.parent.parent / "sessions"
        root.mkdir(parents=True, exist_ok=True)
        return SessionPaths(root=root)

    def day_dir(self, user: str, date_yyyy_mm_dd: Optional[str] = None) -> Path:
        day = date_yyyy_mm_dd or time.strftime("%Y-%m-%d")
        out = self.root / user / day
        out.mkdir(parents=True, exist_ok=True)
        return out

    def new_pose_session_dir(
        self,
        user: str,
        pose_id: str,
        date_yyyy_mm_dd: Optional[str] = None,
    ) -> Path:
        day = self.day_dir(user, date_yyyy_mm_dd)
        ts = time.strftime("%H-%M-%S")
        base = day / "poses" / f"{ts}_pose_{pose_id}"
        session_dir = self.ensure_unique_dir(base)
        self._init_subdirs(session_dir, with_views=False)
        return session_dir

    def new_scan3d_session_dir(
        self,
        user: str,
        mode: Literal["locked", "free"],
        date_yyyy_mm_dd: Optional[str] = None,
    ) -> Path:
        day = self.day_dir(user, date_yyyy_mm_dd)
        ts = time.strftime("%H-%M-%S")
        base = day / "metrics" / f"{ts}_scan3d_{mode}"
        session_dir = self.ensure_unique_dir(base)
        self._init_subdirs(session_dir, with_views=True)
        return session_dir

    def ensure_unique_dir(self, base: Path) -> Path:
        base.parent.mkdir(parents=True, exist_ok=True)
        if not base.exists():
            base.mkdir(parents=True, exist_ok=False)
            return base
        idx = 0
        while True:
            suffix = _alpha_suffix(idx)
            candidate = Path(f"{base}_{suffix}")
            if not candidate.exists():
                candidate.mkdir(parents=True, exist_ok=False)
                return candidate
            idx += 1

    def _init_subdirs(self, session_dir: Path, with_views: bool) -> None:
        for name in ("raw", "derived", "media", "reports", "exports"):
            (session_dir / name).mkdir(parents=True, exist_ok=True)
        if with_views:
            for angle in (0, 90, 180, 270):
                (session_dir / "raw" / f"view_{angle:03d}").mkdir(parents=True, exist_ok=True)
