from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

from ..utils.cachepi import mirror_file
from .session_paths import SessionPaths

@dataclass
class SessionStore:
    root: Path
    _capture_dirs: Dict[str, Path] = field(default_factory=dict, init=False, repr=False)

    @staticmethod
    def default() -> "SessionStore":
        root = Path(__file__).resolve().parent.parent.parent / "sessions"
        root.mkdir(parents=True, exist_ok=True)
        return SessionStore(root=root)

    def save_snapshot(self, profile_name: str, payload: Dict[str, Any]) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = self.root / f"{profile_name}_{ts}.json"
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    def _paths(self) -> SessionPaths:
        return SessionPaths(root=self.root)

    def _pose_session_dir(self, profile_name: str, payload: Dict[str, Any]) -> Path:
        capture_id = str(payload.get("capture_id") or "")
        if capture_id and capture_id in self._capture_dirs:
            return self._capture_dirs[capture_id]
        pose_key = str(payload.get("pose") or "pose")
        session_dir = self._paths().new_pose_session_dir(profile_name, pose_key)
        if capture_id:
            self._capture_dirs[capture_id] = session_dir
        self._update_pose_meta(session_dir, payload)
        return session_dir

    def _update_pose_meta(self, session_dir: Path, payload: Dict[str, Any]) -> Path:
        derived_dir = session_dir / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)
        meta_path = derived_dir / "meta.json"
        try:
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            else:
                meta = {}
        except Exception:
            meta = {}
        meta.update(payload)
        if "pose_id" not in meta and "pose" in meta:
            meta["pose_id"] = meta.get("pose")
        if "created_at" not in meta:
            meta["created_at"] = datetime.now().isoformat(timespec="seconds")
        meta["session_dir"] = str(session_dir)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return meta_path

    def save_capture(
        self,
        profile_name: str,
        payload: Dict[str, Any],
        frame: Any,
        variant: str = "full",
    ) -> Path:
        # Lazy import keeps cv2 optional outside live mode.
        import cv2

        if "capture_id" not in payload:
            payload["capture_id"] = uuid.uuid4().hex
        capture_id = str(payload.get("capture_id") or uuid.uuid4().hex)
        session_dir = self._pose_session_dir(profile_name, payload)
        media_dir = session_dir / "media"
        media_dir.mkdir(parents=True, exist_ok=True)
        pose_key = str(payload.get("pose", "pose"))
        score = payload.get("pose_score")
        try:
            score_tag = f"{float(score):.1f}".replace(".", "p")
        except Exception:
            score_tag = "na"
        fname = f"{capture_id}_{pose_key}_score{score_tag}_{variant}.jpg"
        out_path = media_dir / fname
        cv2.imwrite(str(out_path), frame)
        mirror_file(out_path)

        meta_path = self._update_pose_meta(session_dir, payload)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = dict(payload)
        media = meta.get("media") or {}
        media[str(variant)] = str(out_path.relative_to(session_dir).as_posix())
        meta["media"] = media
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return out_path

    def enforce_pose_storage_limit(
        self,
        profile_name: str,
        pose_key: str,
        limit_mb: float = 100.0,
        batch_size: int = 10,
        date_tag: Optional[str] = None,
    ) -> list[Path]:
        day = date_tag or datetime.now().strftime("%Y-%m-%d")
        date_dir = self.root / profile_name / day / "poses"
        if not date_dir.exists():
            return []

        removed_paths: list[Path] = []
        sessions: list[Dict[str, Any]] = []
        for session_dir in date_dir.iterdir():
            if not session_dir.is_dir():
                continue
            meta_path = session_dir / "derived" / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if str(meta.get("pose") or meta.get("pose_id") or "") != str(pose_key):
                continue
            try:
                score = float(meta.get("pose_score", 0.0))
            except Exception:
                score = 0.0
            ts = str(meta.get("capture_id") or meta.get("created_at") or session_dir.name)
            size = 0
            for path in session_dir.rglob("*"):
                try:
                    if path.is_file():
                        size += path.stat().st_size
                except Exception:
                    continue
            sessions.append({"dir": session_dir, "size": size, "score": score, "ts": ts})

        total_bytes = sum(s["size"] for s in sessions)
        limit_bytes = int(limit_mb * 1024 * 1024)

        sessions.sort(key=lambda s: (s["score"], s["ts"]))
        idx = 0
        while total_bytes > limit_bytes and idx < len(sessions):
            batch = sessions[idx : idx + batch_size]
            idx += batch_size
            for session in batch:
                session_dir = session["dir"]
                for path in session_dir.rglob("*"):
                    if path.is_file():
                        removed_paths.append(path)
                try:
                    import shutil

                    shutil.rmtree(session_dir, ignore_errors=True)
                except Exception:
                    pass
                total_bytes -= int(session.get("size", 0))
        return removed_paths
