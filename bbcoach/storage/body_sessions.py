from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List

from ..utils.cachepi import mirror_file

@dataclass
class BodySessionStore:
    root: Path

    @staticmethod
    def default() -> "BodySessionStore":
        root = Path(__file__).resolve().parent.parent.parent / "sessions"
        root.mkdir(parents=True, exist_ok=True)
        return BodySessionStore(root=root)

    def _profile_root(self, profile_name: str) -> Path:
        out = self.root / profile_name
        out.mkdir(parents=True, exist_ok=True)
        return out

    def body_profile_path(self, profile_name: str) -> Path:
        return self._profile_root(profile_name) / "body_profile.json"

    def body_sessions_root(self, profile_name: str) -> Path:
        out = self._profile_root(profile_name) / "body_sessions"
        out.mkdir(parents=True, exist_ok=True)
        return out

    def load_body_profile(self, profile_name: str) -> Optional[Dict[str, Any]]:
        path = self.body_profile_path(profile_name)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save_body_profile(self, profile_name: str, data: Dict[str, Any]) -> Path:
        path = self.body_profile_path(profile_name)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        mirror_file(path)
        return path

    def create_session(
        self,
        profile_name: str,
        metadata: Dict[str, Any],
        frame_bgr: Any,
        cutout_bgr: Optional[Any],
        metrics: Dict[str, Any],
    ) -> Path:
        # Lazy import keeps cv2 optional outside live mode.
        import cv2

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        session_id = metadata.get("session_id") or ts
        session_dir = self.body_sessions_root(profile_name) / str(session_id)
        assets_dir = session_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        full_path = assets_dir / "full.jpg"
        cv2.imwrite(str(full_path), frame_bgr)
        mirror_file(full_path)

        cutout_path = None
        if cutout_bgr is not None:
            cutout_path = assets_dir / "cutout.jpg"
            cv2.imwrite(str(cutout_path), cutout_bgr)
            mirror_file(cutout_path)

        metrics_path = session_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        session_data = dict(metadata)
        session_data.update(
            {
                "session_id": session_id,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "assets": {
                    "full": str(full_path),
                    "cutout": str(cutout_path) if cutout_path else None,
                },
                "metrics_path": str(metrics_path),
            }
        )
        session_path = session_dir / "session.json"
        session_path.write_text(json.dumps(session_data, indent=2), encoding="utf-8")
        return session_path

    def list_sessions(self, profile_name: str) -> List[Dict[str, Any]]:
        root = self.body_sessions_root(profile_name)
        sessions: List[Dict[str, Any]] = []
        if not root.exists():
            return sessions
        for session_dir in root.iterdir():
            if not session_dir.is_dir():
                continue
            session_path = session_dir / "session.json"
            if not session_path.exists():
                continue
            try:
                data = json.loads(session_path.read_text(encoding="utf-8"))
                data["_path"] = str(session_path)
                sessions.append(data)
            except Exception:
                continue
        sessions.sort(key=lambda d: str(d.get("created_at", "")), reverse=True)
        return sessions

    def load_metrics(self, metrics_path: str) -> Optional[Dict[str, Any]]:
        try:
            path = Path(metrics_path)
            if not path.exists():
                return None
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def save_scan(
        self,
        profile_name: str,
        session_id: str,
        points: Any,
        colors: Optional[Any] = None,
    ) -> Dict[str, Optional[str]]:
        from ..vision.pointcloud import write_pcd, write_ply

        session_dir = self.body_sessions_root(profile_name) / str(session_id)
        scans_dir = session_dir / "scans"
        scans_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        scan_dir = scans_dir / ts
        scan_dir.mkdir(parents=True, exist_ok=True)
        ply_path = scan_dir / "output_person.ply"
        pcd_path = scan_dir / "output_person.pcd"

        write_ply(ply_path, points, colors)
        write_pcd(pcd_path, points)
        mirror_file(ply_path)
        mirror_file(pcd_path)

        session_path = session_dir / "session.json"
        if session_path.exists():
            try:
                data = json.loads(session_path.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        else:
            data = {
                "session_id": str(session_id),
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "assets": {},
                "metrics_path": None,
                "scan_only": True,
            }

        scans = data.get("scans") or []
        scans.append(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "dir": str(scan_dir),
                "ply": str(ply_path),
                "pcd": str(pcd_path),
                "points": int(getattr(points, "shape", [0])[0] or 0),
            }
        )
        data["scans"] = scans
        session_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return {"ply": str(ply_path), "pcd": str(pcd_path)}
