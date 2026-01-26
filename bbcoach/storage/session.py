from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class SessionStore:
    root: Path

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

    def _capture_dir(self, profile_name: str, date_tag: Optional[str] = None) -> Path:
        day = date_tag or datetime.now().strftime("%Y%m%d")
        out = self.root / "captures" / profile_name / day
        out.mkdir(parents=True, exist_ok=True)
        return out

    def save_capture(
        self,
        profile_name: str,
        payload: Dict[str, Any],
        frame: Any,
        variant: str = "full",
    ) -> Path:
        # Lazy import keeps cv2 optional outside live mode.
        import cv2

        out_dir = self._capture_dir(profile_name)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        pose_key = str(payload.get("pose", "pose"))
        score = payload.get("pose_score")
        try:
            score_tag = f"{float(score):.1f}".replace(".", "p")
        except Exception:
            score_tag = "na"
        fname = f"{ts}_{pose_key}_score{score_tag}_{variant}.jpg"
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), frame)

        index_path = out_dir / "index.json"
        entry = dict(payload)
        entry.update({"file": fname, "variant": variant, "ts": ts})
        try:
            if index_path.exists():
                data = json.loads(index_path.read_text(encoding="utf-8"))
                if not isinstance(data, list):
                    data = []
            else:
                data = []
        except Exception:
            data = []
        data.append(entry)
        index_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return out_path

    def enforce_pose_storage_limit(
        self,
        profile_name: str,
        pose_key: str,
        limit_mb: float = 100.0,
        batch_size: int = 10,
        date_tag: Optional[str] = None,
    ) -> list[Path]:
        day = date_tag or datetime.now().strftime("%Y%m%d")
        date_dir = self.root / "captures" / profile_name / day
        index_path = date_dir / "index.json"
        if not index_path.exists():
            return []

        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if not isinstance(data, list):
            return []

        keep_entries = []
        removed_paths: list[Path] = []
        groups: Dict[str, Dict[str, Any]] = {}

        for row in data:
            if row.get("pose") != pose_key:
                keep_entries.append(row)
                continue
            file_name = row.get("file")
            if not file_name:
                continue
            path = date_dir / str(file_name)
            if not path.exists():
                continue
            try:
                size = path.stat().st_size
            except Exception:
                size = 0
            try:
                score = float(row.get("pose_score", 0.0))
            except Exception:
                score = 0.0
            ts = str(row.get("ts", ""))
            capture_id = str(row.get("capture_id") or file_name)
            group = groups.setdefault(
                capture_id,
                {"rows": [], "paths": [], "size": 0, "score": score, "ts": ts},
            )
            group["rows"].append(row)
            group["paths"].append(path)
            group["size"] += size
            if score < group["score"]:
                group["score"] = score
            if ts and (not group["ts"] or ts < group["ts"]):
                group["ts"] = ts

        total_bytes = sum(g["size"] for g in groups.values())
        limit_bytes = int(limit_mb * 1024 * 1024)

        group_list = list(groups.values())
        group_list.sort(key=lambda g: (g["score"], g["ts"]))
        while total_bytes > limit_bytes and group_list:
            batch = group_list[:batch_size]
            group_list = group_list[batch_size:]
            for group in batch:
                for path in group["paths"]:
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    removed_paths.append(path)
                total_bytes -= int(group.get("size", 0))

        remaining_rows = []
        for group in group_list:
            remaining_rows.extend(group["rows"])
        keep_entries.extend(remaining_rows)
        index_path.write_text(json.dumps(keep_entries, indent=2), encoding="utf-8")
        return removed_paths
