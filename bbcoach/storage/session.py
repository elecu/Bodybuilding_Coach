from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


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
