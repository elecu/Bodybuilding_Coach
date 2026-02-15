from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import yaml


def load_federation_config(federation_id: str) -> Dict[str, Any]:
    root = Path(__file__).resolve().parent.parent.parent
    cfg_dir = root / "data" / "federations" / "configs"
    cfg_path = cfg_dir / f"{federation_id.lower()}.yaml"
    if not cfg_path.exists():
        return {}
    try:
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    return data
