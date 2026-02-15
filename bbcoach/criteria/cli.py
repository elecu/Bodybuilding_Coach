from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from .scoring import score_category


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_session_summary(session_dir: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"session_dir": str(session_dir)}
    for summary_path in (session_dir / "session_summary.json", session_dir / "derived" / "session_summary.json"):
        if summary_path.exists():
            loaded = _load_json(summary_path)
            if isinstance(loaded, dict):
                loaded["session_dir"] = str(session_dir)
                return loaded

    for metrics_path in (session_dir / "metrics.json", session_dir / "derived" / "metrics.json"):
        if metrics_path.exists():
            summary["metrics"] = _load_json(metrics_path)
            break

    for condition_path in (session_dir / "condition.json", session_dir / "derived" / "condition.json"):
        if condition_path.exists():
            summary["condition"] = _load_json(condition_path)
            break

    for pose_path in (session_dir / "pose_features.json", session_dir / "derived" / "pose_features.json"):
        if pose_path.exists():
            summary["pose_features"] = _load_json(pose_path)
            break

    for gates_path in (session_dir / "derived" / "quality_gates.json", session_dir / "quality_gates.json"):
        if gates_path.exists():
            summary["quality_gates"] = _load_json(gates_path)
            break

    for meta_path in (session_dir / "derived" / "meta.json", session_dir / "meta.json"):
        if meta_path.exists():
            summary["meta"] = _load_json(meta_path)
            break

    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bbcoach.criteria.cli",
        description="Score a scan session against federation criteria (proxy-based).",
    )
    p.add_argument("--session_dir", required=True, help="Directory containing session_summary.json or metrics.json")
    p.add_argument("--federation", required=True, help="Federation id (e.g., PCA, UKBFF, IFBB_PRO)")
    p.add_argument("--category", required=True, help="Category id (e.g., mens_physique)")
    p.add_argument("--out", default=None, help="Optional output JSON path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session_dir = Path(args.session_dir)
    summary = build_session_summary(session_dir)
    score = score_category(summary, args.federation, args.category)

    out_json = json.dumps(score, indent=2)
    if args.out:
        Path(args.out).write_text(out_json, encoding="utf-8")
    print(out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
