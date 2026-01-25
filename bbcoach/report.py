from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import json
import glob

import pandas as pd
import matplotlib.pyplot as plt

from .profile import UserProfile


def run_report(profile: UserProfile) -> None:
    sess_dir = Path(__file__).resolve().parent.parent / "sessions"
    files = sorted(sess_dir.glob(f"{profile.name}_*.json"))
    if not files:
        print("No sessions found yet. Use live mode and press 'S' to save snapshots.")
        return

    rows = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        data["file"] = f.name
        rows.append(data)

    df = pd.DataFrame(rows)

    # Show basic trend for pose score
    if "pose_score" in df.columns:
        df["ts"] = df["file"].str.extract(r"_(\d{8}_\d{6})")[0]
        df = df.sort_values("ts")
        plt.figure()
        plt.plot(df["pose_score"].values)
        plt.xlabel("Session")
        plt.ylabel("Pose score")
        plt.title(f"Pose score trend — {profile.name}")
        out = sess_dir / f"{profile.name}_pose_score_trend.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

    # Print last snapshot summary
    last = df.iloc[-1].to_dict()
    print("--- Latest snapshot ---")
    for k in ("federation", "category", "pose", "pose_score", "competition_date", "first_timers", "prep_mode"):
        if k in last:
            print(f"{k}: {last[k]}")
