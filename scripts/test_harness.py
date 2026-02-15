#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from bbcoach.federations.config_loader import load_federation_config
from bbcoach.metrics.body_compare import compare_metrics
from bbcoach.metrics.body_proxy import compute_body_proxy
from bbcoach.metrics.body_scoring import score_metrics
from bbcoach.storage.body_sessions import BodySessionStore


def _make_mask(scale: float = 1.0) -> np.ndarray:
    h, w = 512, 256
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, int(h * 0.55))
    axes = (int(70 * scale), int(200 * scale))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (w // 2 - 30, int(h * 0.8)), (25, 80), 0, 0, 360, 255, -1)
    cv2.ellipse(mask, (w // 2 + 30, int(h * 0.8)), (25, 80), 0, 0, 360, 255, -1)
    return mask


def main() -> None:
    profile = "test_metrics_profile"
    store = BodySessionStore.default()
    store.save_body_profile(
        profile,
        {
            "height_cm": 180,
            "weight_kg": 80,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    mask_a = _make_mask(scale=1.0)
    mask_b = _make_mask(scale=1.05)

    metrics_a = compute_body_proxy(mask_a, 180, 80, proportions=None)
    metrics_b = compute_body_proxy(mask_b, 180, 80, proportions=None)

    comp = compare_metrics(metrics_a, metrics_b)
    cfg = load_federation_config("ukbff")
    score = score_metrics(metrics_b, cfg)

    print("Metrics A regions:", list((metrics_a.get("proxy_regions") or {}).keys()))
    print("Metrics B regions:", list((metrics_b.get("proxy_regions") or {}).keys()))
    print("Compare deltas:", comp.get("region_deltas"))
    print("Score summary:", score.get("summary"))
    print("Missing areas:", score.get("missing_areas"))

    frame = np.zeros((mask_a.shape[0], mask_a.shape[1], 3), dtype=np.uint8)
    meta = {
        "profile": profile,
        "federation_id": "UKBFF",
        "category_id": "Mens Physique",
        "pose_id": "mp_front",
        "session_id": "demo_a",
    }
    store.create_session(profile, meta, frame, frame, metrics_a)
    meta["session_id"] = "demo_b"
    store.create_session(profile, meta, frame, frame, metrics_b)
    print("Saved demo sessions under:", Path("sessions") / profile / "body_sessions")


if __name__ == "__main__":
    main()
