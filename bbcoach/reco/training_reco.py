from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional


EVIDENCE_CITATIONS = [
    {
        "title": "Baz-Valle 2022 systematic review (weekly sets by muscle group)",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8884877/",
    },
    {
        "title": "Krzysztofik 2019 review (ACSM hypertrophy recommendations)",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6950543/",
    },
    {
        "title": "Schoenfeld 2016 frequency meta-analysis (>=2x/week tends to outperform 1x/week)",
        "url": "https://paulogentil.com/pdf/Effects%20of%20Resistance%20Training%20Frequency%20on%20Measures%20of%20Muscle%20Hypertrophy%20-%20A%20Systematic%20Review%20and%20Meta-Analysis.pdf",
    },
]


EXERCISE_MENU = {
    "lats": ["pull-up (various grips)", "lat pulldown", "one-arm cable lat pulldown", "chest-supported row"],
    "lateral_delts": ["cable lateral raise", "machine lateral raise", "lean-away lateral raise"],
    "rear_delts": ["reverse pec deck", "cable rear-delt flye", "chest-supported rear-delt raise"],
    "upper_chest": ["incline dumbbell press", "incline barbell press", "low-to-high cable flye"],
    "mid_chest": ["flat bench press", "dumbbell press", "machine press"],
    "quads": ["high-bar squat", "hack squat", "leg press", "leg extension"],
    "hamstrings": ["Romanian deadlift", "seated leg curl", "lying leg curl"],
    "glutes": ["hip thrust", "split squat", "cable kickback"],
    "calves": ["standing calf raise", "seated calf raise", "leg-press calf raise"],
    "biceps": ["incline curl", "bayesian cable curl", "hammer curl"],
    "triceps": ["cable pressdown", "overhead cable extension", "skull crusher"],
}


WEAKNESS_TO_MUSCLES = {
    "v_taper_ratio": ["lats", "lateral_delts"],
    "x_frame_ratio": ["lats", "lateral_delts", "rear_delts"],
    "leg_to_torso_balance": ["quads", "hamstrings", "glutes"],
    "thigh_symmetry": ["quads", "hamstrings"],
    "calf_symmetry": ["calves"],
    "arm_symmetry": ["biceps", "triceps"],
    "overall_muscularity": ["lats", "upper_chest", "lateral_delts", "quads"],
    "muscularity": ["lats", "upper_chest", "lateral_delts", "quads"],
    "balance_proportions": ["lats", "lateral_delts", "quads", "hamstrings"],
}


def _unique(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def build_training_recommendation(
    weaknesses: List[str],
    user_profile: Optional[Dict[str, object]] = None,
    kb_version: str = "v1",
) -> Dict[str, object]:
    user_profile = user_profile or {}
    muscles: List[str] = []
    for weakness in weaknesses:
        muscles.extend(WEAKNESS_TO_MUSCLES.get(weakness, []))
    muscles = _unique(muscles) or ["lats", "lateral_delts", "quads"]

    blocks = []
    for muscle in muscles:
        blocks.append(
            {
                "muscle_group": muscle,
                "weekly_sets_range": "12-20 sets/week per target muscle (advanced), adjust by recovery",
                "frequency": ">=2x/week per muscle if volume allows",
                "exercise_menu": EXERCISE_MENU.get(muscle, []),
                "progression": "Add reps or small load increases week-to-week while keeping form consistent.",
                "deload_notes": "Every 4-8 weeks consider a 30-50% volume reduction to maintain recovery.",
                "evidence": EVIDENCE_CITATIONS,
            }
        )

    summary_bullets = []
    for block in blocks[:6]:
        summary_bullets.append(
            f"Prioritize {block['muscle_group']} with {block['weekly_sets_range']} and {block['frequency']}."
        )
    if len(summary_bullets) > 10:
        summary_bullets = summary_bullets[:10]

    return {
        "kb_version": kb_version,
        "user_context": {
            "name": user_profile.get("name"),
            "prep_mode": user_profile.get("prep_mode"),
        },
        "training_blocks": blocks,
        "summary_bullets": summary_bullets,
        "citations_used": EVIDENCE_CITATIONS,
    }


def write_training_recommendation(
    out_path: Path,
    weaknesses: List[str],
    user_profile: Optional[Dict[str, object]] = None,
    kb_version: str = "v1",
) -> Dict[str, object]:
    out = build_training_recommendation(weaknesses, user_profile=user_profile, kb_version=kb_version)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
