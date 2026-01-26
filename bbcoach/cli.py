from __future__ import annotations

import argparse
from pathlib import Path

from .profile import ProfileStore
from .app import run_live
from .report import run_report
from .compare import run_compare


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bbcoach",
        description="Webcam bodybuilding coach: live posing + progress tracking.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_live = sub.add_parser("live", help="Start live coach")
    p_live.add_argument("--profile", required=True, help="Profile name")
    # Accept both --camera and the more explicit --camera-index
    p_live.add_argument(
        "--camera",
        "--camera-index",
        dest="camera",
        type=str,
        default="0",
        help="Camera index (e.g. 0, 1, 2) or Linux device path (e.g. /dev/video2)",
    )
    p_live.add_argument("--width", type=int, default=1280)
    p_live.add_argument("--height", type=int, default=720)
    p_live.add_argument(
        "--voice",
        action="store_true",
        help="Enable offline voice commands (requires a Vosk model directory).",
    )
    p_live.add_argument(
        "--voice-model",
        default=None,
        help="Path to a Vosk model directory (default: ./models/vosk if present).",
    )

    p_prof = sub.add_parser("profile", help="Profile management")
    subp = p_prof.add_subparsers(dest="action", required=True)

    p_new = subp.add_parser("new", help="Create a new profile")
    p_new.add_argument("--name", required=True)

    p_edit = subp.add_parser("edit", help="Edit a profile interactively")
    p_edit.add_argument("--name", required=True)

    p_show = subp.add_parser("show", help="Show a profile")
    p_show.add_argument("--name", required=True)

    p_report = sub.add_parser("report", help="Generate a simple progress report")
    p_report.add_argument("--profile", required=True)

    p_compare = sub.add_parser("compare", help="Create a side-by-side pose comparison")
    p_compare.add_argument("--profile", required=True)
    p_compare.add_argument("--pose", required=True, help="Pose key or display name")
    p_compare.add_argument("--date-a", default=None, help="Date YYYYMMDD (optional)")
    p_compare.add_argument("--date-b", default=None, help="Date YYYYMMDD (optional)")
    p_compare.add_argument(
        "--variant",
        default="cutout",
        choices=["full", "cutout", "story"],
        help="Capture variant to compare",
    )
    p_compare.add_argument("--out", default=None, help="Output file path (optional)")

    return p


def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    store = ProfileStore.default()

    if args.cmd == "profile":
        if args.action == "new":
            store.create(args.name)
            print(f"Created profile: {args.name}")
            return 0
        if args.action == "edit":
            store.edit_interactive(args.name)
            return 0
        if args.action == "show":
            prof = store.load(args.name)
            print(prof.model_dump_json(indent=2, by_alias=True))
            return 0

    if args.cmd == "live":
        prof = store.load(args.profile)
        cam: int | str
        cam = int(args.camera) if str(args.camera).isdigit() else str(args.camera)
        run_live(
            profile=prof,
            camera=cam,
            width=args.width,
            height=args.height,
            voice=args.voice,
            voice_model=args.voice_model,
        )
        store.save(prof)
        return 0

    if args.cmd == "report":
        prof = store.load(args.profile)
        run_report(profile=prof)
        return 0

    if args.cmd == "compare":
        run_compare(
            profile_name=args.profile,
            pose=args.pose,
            date_a=args.date_a,
            date_b=args.date_b,
            variant=args.variant,
            out_path=args.out,
        )
        return 0

    return 2
