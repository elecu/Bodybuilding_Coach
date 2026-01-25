from __future__ import annotations

import argparse
from pathlib import Path

from .profile import ProfileStore
from .app import run_live
from .report import run_report


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
        run_live(profile=prof, camera=cam, width=args.width, height=args.height)
        store.save(prof)
        return 0

    if args.cmd == "report":
        prof = store.load(args.profile)
        run_report(profile=prof)
        return 0

    return 2
