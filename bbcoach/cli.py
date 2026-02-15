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
    p_live.add_argument(
        "--source",
        choices=["v4l2", "kinect2"],
        default="v4l2",
        help="Video source backend (default: v4l2).",
    )
    # Accept both --cam-index and legacy --camera / --camera-index.
    p_live.add_argument(
        "--cam-index",
        "--camera",
        "--camera-index",
        dest="cam_index",
        type=str,
        default="0",
        help="Camera index (e.g. 0, 1, 2) or Linux device path (e.g. /dev/video2).",
    )
    p_live.add_argument("--width", type=int, default=1280)
    p_live.add_argument("--height", type=int, default=720)
    p_live.add_argument("--depth-min", type=float, default=None, help="Optional depth clamp min (meters).")
    p_live.add_argument("--depth-max", type=float, default=None, help="Optional depth clamp max (meters).")
    p_live.add_argument("--mic", default=None, help="Prefer input device by substring (pactl source name).")
    p_live.add_argument("--pose-every-n", type=int, default=None, help="Run pose inference every N frames.")
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
    p_live.add_argument(
        "--coach-voice",
        action="store_true",
        default=None,
        help="Enable spoken coaching (TTS).",
    )
    p_live.add_argument(
        "--tts-backend",
        choices=["piper_bin", "espeak", "auto"],
        default=None,
        help="TTS backend (piper_bin|espeak|auto).",
    )

    p_desktop = sub.add_parser("desktop", help="Start desktop app")
    p_desktop.add_argument(
        "--source",
        choices=["none", "v4l2", "kinect2"],
        default="none",
        help="Initial source selection (default: none/offline).",
    )
    p_desktop.add_argument(
        "--cam-index",
        "--camera",
        "--camera-index",
        dest="cam_index",
        type=str,
        default="0",
        help="Camera index (e.g. 0, 1, 2) or Linux device path (e.g. /dev/video2).",
    )
    p_desktop.add_argument("--width", type=int, default=1280)
    p_desktop.add_argument("--height", type=int, default=720)

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

    p_scan = sub.add_parser("scan_capture", help="Capture 4-view scan (Kinect)")
    p_scan.add_argument("--user", required=True, help="User/profile name")
    p_scan.add_argument("--out", required=True, help="Sessions root directory")
    p_scan.add_argument("--width", type=int, default=1280)
    p_scan.add_argument("--height", type=int, default=720)

    p_metrics = sub.add_parser("compute_metrics", help="Compute metrics from 4-view scan")
    p_metrics.add_argument("--scan_dir", required=True, help="Scan directory (contains view_*)")

    p_merge = sub.add_parser("auto_merge", help="Auto merge 4-view scan")
    p_merge.add_argument("--scan_dir", required=True, help="Scan directory (contains view_*)")

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
        cam = int(args.cam_index) if str(args.cam_index).isdigit() else str(args.cam_index)
        if args.source in ("kinect2", "kinect"):
            cam = None
        run_live(
            profile=prof,
            camera=cam,
            width=args.width,
            height=args.height,
            voice=args.voice,
            voice_model=args.voice_model,
            coach_voice=args.coach_voice,
            tts_backend=args.tts_backend,
            source_kind=args.source,
            depth_min=args.depth_min,
            depth_max=args.depth_max,
            mic=args.mic,
            pose_every_n=args.pose_every_n,
        )
        store.save(prof)
        return 0

    if args.cmd == "desktop":
        from .ui.desktop_app import run_desktop

        cam: int | str | None
        cam = int(args.cam_index) if str(args.cam_index).isdigit() else str(args.cam_index)
        if args.source in ("kinect2", "kinect"):
            cam = None
        if args.source in ("none", "offline"):
            cam = None
        run_desktop(
            source_kind=args.source,
            camera=cam,
            width=args.width,
            height=args.height,
        )
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

    if args.cmd == "scan_capture":
        from .core.cli_scan import run_scan_capture

        return run_scan_capture(args.user, Path(args.out), width=args.width, height=args.height)

    if args.cmd == "compute_metrics":
        from .core.cli_scan import run_compute_metrics

        return run_compute_metrics(Path(args.scan_dir))

    if args.cmd == "auto_merge":
        from .core.cli_scan import run_auto_merge

        return run_auto_merge(Path(args.scan_dir))

    return 2
